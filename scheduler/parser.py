"""自然语言 → 日程事件（文字的入口）

【这一层要解决什么】
    用户说："下周四晚上七点在东区 A201 开班会，大概一小时"
    要变成：{title: 班会, start: 2026-10-22T19:00, duration_min: 60, location: 东区 A201}

【最关键的一个设计决定：日期**不由模型算**】
    模型在这件事上有两个能力层级：
        ① 认出"下周四晚上七点"是一段时间表达   ← 它擅长
        ② 算出"下周四"到底是几月几号           ← 它不可靠（小模型尤其）

    实测过：直接问模型"下周四是几号"，它会自信地给出错误答案 ——
    因为它不知道今天是几号。而它给出的错误日期**看起来完全合理**，
    人一眼看不出问题。

    所以本模块把职责切开：

        模型  →  只做"语言到结构化"的搬运：
                 把"下周四晚上七点"标成 {expr_type: "weekday", weekday: 4, week: "next", time: "19:00"}
        代码  →  拿着这个"零件"去算真正的日期（有 today 做基准，算得出来）

    ⚠️ 这是**纯语言解析**，不查冲突、不落库。冲突交给 conflict.py，
       因为那是裁决，属于内核的地盘。别把两件事混在一起。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from .model import WEEKDAY_NAMES, Event, ValidationError, make_id

# 中文数字，用于"晚上七点""三点半"
CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 一天里的时段词 → 小时的修正倾向
PERIOD_HINTS = {
    "凌晨": (0, 5),
    "早上": (6, 9), "早晨": (6, 9), "上午": (8, 11),
    "中午": (11, 13),
    "下午": (13, 17),
    "傍晚": (17, 19),
    "晚上": (18, 22), "今晚": (18, 22), "夜里": (20, 23),
}


def _cn_number(text: str) -> int | None:
    """把中文数字转成阿拉伯数字。只处理 0~59 这种日常范围。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "半":                               # "半小时""点半"里的半
        return 30
    if text == "十":
        return 10
    if text.startswith("十"):                      # 十三
        rest = _cn_number(text[1:])
        return 10 + rest if rest is not None else None
    if "十" in text:                               # 二十、二十五
        head, _, tail = text.partition("十")
        tens = CN_DIGITS.get(head, 1) if head else 1
        ones = CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    if len(text) == 1:
        return CN_DIGITS.get(text)
    return None


def _duration_minutes(num_text: str, unit: str) -> int | None:
    """把"N 个单位"换算成分钟。

    【为什么要单独写一个函数】"半小时"和"半个小时"里的"半"，
    用 _cn_number 会得到 30 —— 但 30 的含义取决于单位：
        "半小时"   → 30 分钟   （30 个单位就是分钟）
        "半个小时" → 30 分钟
        "半钟头"   → 30 分钟   （半 × 60 = 30）
        如果哪天真出现"半天"，那应该是 720 分钟，而不是 30。
    所以"数字"和"单位"必须一起看，不能分开算。
    """
    value = _cn_number(num_text)
    if value is None:
        return None
    is_hour = "小时" in unit or "钟头" in unit or unit == "h"
    if num_text.strip() == "半":
        return 30 if is_hour else value      # 半小时=30分；半分=30秒（不常见，取 30 分兜底）
    return value * 60 if is_hour else value


def _normalize_clock(hour: int, minute: int, period: str | None,
                     explicit: bool = False) -> tuple[int, int]:
    """把"下午三点"这种讲法规范化成 24 小时制。

    规则（人的直觉，不是数学）：
        - **写了冒号的（8:00 / 15:30）一律照原样**，不猜 —— 那是 24 小时制写法
        - 说了时段（下午/晚上）→ 按该时段调整
        - 没说时段时：
            1~7 点  → 下午（13~19）—— 因为"三点开会"说的一定不是凌晨三点
            8~11 点 → 上午
            12 点   → 中午 12

    【踩过的坑】第一版漏了 `explicit` 这条，于是课表里的 "8:00" 被"1~7 点算下午"
    的规则改成了 20:00 —— **早上八点的课变成了晚上八点**。
    这类错误特别危险：解析结果看上去完全合理，只有学生本人知道不对。
    """
    minute = max(0, min(59, minute))
    if explicit:
        return hour % 24, minute

    if period is None:
        # 12 小时制的口语：1~7 点几乎一定指下午/晚上
        if 1 <= hour <= 7:
            hour += 12
    elif period in ("下午", "傍晚") and hour < 12:
        hour += 12
    elif period in ("晚上", "今晚", "夜里") and hour < 12:
        hour += 12
    elif period in ("中午",) and hour < 11:
        hour += 12
    return hour % 24, minute


# ═══════════════════════════ 解析结果 ═══════════════════════════

@dataclass
class _Span:
    """一个"要挖掉的区间" + 它代表的值。

    用它替代 re.Match 是因为有时候区间不是正则直接匹配出来的，
    而是"匹配结果再往前扩了几个字"（比如"食品316"里的"食品"）。
    硬凑一个 match 对象反而更难读。
    """

    start: int
    end: int
    value: str

    def span(self, group: int = 0) -> tuple[int, int]:
        return (self.start, self.end)

    def group(self, index: int = 0) -> str:
        return self.value


@dataclass
class ParsedEvent:
    """从一段文字里抽出来的一件事（还没落库）。

    `missing` 与 `note` 是这个类存在的主要理由：
    解析必然有抽不全的时候，**如实告诉用户缺什么，比猜一个填上去强**。
    猜错了用户不会发现，直到他按这个日程去开会。
    """

    title: str = ""
    start: datetime | None = None
    duration_min: int = 60
    location: str = ""
    category: str = "other"
    prep_min: int = 0                                      # 前置准备时间（"提前一小时准备"）
    weekdays: list[int] = field(default_factory=list)      # 非空表示"每周重复"
    until: str = ""                                        # 重复到哪天（可空 → 用默认学期末）
    missing: list[str] = field(default_factory=list)       # 缺哪些关键信息
    note: str = ""                                         # 给人看的解析说明
    conflict: dict | None = None                           # 落库前的裁决结果

    @property
    def ok(self) -> bool:
        return not self.missing and self.start is not None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"] = self.start.strftime("%Y-%m-%dT%H:%M") if self.start else None
        data["ok"] = self.ok
        data["weekday_label"] = (WEEKDAY_NAMES[self.start.isoweekday()]
                                 if self.start else "")
        return data


# ═══════════════════════════ 规则解析（不依赖模型）═══════════════════════════

TIME_RE = re.compile(
    r"(?P<period>凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|今晚|夜里)?\s*"
    r"(?P<hour>\d{1,2}|[零一二两三四五六七八九十]{1,3})"
    # (?!\d) 很重要：没有它，"A301" 里的 "30" 会被当成 30 点
    r"(?!\d)"
    r"\s*(?:[:：点时]\s*(?P<minute>\d{1,2}|[零一二两三四五六七八九十]{1,3})?\s*分?)?"
)
# "半"是独立处理的（三点半 / 七点半）
HALF_RE = re.compile(r"(?:[:：点时])\s*半")
# 时间是"几点"这件事的证据：有冒号、有点/时、或者带了时段词。
# 没有这三样，就不该当成时间 —— 否则 "周五" 的"五"、"A301" 的"30" 全会被误认成钟点。
TIME_EVIDENCE_RE = re.compile(r"[:：点时]|凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|今晚|夜里")
DATE_CN_RE = re.compile(r"(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]?")
DATE_ISO_RE = re.compile(r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})")
WEEKDAY_RE = re.compile(r"(?P<week>下下|下|这|本|上)?\s*(?:周|星期|礼拜)\s*(?P<day>[一二三四五六日天末]|七)")
DURATION_RE = re.compile(
    r"(?:大概|大约|约|差不多)?\s*(?P<num>\d{1,3}|半|[零一二两三四五六七八九十]{1,3})\s*"
    r"(?P<unit>个?小时|分钟|个?钟头|h|min)"
)
# 前置准备时间（"提前一小时准备"）。这跟事件时长是两回事，必须分开认。
PREP_RE = re.compile(
    r"(?:提前|事先|预先|需要)\s*(?P<num>\d{1,3}|半|[零一二两三四五六七八九十]{1,3})\s*"
    r"(?P<unit>个?小时|分钟|个?钟头|h|min)"
)
PREP_TAIL_RE = re.compile(r"\s*(?:准备|出发|到|到达|集合)")
LOCATION_RE = re.compile(r"(?:在|地点[:：]?\s*|@)\s*(?P<loc>[^，。；,;\n]{2,20})")
# 地点后面这些词一出现，说明地点已经说完了（"在实验楼开组会" → 实验楼）
LOCATION_STOP = re.compile(r"(?:开|举行|进行|上课|集合|碰头|见|有个|有一场|签到|门口等)")

# 事件名后面这些"尾巴"要剪掉（"高等数学，到 12 月底" → "高等数学"）
# 它们不是事件的一部分，而是对事件的补充说明。
TITLE_TAIL_RE = re.compile(
    r"[,，]?\s*(?:到|至|截止|截止到|持续到|一直到)\s*[^,，。;；]{0,12}$"   # "到 12 月底"
    r"|(?:请|记得|别忘[了]?)\s*[^,，。;；]{0,8}$"                        # "请提前到"
)

# 【踩过的坑】第一版只用了"西区""活动室"这种短词，于是"西区活动室"会被
# 更短的"活动室"抢先匹配，剩下"西区"混进标题里。后来加了长词 + 按长度倒序，
# 但真实写法是千变万化的（"西区活动室"这种复合词列不完）。
# 现在改成**兜住"区"前缀**：短词命中后，如果前面紧挨着一个"X区"，就一起收进来。
# 列不完的东西，就不要靠枚举去列。
AREA_PREFIX_RE = re.compile(r"([东西南北中][区校])$")

# 校园场景里常见的地点词
KNOWN_PLACES = ("东区", "西区", "南区", "北区", "体育馆", "操场", "图书馆", "食堂",
                "活动室", "线上", "腾讯会议", "会议室", "实验楼", "教学楼", "机房")
# 这些词**本身有歧义** —— 它们可能只是标题的一部分（"图书馆使用指南"），
# 所以只有在前面有介词（在/@/到/去）时才当成地点。
# 而 东区/体育馆/操场 这类出现在句子里几乎一定指地点，不需要介词。
AMBIGUOUS_PLACES = frozenset({"图书馆", "食堂", "活动室", "会议室", "实验楼",
                              "教学楼", "机房", "线上", "腾讯会议"})
# 常见地点后面紧跟的房间号（A301 / B202 / D105）
ROOM_RE = re.compile(r"[A-Za-z]\s?\d{1,4}")

WEEKDAY_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7, "末": 6}

# 这类词一出现，说明这是一件"每周都发生"的事
RECURRING_HINTS = ("每周", "每周的", "例会", "每周一", "每周二", "每周三", "每周四",
                   "每周五", "每周六", "每周日", "固定", "每次")
# 标题里要剥掉的尾巴（时间/地点已经单独抽出去了）
TITLE_TAIL = re.compile(r"^\s*")   # 已废弃：标题现在靠"挖洞"生成，不再做后缀裁剪
TITLE_TAIL = re.compile(
    r"(?:，|,)?\s*(?:大概|大约|约|差不多)?\s*\d{1,3}\s*(?:个?小时|分钟|个?钟头|h|min).*$"
)


def resolve_date(expr: dict, today: date) -> date | None:
    """把"日期零件"算成真实日期。

    这是本模块的核心 —— **日期一律由这里算，模型不许算**。
    """
    if not isinstance(expr, dict):
        return None
    kind = expr.get("type")
    if kind == "absolute":                       # 3月5日 / 2026-03-05
        month, day = expr.get("month"), expr.get("day")
        if not (month and day):
            return None
        year = expr.get("year") or today.year
        try:
            candidate = date(int(year), int(month), int(day))
        except ValueError:
            return None
        # 只写了"3月20日"这种没写年份的，算成"最近的那个该日期"。
        # 已经过去了就顺延到明年 —— 排日程的人说一个过去的日期，
        # 意思一定是下一次，不可能是"去年那次"。
        # （原来写的"超过半年才算明年"是个多余的复杂度：半年以内的过去日期
        #   既不是要排的事、又让人琢磨，不如统一顺延。）
        if candidate < today:
            try:
                candidate = candidate.replace(year=candidate.year + 1)
            except ValueError:                   # 2月29日遇到平年
                candidate = candidate.replace(year=candidate.year + 1, day=28)
        return candidate

    if kind == "relative":                       # 今天 / 明天 / 后天
        offset = int(expr.get("offset") or 0)
        return today + timedelta(days=offset)

    if kind == "weekday":                        # 这周三 / 下周四
        target = int(expr.get("weekday") or 0)
        if not 1 <= target <= 7:
            return None
        week = expr.get("week") or "this"
        monday = today - timedelta(days=today.isoweekday() - 1)
        shift = {"this": 0, "next": 1, "after_next": 2, "last": -1}.get(week, 0)
        candidate = monday + timedelta(days=shift * 7 + target - 1)
        # 说"这周三"但今天已经周四了 —— 人指的是下周，不是昨天
        if week == "this" and candidate < today:
            candidate += timedelta(days=7)
        return candidate
    if kind == "weekdays":                       # 每周一和周四
        return None                              # 重复规则，不产生单一日期

    return None


def parse_text(text: str, today: date | None = None) -> list[ParsedEvent]:
    """纯规则解析：不调用任何模型。

    能处理的就是校园通知里最常见的那些句式：
        "周三下午两点在实验楼有个组会"
        "每周一 8:00 高等数学 东区 A301，到 6 月底"
        "10月20号 15:00-17:00 面试（线上），要提前一小时准备"
    超出这个范围的（"下下周找个时间聊一下"）抽不全，就如实报缺什么。

    **为什么先做规则解析，而不是直接上模型**
        1. 快、免费、离线 —— 而且是确定性的，能被测试
        2. 它给模型提供了一个"兜底"：模型挂了 / 没启动，这页还能用
        3. 更重要的是，它是第 5 天那个 A/B 实验的**对照组**之一：
           规则解析 vs 纯模型 vs 模型+内核，三者的正确率可以直接比
    """
    today = today or date.today()
    text = (text or "").strip()
    if not text:
        return []

    # 一段文字可能包含多件事（一行一件 / 分号分隔）
    chunks = [c.strip() for c in re.split(r"[;\n；]+", text) if c.strip()]
    return [_parse_one(chunk, today) for chunk in chunks]


def _parse_one(chunk: str, today: date) -> ParsedEvent:
    """把一句话解析成一件日程。

    【结构：三阶段，不要混着做】
        ① 抽取：日期零件 / 钟点 / 时长 / 地点 / 星期几 —— 各自独立地找
        ② 算标题：把①占用的**字符区间**挖掉，剩下的就是标题
        ③ 组装：用 resolve_date 把日期零件算成真日期，缺什么如实报什么

    【为什么要分成三阶段（这是返工两次换来的）】
        第一版是"边解析边在原文上删掉已识别的部分"，结果是**错误会级联**：
            `A301` 被时间正则削成 `A` → 于是"东区 A301"这个地点再也匹配不上
            → 标题留下半截 "东区 A 开班会"。
        一个字段解析错了，后面的字段就全都建立在坏数据上。
        拆开之后，"哪些字符已经被别的字段占了"变成一份显式的 spans 列表，
        标题只负责"挖洞"，不再参与任何判断。
    """
    item = ParsedEvent()
    text = re.sub(r"\s+", " ", chunk).strip()

    # ═══════════ 阶段①：抽取 ═══════════

    # ── 星期几
    weekday_matches = list(WEEKDAY_RE.finditer(text))
    weekday_spans = [m.span() for m in weekday_matches]
    weekdays = sorted({WEEKDAY_MAP[m.group("day")] for m in weekday_matches
                       if WEEKDAY_MAP.get(m.group("day"))})
    recurring = any(hint in text for hint in RECURRING_HINTS)

    # ── 日期零件
    date_expr: dict | None = None
    relative_match = None
    if weekdays:
        # 周期词取"第一个"匹配里的：一段话里混用"下周三/这周五"很少见，
        # 与其猜，不如取第一个、然后在界面上让用户确认
        raw_week = weekday_matches[0].group("week") or ""
        week_token = {"下下": "after_next", "下": "next", "上": "last"}.get(raw_week, "this")
        if recurring or len(weekdays) > 1:
            date_expr = {"type": "weekdays", "weekdays": weekdays}
            item.weekdays = weekdays
        else:
            date_expr = {"type": "weekday", "weekday": weekdays[0], "week": week_token}
    else:
        iso = DATE_ISO_RE.search(text)
        cn = DATE_CN_RE.search(text)
        if iso:
            date_expr = {"type": "absolute", "year": int(iso.group("y")),
                         "month": int(iso.group("m")), "day": int(iso.group("d"))}
        elif cn:
            date_expr = {"type": "absolute", "month": int(cn.group("month")),
                         "day": int(cn.group("day"))}
        else:
            # "大后天" 必须排在 "后天" 前面，否则 3 天的偏移会被吃成 2 天
            for word, offset in (("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0)):
                relative_match = re.search(word, text)
                if relative_match:
                    date_expr = {"type": "relative", "offset": offset}
                    break

    # ═══════════ 阶段①：钟点 ═══════════
    # 【这个循环改了五版，把踩过的坑按顺序记下来 —— 这是一段真实的调试史】
    #   ① 只认"第一个能匹配上的数字" → "周五三点" 把"五"当成了 5 点
    #   ② 想用 (?<![\u4e00-\u9fff]) 禁止"从汉字中间起匹配"
    #      → 把**所有**中文数字钟点全禁掉了（"晚上七点"的"七"前面是"上"）。
    #        中文数字前面几乎总是汉字，这条禁令等于删掉这个功能。
    #   ③ 想用 _cn_number(...) is None 过滤垃圾匹配
    #      → "五三点" 的 hour 是"五三"，能滤掉；但 finditer **不重叠**，
    #        位置 1 的匹配吃掉了 1~4，真正在位置 2 的"三点"根本没机会被匹配。
    #        过滤器滤掉了垃圾，也顺手丢了正确答案。
    #   ④ 想先把"周几"挖空再找时间 → 挖空区间会和时段词重叠
    #      （"下周四晚上七点"里挖掉 0~3 会把"晚上"的"晚"也吃掉），
    #      于是下标错位，标题被挖掉半截（变成"上七点"）。
    #   ⑤ 最终解法：**允许重叠匹配**。用 search(pos) 滑动起点，
    #      每个起点只取"最长的、有证据的"那一个匹配，然后跳过落在"周几"内部的假匹配，
    #      pos 往后挪一位继续找。这样"三点"必然在位置 2 被找到 ——
    #      因为在那个起点上，最长匹配"三点"本身就是合法的。
    def _evidence_match(start_at: int):
        """从 start_at 起，取最长的"像个钟点"的匹配（贪婪策略，天然优先长的）。"""
        best = None
        for candidate in TIME_RE.finditer(text, start_at):
            if candidate.start() != start_at:
                break
            if not TIME_EVIDENCE_RE.search(text[candidate.start():candidate.end()]):
                break
            best = candidate
        return best

    clock = None
    time_span = None
    search_from = 0
    while search_from <= len(text):
        found = _evidence_match(search_from)
        if found is None:
            search_from += 1
            continue
        if any(start < found.end() and found.start() < end for start, end in weekday_spans):
            search_from = max(search_from + 1, found.start() + 1)
            continue
        hour = _cn_number(found.group("hour"))
        if hour is None or hour > 24:
            search_from = found.start() + 1
            continue
        minute = _cn_number(found.group("minute") or "0") or 0
        end = found.end()
        if text[end:end + 1] == "半":                      # "三点半"的"半"
            minute = 30
            end += 1
        explicit = ":" in found.group(0) or "：" in found.group(0)
        clock = _normalize_clock(hour, minute, found.group("period"), explicit=explicit)
        time_span = (found.start(), end)
        break

    # ═══════════ 阶段①：时间区间（15:00-17:00 这种）═══════════
    # 【为什么值得单独处理】写了结束时间就等于给了时长 ——
    # 还默认填 60 分钟是白白丢掉用户已经说出口的信息。
    # 而且"17:00"会作为另一个时间匹配残留进标题（实测标题变成 "17 00 组会"）。
    if time_span:
        gap = re.match(r"\s*[-—~～至到]\s*", text[time_span[1]:])
        if gap:
            after = time_span[1] + gap.end()
            end_match = _evidence_match(after)
            if end_match is not None:
                end_hour = _cn_number(end_match.group("hour"))
                end_minute = _cn_number(end_match.group("minute") or "0") or 0
                if end_hour is not None:
                    end_clock = _normalize_clock(
                        end_hour, end_minute, end_match.group("period"),
                        explicit=(":" in end_match.group(0) or "：" in end_match.group(0)))
                    span_min = (end_clock[0] * 60 + end_clock[1]) - (clock[0] * 60 + clock[1])
                    if 0 < span_min <= 24 * 60:
                        item.duration_min = span_min
                        # 整个区间都算作"已识别"，否则残留会混进标题
                        time_span = (time_span[0], end_match.end())

    # ═══════════ 阶段①：时长 ═══════════
    # 注意顺序：先看有没有"时间区间"（那是明确的时长），再看"大概一小时"这种描述。
    # 反过来的话，"15:00-17:00 大概两小时"会被写成 2 小时（其实区间是 2 小时，正好），
    # 但 "15:00-17:00 大概三小时" 就会自相矛盾 —— 以用户显式写出的区间为准。
    # ── 时长
    # 先看"提前/需要"这类**准备时间**的讲法（"提前一小时准备"）——
    # 这里的"一小时"是准备时间，不是事件时长，两者必须分开。
    # 【踩过的坑】第一版把"提前一小时准备"里的"一小时"当成了事件时长，
    # 于是面试本身变成 60 分钟、准备时间也没记下来，两个信息都错了。
    prep_span = None
    prep_match = PREP_RE.search(text)
    if prep_match:
        prep_minutes = _duration_minutes(prep_match.group("num"), prep_match.group("unit"))
        if prep_minutes:
            item.prep_min = prep_minutes
            # 把"准备/出发"这类尾巴一起算进区间，否则它会被当成标题的一部分。
            # （实测标题变成 "面试 准备" —— "提前一小时"被挖掉了，"准备"孤零零地留下）
            tail = PREP_TAIL_RE.match(text, prep_match.end())
            prep_span = (prep_match.start(), tail.end() if tail else prep_match.end())

    described_span = None
    for match in DURATION_RE.finditer(text):
        if prep_span and prep_span[0] <= match.start() < prep_span[1]:
            continue                       # 这段已经被算作准备时间了
        described = _duration_minutes(match.group("num"), match.group("unit"))
        if described is None:
            continue
        described_span = match.span()
        if item.duration_min == 60:            # 没有区间信息时才采用描述时长
            item.duration_min = described
        break
    duration_span = described_span
    # ═══════════ 阶段①：地点 ═══════════
    # 【这一段是整个解析器里返工最多的地方，把每一版的错法记下来】
    #   ① 找到"在"就在后面圈地点 → 把"走在前列的广东实践"切成 标题="走" + 地点="前列的..."
    #      （标题里恰好有个"在"字）
    #   ② 修成"必须有介词" → "高等数学 东区 A301" 认不出地点（真课表就是没介词的写法）
    #   ③ 修成"已知地点词都认" → 长词优先的排序让"西区活动室"命中"活动室"，
    #      地点变成"活动室"，"西区"留在标题里
    #   ④ 修成"裸教室号优先" → "东区 A301" 里的 301 被裸数字规则抢走，地点=301
    #
    # 最终定的优先级（按可靠性，不是按长度）：
    #   ① 显式标记 "地点：X" / "@X"      —— 写了就是写了，最可靠
    #   ② 已知地点词（含"X区"前缀合并）   —— 按**出现位置最靠前**选，不是按词最长
    #   ③ 行尾裸教室号                   —— 兜底，且要排除被"东区 A301"这类吸收过的情况
    loc_match = LOCATION_RE.search(text)
    item.location = ""
    location_span = None

    explicit = bool(loc_match and re.match(r"^(?:地点[:：]?|@)", loc_match.group(0).strip()))

    # ── ② 已知地点词
    place_hit = None
    for place in KNOWN_PLACES:
        index = text.find(place)
        if index == -1:
            continue
        before = text[:index]
        has_prep = bool(before) and before.rstrip()[-1] in "在@到去"
        if place in AMBIGUOUS_PLACES and not has_prep:
            continue
        # 选**出现位置最靠前**的，而不是最长的 ——
        # "组会 西区活动室" 里"西区"(pos 3) 比"活动室"(pos 5) 靠前，
        # 而"西区"正是我们要的；按长度选会选中"活动室"，把"西区"留在标题里。
        if place_hit is None or index < place_hit[1]:
            place_hit = (place, index)
    if place_hit:
        place, index = place_hit
        start, end = index, index + len(place)
        # "西区" + "活动室" 这种复合写法：往前并一个"X区"，往后并一个已知地点词
        start = _extend_area_prefix(text, start)
        end = _extend_place_tail(text, end)
        tail = text[end:]
        room = ROOM_RE.match(tail.strip())
        name = text[start:end]
        if room:
            name = f"{name} {room.group(0)}"
            end = text.find(room.group(0), end) + len(room.group(0))
        place_hit = (name, start, end)

    # ── ③ 教室号（仅在没识别到地点词时用）。两种形态都要认：
    #      "程序设计基础 3605"    → 行尾裸数字
    #      "高等数学A I 食品316"  → 文字后面紧跟数字（真课表的"食品楼 316"）
    #    ⚠️ 第一版只认"行尾恰好是数字"的情况，于是"食品316"整串留在标题里、
    #       地点还被取成后三位"316"。**数字前面紧挨汉字时，那个汉字属于地点名**，
    #       要把它一起收进地点、并从标题里挖掉。
    room_match = None
    if not place_hit and not explicit:
        room_match = re.search(r"(?<![\dA-Za-z\-/—~])(\d{3,4})\s*$", text)
        if room_match and re.search(r"\d\s*周\s*$", text):
            room_match = None                      # "4-16周"结尾，不是教室
        elif room_match and not 100 <= int(room_match.group(1)) <= 9999:
            room_match = None
        elif room_match:
            head = text[:room_match.start()].rstrip()
            label = re.search(r"([\u4e00-\u9fff]{1,4})$", head)
            if label:
                room_match = _Span(label.start(), room_match.end(),
                                   f"{label.group(1)}{room_match.group(1)}")

    if explicit:
        raw_loc = loc_match.group("loc").strip()
        stop = LOCATION_STOP.search(raw_loc)
        if stop:
            raw_loc = raw_loc[:stop.start()]
        raw_loc = re.sub(r"\s*(?:开|举行|进行|上课|集合|碰头|签到).*$", "", raw_loc)
        raw_loc = DURATION_RE.split(raw_loc)[0]
        item.location = re.sub(r"^(?:在|@|地点[:：]?|到|去)\s*", "", raw_loc).strip(" ，,。")
        probe = text.find(item.location, loc_match.start()) if item.location else -1
        location_span = (probe - 1, probe + len(item.location)) if probe > 0 else None
    elif place_hit:
        item.location, location_span = place_hit[0].strip(), (place_hit[1], place_hit[2])
        # 介词本身也算进"要挖掉"的区间（"在""@"不属于标题）
        if location_span[0] > 0 and text[location_span[0] - 1] in "在@":
            location_span = (location_span[0] - 1, location_span[1])
    elif room_match:
        item.location = room_match.group(1)
        location_span = room_match.span(1)

    # ═══════════ 阶段②：算标题（挖洞） ═══════════
    spans = list(weekday_spans)
    for regex in (DATE_ISO_RE, DATE_CN_RE):
        spans.extend(m.span() for m in regex.finditer(text))
    if relative_match:
        spans.append(relative_match.span())        # "明天/后天/今天"也要挖掉，否则会混进标题
    for span in (time_span, duration_span, location_span, prep_span):
        if span:
            spans.append(span)
    for word in ("每周", "大概", "大约", "差不多", "提前",
                 "记得", "别忘", "别忘了", "打算", "计划", "安排", "需要"):
        start = 0
        while (index := text.find(word, start)) != -1:
            spans.append((index, index + len(word)))
            start = index + len(word)

    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "

    # 第二遍：清理残留。
    # ⚠️ 千万别在这里"删掉所有数字" —— 第一版这么干，把房间号 A301 削成了 A。
    #    **只清理已知的残留，不要泛化地删。**
    title = "".join(chars)
    title = re.sub(r"[（）()]", " ", title)
    title = re.sub(r"[，,、。;；:：@\-—~～]+", " ", title)
    # 多个星期几之间的连接词会孤零零地留下（"每周三和周五" → 挖掉两个"周X"后剩一个"和"）
    title = re.sub(r"^\s*(?:和|与|、|及|跟)+\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    # 先剪尾巴，再剥前缀动词 —— 顺序反了的话，"面试 提前一小时准备" 这种
    # 尾巴没剪掉，前缀剥离的判断也会被带偏
    title = TITLE_TAIL_RE.sub("", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    item.title = _strip_leading_verb(title)

    # ═══════════ 阶段③：组装 ═══════════
    if date_expr and date_expr["type"] == "weekdays":
        weekdays = date_expr["weekdays"]
        # 重复事件的起点取"这周的那一天"；就算这周已经过了也没关系，
        # 因为 weekly 规则是从起点往后每 7 天生效一次，起点本身不会造成误报
        anchor = resolve_date({"type": "weekday", "weekday": weekdays[0], "week": "this"}, today)
        if anchor and clock:
            item.start = datetime.combine(anchor, datetime.min.time()).replace(
                hour=clock[0], minute=clock[1])
        item.note = f"识别为每周重复（{'、'.join(WEEKDAY_NAMES[d] for d in weekdays)}）"
        if not item.until:
            # 默认算到二十周后 —— 说"每周三组会"时，没人希望它开到天荒地老
            item.until = _default_until(today)
    else:
        day = resolve_date(date_expr, today) if date_expr else None
        if day and clock:
            item.start = datetime.combine(day, datetime.min.time()).replace(
                hour=clock[0], minute=clock[1])
        elif day:
            item.note = "识别到了日期，但没识别到具体几点"

    if not item.title:
        item.missing.append("事件名称")
    if item.start is None:
        item.missing.append("开始时间")
    if not item.location:
        item.note = (item.note + "；" if item.note else "") + "没写地点"
    item.category = _guess_category(item.title)

    return item


def _extend_area_prefix(text: str, start: int) -> int:
    """地点词前面可能紧挨着"X区"（"西区" + "活动室" → "西区活动室"）。

    只在**确实构成"X区"且合并后仍在原文里**时才往前并，否则会把
    "我在西区"里的"在"也吃进来。
    """
    if start >= 2 and text[start - 2:start] in ("东区", "西区", "南区", "北区", "中区"):
        return start - 2
    if start >= 3 and text[start - 3] in "在@到去" and \
            text[start - 2:start] in ("东区", "西区", "南区", "北区", "中区"):
        return start - 2          # 前面的"在"留给"挖介词"那一步处理
    return start


def _extend_place_tail(text: str, end: int) -> int:
    """地点词后面可能紧跟另一个地点词（"西区" + "活动室" → "西区活动室"）。

    只在"并起来之后整串确实在原文里"才合并。
    """
    for place in KNOWN_PLACES:
        if text.startswith(place, end):
            return end + len(place)
    return end


def _strip_leading_verb(title: str) -> str:
    """剥掉标题开头那个"动作词"（"开班会" → "班会"）。

    【为什么要写得这么小心】
        天真的做法是"以这些动词开头就删掉"，但那样 "开会安排" 会被削成 "会安排",
        "上班时间" 会被削成 "班时间" —— 删出了新错误，而且看不出来是解析干的。

        所以这里加了双保险：**只有当"剩下的部分"确实像个事件词时才剥**。
        换句话说，剥的代价必须换来一个更像事件名的结果，否则原样保留。
        解析器宁可留下一个啰嗦但正确的标题，也不要一个干净但错误的标题。
    """
    prefixes = ("去", "参加", "开", "上", "举行", "进行", "有")
    # 剥完之后应该剩下的东西（这些词说明"剥对了"）
    expect = ("班会", "会", "组会", "例会", "会议", "讨论", "讲座", "课", "考试",
              "面试", "笔试", "复试", "比赛", "活动", "招新", "培训", "答辩", "体检")
    for prefix in prefixes:
        if not title.startswith(prefix) or len(title) <= len(prefix):
            continue
        rest = title[len(prefix):].strip()
        if any(word in rest for word in expect):
            return rest
    return title


def _default_until(today: date) -> str:
    """重复事件的默认结束：往后 20 周（差不多一个学期）。"""
    return (today + timedelta(weeks=20)).isoformat()


def _guess_category(title: str) -> str:
    """按关键词猜类别。只影响展示分组和配色，猜错无副作用 —— 所以可以放心用规则。"""
    rules = [
        ("interview", ("面试", "笔试", "复试", "宣讲")),
        ("exam", ("考试", "测验", "期中", "期末", "补考")),
        ("meeting", ("班会", "例会", "组会", "会议", "开会", "讨论")),
        ("activity", ("社团", "活动", "讲座", "比赛", "招新", "志愿", "英语角", "晚会")),
        ("course", ("课", "实验", "习题", "辅导", "选修", "大学", "导论", "数学", "英语")),
    ]
    for category, keys in rules:
        if any(key in title for key in keys):
            return category
    return "other"


# ═══════════════════════════ 模型解析（可选）═══════════════════════════

EXTRACT_PROMPT = """你是一个信息抽取器。从下面这句话里抽出日程信息，**只输出一行 JSON**。

规则：
- 日期不要自己算！只填"零件"：{{"type":"weekday","weekday":4,"week":"next"}} 表示"下周四"
  type 可以是 relative（用 offset，今天=0）、weekday（weekday:1~7，week:this/next/after_next）、
  absolute（year/month/day）
- 时间用 24 小时制，填 {{"hour":19,"minute":0}}
- 认不出来的字段留空字符串，**不要猜，不要编**

输出格式（只这一行，不要解释、不要代码块）：
{{"title":"事件名","date":{{...}},"time":{{"hour":0,"minute":0}},"duration_min":60,"location":"","recurring":false}}

要抽取的句子：
{text}

你的输出："""


def parse_with_llm(text: str, today: date | None = None, ask=None) -> list[ParsedEvent]:
    """让模型抽取"零件"，再由代码算日期。

    参数 ask 是一个 callable(prompt) -> str。默认用 agent 的 ask_llm。
    把 ask 做成参数是为了能测试（不用真的起一个 Ollama）。
    """
    today = today or date.today()
    if ask is None:
        from agent import ask_llm as ask      # 延迟导入：这个模块不该强依赖 Ollama

    rules_result = parse_text(text, today)
    raw = ask(EXTRACT_PROMPT.format(text=text.strip()))
    data = _extract_json(raw)
    if not data:
        for item in rules_result:
            item.note = (item.note + "；" if item.note else "") + "模型没给出可用的 JSON，已退回规则解析"
        return rules_result

    item = ParsedEvent(
        title=str(data.get("title") or "").strip(),
        location=str(data.get("location") or "").strip(),
        duration_min=int(data.get("duration_min") or 60),
        category="other",
    )
    date_part = data.get("date") or {}
    time_part = data.get("time") or {}
    recurring = bool(data.get("recurring"))

    if recurring and date_part.get("type") == "weekday":
        weekday = int(date_part.get("weekday") or 0)
        item.weekdays = [weekday] if 1 <= weekday <= 7 else []
        date_part = {"type": "weekday", "weekday": weekday, "week": "this"}

    day = resolve_date(date_part, today)
    hour = int(time_part.get("hour") or 0)
    minute = int(time_part.get("minute") or 0)
    if day:
        item.start = datetime.combine(day, datetime.min.time()).replace(
            hour=hour % 24, minute=minute % 60)
    if item.weekdays:
        item.until = _default_until(today)
        item.note = "模型识别为每周重复"

    # 模型和规则谁更全就用谁 —— 两者互补，合起来比单独用任何一个都强
    fallback = rules_result[0] if rules_result else ParsedEvent()
    if not item.title:
        item.title = fallback.title
    if item.start is None:
        item.start = fallback.start
    if not item.location:
        item.location = fallback.location
    if item.duration_min == 60 and fallback.duration_min != 60:
        item.duration_min = fallback.duration_min

    if not item.title:
        item.missing.append("事件名称")
    if item.start is None:
        item.missing.append("开始时间")
    item.category = _guess_category(item.title)
    item.note = (item.note + "；" if item.note else "") + "由大模型抽取"
    return [item]


def _extract_json(raw: str) -> dict | None:
    """从模型输出里抠出 JSON。小模型爱加废话和代码块，所以逐层容错。"""
    cleaned = (raw or "").replace("```json", "").replace("```", "")
    depth, start = 0, None
    for index, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


# ═══════════════════════════ 交给内核之前：变成 Event ═══════════════════════════

def to_event(item: ParsedEvent) -> Event:
    """把解析结果变成内核认识的 Event。"""
    if item.start is None:
        raise ValidationError("解析结果里没有开始时间，不能落库")
    event = Event(
        title=item.title or "未命名日程",
        start=item.start,
        duration_min=item.duration_min,
        location=item.location,
        category=item.category,
        recurrence="weekly" if item.weekdays else "one_off",
        weekdays=list(item.weekdays),
        until=date.fromisoformat(item.until) if item.weekdays and item.until else None,
        prep_min=item.prep_min,
        source="text",
    )
    event.validate()
    event.ensure_id()
    return event
