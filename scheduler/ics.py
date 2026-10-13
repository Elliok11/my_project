"""导出 .ics（iCalendar）—— 让这个工具真的被人用起来

【为什么这是整个项目最关键的一步】
    一个只能自己看的网页，用两天就会关掉。
    但 .ics 导进手机系统日历之后：锁屏能看到、有人找你时你能当场查、到点会响。
    从"我做了个demo"变成"我在用这个工具" —— 分界线就在这个文件里。

【格式要点（RFC 5545）】
    - 行尾必须是 CRLF，不是 \\n
    - 单行超过 75 字节要折行（续行以空格开头）
    - 文本里的 , ; \\ 换行 要转义
    - 重复事件用 RRULE；因为本项目的时间是"墙上时钟"（不带时区），
      所以 DTSTART 要写成浮动的本地时间（不带 Z 后缀），否则导入后会差时区。

【有意不做的】
    不做 VTIMEZONE、不做提醒（VALARM）、不做跨时区。
    理由：目标是"能在手机上正确显示"，加这些只会引入新 bug。
"""
from __future__ import annotations

from datetime import datetime

from .model import WEEKDAY_NAMES, Event

# ICS 用的星期缩写：注意 SU=周日，跟 isoweekday 的 7 对应
ICS_WEEKDAYS = {1: "MO", 2: "TU", 3: "WE", 4: "TH", 5: "FR", 6: "SA", 7: "SU"}

CATEGORY_LABELS = {
    "course": "课程",
    "meeting": "会议",
    "activity": "活动",
    "interview": "面试",
    "exam": "考试",
    "personal": "个人",
    "other": "其他",
}

# 事件的正文里带上 id，便于在日历里反查是哪一条
MAX_LINE_BYTES = 75


def _escape(text: str) -> str:
    """转义 ICS 文本：反斜杠、分号、逗号、换行。顺序很重要（先转义反斜杠）。"""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def _fits(raw: bytes, cut: int) -> bool:
    """前 cut 个字节是否构成合法的 UTF-8（也就是切在字符边界上）。

    用它做判据，而不是自己推算 UTF-8 的长度规律 ——
    因为"切点合不合法"这件事，让 Python 的解码器来回答最可靠。
    """
    if cut >= len(raw):
        return True
    try:
        raw[:cut].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _character_width(lead: int) -> int:
    """一个 UTF-8 字符占几个字节（用于"往前站到下一个边界"）。"""
    if lead < 0x80:
        return 1
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC2:
        return 2
    return 1          # 按 1 处理，配合 _fits 的检查也不会切坏


def _fold(line: str) -> str:
    """按 75 字节折行。按字节而不是按字符折 —— 中文一个字 3 字节，按字符折会超限。

    【这个函数错了四版，是整份代码里最值得记的一段】
        RFC 5545：一行不超过 75 **字节**，续行以空格开头，切点必须落在字符边界上。

        前四版的毛病分别是：
        ① 只回退结尾的续字节 → 切点落在"首字节"上时留下半截字符（有头没身）
        ② 想补一条"首字节也回退"，写成 (trail & 0xC0) == 0xC0
           → 但 0xAC & 0xC0 == 0xC0 也成立，续字节被误判，回退一路啃到底，
             把前面完整汉字的头也啃掉了
        ③ 只管"尾巴别切坏"，没管"下一段会从续字节开头"
        ④ 计数对不上一位：辅助函数返回的是"包含第 pos-1 字节的字符的起点"，
           而我拿它去判断"前 pos 个字节合不合法" —— **差一位，所以永远不生效**，
           代码看起来在做边界处理，实际上一次都没生效过

        ⭐ 最后是怎么收敛的：**别再自己推算 UTF-8 的长度规律，改用不变量。**
           "能不能切在这里" ≡ "前 n 个字节能不能解码成合法 UTF-8"。
           这个判断交给解码器去做，逻辑立刻变得不可能写错。

        更坏的一点：这些 bug **切出来的字节总数一个不差**，
        所以任何长度检查都发现不了，只有"折完拼回去、逐字节比对"的测试能抓住。
        而且只在"中文 + 长度刚好卡在边界"时出现 —— 只测一两个长度会整片躲过去。
    """
    raw = line.encode("utf-8")
    if len(raw) <= MAX_LINE_BYTES:
        return line

    chunks: list[bytes] = []
    pos = 0
    first = True
    while pos < len(raw):
        budget = MAX_LINE_BYTES if first else MAX_LINE_BYTES - 1   # 续行开头那个空格占 1 字节
        cut = min(pos + budget, len(raw))

        # 回退到最近的合法切点（中文是 3 字节，最多退 2 步）
        while cut > pos and not _fits(raw, cut):
            cut -= 1

        # 边界可能"浪费"了一整个字符的位置。若往前站一个字符仍在预算内，就往前站
        if cut < len(raw) and _fits(raw, cut + _character_width(raw[cut])):
            candidate = cut + _character_width(raw[cut])
            if candidate <= pos + budget:
                cut = candidate

        if cut <= pos:                     # 兜底：预算再小也要前进，避免死循环
            cut = min(pos + budget, len(raw))
        chunks.append(raw[pos:cut])
        pos = cut
        first = False

    head, *tail = chunks
    return "\r\n ".join([head.decode("utf-8")] + [c.decode("utf-8") for c in tail])


def _stamp(moment: datetime) -> str:
    """浮动本地时间：YYYYMMDDTHHMMSS，不带 Z。"""
    return moment.strftime("%Y%m%dT%H%M%S")


def event_to_vevent(event: Event, now: datetime | None = None) -> list[str]:
    """把一个 Event 变成 VEVENT 的若干行（未折行）。"""
    now = now or datetime.now()
    lines = ["BEGIN:VEVENT", f"UID:{event.id}@my-scheduler"]
    lines.append(f"DTSTAMP:{_stamp(now)}")

    description = [f"来源：{event.source}", f"类别：{CATEGORY_LABELS.get(event.category, event.category)}"]
    if event.prep_min:
        description.append(f"需要提前 {event.prep_min} 分钟准备")
    if event.note:
        description.append(event.note)
    description.append(f"内部 id：{event.id}")

    lines.append(f"SUMMARY:{_escape(event.title)}")
    lines.append(f"DESCRIPTION:{_escape(chr(10).join(description))}")
    if event.location:
        lines.append(f"LOCATION:{_escape(event.location)}")
    lines.append(f"CATEGORIES:{_escape(CATEGORY_LABELS.get(event.category, event.category))}")

    if event.recurrence == "weekly":
        lines.append(f"DTSTART:{_stamp(event.start)}")
        lines.append(f"DTEND:{_stamp(event.end)}")
        days = ",".join(ICS_WEEKDAYS[d] for d in sorted(set(event.weekdays)))
        # UNTIL 用当天 23:59:59 保证这一天也被包含进去
        until = event.end_date().strftime("%Y%m%d") + "T235959"
        lines.append(f"RRULE:FREQ=WEEKLY;BYDAY={days};UNTIL={until}")
    else:
        lines.append(f"DTSTART:{_stamp(event.start)}")
        lines.append(f"DTEND:{_stamp(event.end)}")

    # 后果等级高的给个高优先级，日历里会显示成重要
    if event.consequence == "high":
        lines.append("PRIORITY:1")

    lines.append("END:VEVENT")
    return lines


def build_calendar(events: list[Event], calendar_name: str = "我的日程") -> str:
    """生成整个 .ics 的内容。"""
    now = datetime.now()
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//my_project//scheduler//CN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_escape(calendar_name)}",
    ]
    for event in events:
        lines.extend(event_to_vevent(event, now))
    lines.append("END:VCALENDAR")

    # 折行 + CRLF 收尾
    folded = [_fold(line) for line in lines]
    return "\r\n".join(folded) + "\r\n"


def write_calendar(events: list[Event], path, calendar_name: str = "我的日程") -> int:
    """写文件，返回写入的事件数。"""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        # 空的 .ics 会让部分日历 App 报错，明确拒绝而不是写个坏文件
        raise ValueError("没有任何事件可以导出，先添加日程再导出。")
    # newline="" 很重要：否则 Python 会把 \r\n 再翻译一遍，变成 \r\r\n
    target.write_text(build_calendar(events, calendar_name), encoding="utf-8", newline="")
    return len(events)


def weekday_label(day: int) -> str:
    return WEEKDAY_NAMES.get(day, str(day))
