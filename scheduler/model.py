"""事件数据模型与校验

【一个设计决定：时间用"本地时间的裸 datetime"，不带时区】
    理由：人的课表和日程是"墙上时钟"的概念 —— "周四 19:00 开会"就是 19:00，
    不会因为夏令时或跨时区而变化。带上时区反而会让 .ics 导入后差一小时。
    （代价：跨时区场景不支持。这是有意的取舍，属"未来工作"。）

【另一个决定：weekdays 用 1=周一 … 7=周日】
    跟 Python 的 datetime.isoweekday() 对齐，少一次换算就少一个 bug。
    导出 .ics 时才翻译成 MO/TU/WE/... 。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

# 事件类型：只影响展示分组和默认配色，不参与冲突判定
CATEGORIES = ("course", "meeting", "activity", "interview", "exam", "personal", "other")

# 事件来源：Day 5 的 A/B 实验要按来源统计识别正确率，所以从第一天就记下来
#   manual  手工录入（基准）
#   text    自然语言文字（agent 抽取）
#   ocr     截图 → OCR → 文字
#   vlm     截图 → 多模态模型直接看图
#   voice   语音 → ASR → 文字
SOURCES = ("manual", "text", "ocr", "vlm", "voice")

WEEKDAY_NAMES = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}

# 两个事件在同一地点连续进行时，默认不留赶路时间
DEFAULT_SAME_PLACE_BUFFER = 0
# 两个事件在不同地点之间，默认预留的赶路时间（分钟）
DEFAULT_TRAVEL = 10
# 一天排超过这么多件事，即使不重叠也提示"精力不够"
DEFAULT_MAX_EVENTS_PER_DAY = 5


class ValidationError(ValueError):
    """事件不合法。错误信息必须人能读懂 —— 它会被原样喂回给模型去重试。"""


def _parse_dt(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None, microsecond=0)
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(second=0, microsecond=0)
        except ValueError:
            continue
    raise ValidationError(
        f"{field_name} 的时间格式不对：{value!r}。请用 2026-03-05T19:00 这种格式（年-月-日T时:分）。"
    )


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValidationError(f"{field_name} 的日期格式不对：{value!r}。请用 2026-03-05 这种格式。")


def make_id(title: str, start: datetime, salt: str = "") -> str:
    """由内容生成稳定的 id。

    为什么不用自增序号：因为要给 .ics 里的 UID 用。重复导出同一件事时，
    UID 必须不变，否则手机日历会把它当成一个新事件，越导越多。
    """
    raw = f"{title}|{start.isoformat()}|{salt}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


@dataclass
class Event:
    """一个日程事项。

    支持两种形态，用 recurrence 区分：
      - one_off      一次性（面试、班会、报名截止）
      - weekly       每周重复（社团例会、体育课），用 weekdays + 起止日期圈定范围
    """

    title: str
    start: datetime
    duration_min: int = 60
    location: str = ""
    category: str = "other"
    recurrence: str = "one_off"                 # one_off | weekly
    weekdays: list[int] = field(default_factory=list)   # 仅 weekly 用，1=周一…7=周日
    until: date | None = None                    # 仅 weekly 用：最后一次有效日期
    # 前置准备时间：面试前要留 1 小时准备。它会让"事件之前"的一段时间变忙。
    prep_min: int = 0
    # 后果等级：high 的事情冲突了必须解决，low 的可以让步（用于替代方案打分）
    consequence: str = "normal"                  # low | normal | high
    source: str = "manual"
    note: str = ""
    id: str = ""

    # ────────────────────────── 校验 ──────────────────────────

    def validate(self) -> None:
        """返回可读的错误。这些错误会被原样喂回给模型，所以必须说人话。"""
        if not str(self.title).strip():
            raise ValidationError("缺少 title（这件事叫什么）。")
        if not isinstance(self.start, datetime):
            raise ValidationError("start 必须是 datetime。")
        if not isinstance(self.duration_min, int) or self.duration_min <= 0:
            raise ValidationError(f"duration_min 必须是正整数（分钟），当前是 {self.duration_min!r}。")
        if self.duration_min > 24 * 60:
            raise ValidationError("duration_min 不能超过 1440 分钟（一天）。跨天事项请拆成两个事件。")
        if self.category not in CATEGORIES:
            raise ValidationError(f"category 只能是这些之一：{'、'.join(CATEGORIES)}，当前是 {self.category!r}。")
        if self.source not in SOURCES:
            raise ValidationError(f"source 只能是这些之一：{'、'.join(SOURCES)}，当前是 {self.source!r}。")
        if self.consequence not in ("low", "normal", "high"):
            raise ValidationError(f"consequence 只能是 low/normal/high，当前是 {self.consequence!r}。")
        if self.prep_min < 0:
            raise ValidationError("prep_min 不能是负数。")

        if self.recurrence == "weekly":
            if not self.weekdays:
                raise ValidationError("weekly 事件必须给出 weekdays，例如 [3] 表示每周三。")
            bad = [d for d in self.weekdays if d not in WEEKDAY_NAMES]
            if bad:
                raise ValidationError(
                    f"weekdays 只能是 1~7（1=周一，7=周日），这些不对：{bad}。"
                )
            if self.until is None:
                raise ValidationError("weekly 事件必须给出 until（最后一次有效的日期），否则它会无限重复。")
            if self.until < self.start.date():
                raise ValidationError(
                    f"until（{self.until}）不能早于起始日期（{self.start.date()}）。"
                )
            if self.start.isoweekday() not in self.weekdays:
                # 不是硬错误，但几乎总是笔误，提示出来
                raise ValidationError(
                    f"起始日期 {self.start.date()} 是{WEEKDAY_NAMES[self.start.isoweekday()]}，"
                    f"不在 weekdays={self.weekdays} 里。请确认 weekdays 或起始日期。"
                )
        elif self.recurrence != "one_off":
            raise ValidationError(f"recurrence 只能是 one_off 或 weekly，当前是 {self.recurrence!r}。")

    def ensure_id(self) -> str:
        if not self.id:
            self.id = make_id(self.title, self.start, self.location)
        return self.id

    # ────────────────────────── 便利方法 ──────────────────────────

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_min)

    @property
    def busy_start(self) -> datetime:
        """含前置准备时间的"开始忙"时刻。"""
        return self.start - timedelta(minutes=self.prep_min)

    def end_date(self) -> date:
        """一次性事件等价于"只在这一天重复"。"""
        return self.until if self.recurrence == "weekly" and self.until else self.start.date()

    def describe(self) -> str:
        when = f"{self.start:%Y-%m-%d %H:%M}"
        if self.recurrence == "weekly":
            days = "、".join(WEEKDAY_NAMES[d] for d in sorted(self.weekdays))
            when = f"每{days} {self.start:%H:%M}（至 {self.until}）"
        where = f" @ {self.location}" if self.location else ""
        return f"[{self.id}] {self.title} | {when} | {self.duration_min} 分钟{where}"

    # ────────────────────────── 序列化 ──────────────────────────

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"] = self.start.strftime("%Y-%m-%dT%H:%M")
        data["until"] = self.until.isoformat() if self.until else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        payload = dict(data)
        payload.pop("op", None)          # JSONL 里的操作类型标记，不属于事件本身
        payload["start"] = _parse_dt(payload.get("start"), "start")
        if payload.get("until"):
            payload["until"] = _parse_date(payload["until"], "until")
        else:
            payload["until"] = None
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValidationError(f"不认识的字段：{'、'.join(sorted(unknown))}。可用字段：{'、'.join(sorted(known))}。")
        event = cls(**payload)
        event.validate()
        return event

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
