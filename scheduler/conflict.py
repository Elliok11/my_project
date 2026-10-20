"""冲突检测 / 找空档 / 提议替代方案

【这一层的定位：它是"裁判"，不是"助手"】
    agent 说"把班会放到周四 19:00"，这句话只是一个**提议**。
    能不能放，由这里决定 —— 而且必须是**确定性**的、可复现的、可被单元测试的。

    所以本模块：不调用任何模型、不含随机数、时间窗口以外的东西一律不看。

【冲突有三种，严重程度不同 —— 这是本项目的核心设计】
    1. overlap  时间真的重叠了                      → 硬冲突，必须解决
    2. travel   上一件事结束到下一件开始不够赶路    → 硬冲突（如果地点不同）
    3. overload 一天排的事太多，精力不够            → 只是警告，不阻止
       prep     前置准备时间和别的事撞了            → 硬冲突

    为什么 overload 不阻止：因为"今天有四件事"是不是问题，只有人知道。
    工具的职责是把事实摆出来，不是替人做决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from .model import (
    DEFAULT_MAX_EVENTS_PER_DAY,
    DEFAULT_SAME_PLACE_BUFFER,
    DEFAULT_TRAVEL,
    WEEKDAY_NAMES,
    Event,
)


# ═══════════════════════════ 配置 ═══════════════════════════

@dataclass
class Rules:
    """排程规则。全部可配置，因为"多紧算紧"是个人偏好，不该写死在代码里。"""

    travel_min: int = DEFAULT_TRAVEL                 # 不同地点之间默认赶路时间
    same_place_buffer: int = DEFAULT_SAME_PLACE_BUFFER
    max_events_per_day: int = DEFAULT_MAX_EVENTS_PER_DAY
    day_start: time = time(8, 0)                     # 找空档时的"一天开始"
    day_end: time = time(22, 30)                     # 找空档时的"一天结束"
    # 地点之间的实测赶路时间，覆盖 travel_min。例如 {"东区": {"西区": 20}}
    travel_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def travel_between(self, from_place: str, to_place: str) -> int:
        a, b = (from_place or "").strip(), (to_place or "").strip()
        if a == b:
            return self.same_place_buffer
        if a in self.travel_matrix and b in self.travel_matrix[a]:
            return int(self.travel_matrix[a][b])
        if b in self.travel_matrix and a in self.travel_matrix[b]:
            return int(self.travel_matrix[b][a])
        return self.travel_min


# ═══════════════════════════ 具体发生的一次 ═══════════════════════════

@dataclass
class Occurrence:
    """一个事件在某个具体日期的具体一次。

    为什么要这个中间层：`Event` 描述的是"每周三 19:00 的例会"（规则），
    而冲突检测要处理的是"3 月 5 日 19:00 这一次"（事实）。
    规则和事实混在一起是排程代码最常见的 bug 来源。
    """

    event: Event
    day: date
    start: datetime
    end: datetime
    busy_start: datetime          # 含前置准备时间

    @property
    def title(self) -> str:
        return self.event.title

    @property
    def location(self) -> str:
        return self.event.location

    def describe(self) -> str:
        return f"{self.start:%m-%d %H:%M}-{self.end:%H:%M} {self.title}"


def occurrences_of(event: Event, start_day: date, end_day: date) -> list[Occurrence]:
    """把一条事件规则展开成 [start_day, end_day] 区间内的所有具体发生。"""
    result: list[Occurrence] = []
    clock = event.start.time()

    if event.recurrence == "one_off":
        days = [event.start.date()]
    else:
        first = event.start.date()
        last = min(event.end_date(), end_day)
        days = []
        # 【踩过的坑】第一版只取了"离起始日最近的那个 weekday"，
        # 于是 weekdays=[1,4] 的体育课只展开了周一、周四整个消失。
        # 正确做法是：对每一个 weekday 各自算出一个星期几序列，再合并排序。
        for weekday in sorted(set(event.weekdays)):
            delta = (weekday - first.isoweekday()) % 7
            cursor = first + timedelta(days=delta)
            while cursor <= last:
                days.append(cursor)
                cursor += timedelta(days=7)
        days = sorted(set(days))

    for day in days:
        if day < start_day or day > end_day:
            continue
        start = datetime.combine(day, clock)
        end = start + timedelta(minutes=event.duration_min)
        result.append(
            Occurrence(
                event=event,
                day=day,
                start=start,
                end=end,
                busy_start=start - timedelta(minutes=event.prep_min),
            )
        )
    return result


def expand(events: list[Event], start_day: date, end_day: date) -> list[Occurrence]:
    """展开一组事件，按时间排序。"""
    everything: list[Occurrence] = []
    for event in events:
        everything.extend(occurrences_of(event, start_day, end_day))
    return sorted(everything, key=lambda o: (o.start, o.title))


# ═══════════════════════════ 冲突 ═══════════════════════════

@dataclass
class Issue:
    """一条冲突/警告。`message` 会被原样喂回给模型，所以必须是人话。"""

    kind: str                     # overlap | travel | overload | prep
    severity: str                 # hard | warn
    message: str
    events: list[str] = field(default_factory=list)      # 涉及的 event id
    day: date | None = None

    def __str__(self) -> str:
        mark = "❌" if self.severity == "hard" else "⚠️ "
        return f"{mark} {self.message}"


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """半开区间相交判定：[start, end) 与 [start, end)。"""
    return a_start < b_end and b_start < a_end


def _minutes(a: datetime, b: datetime) -> int:
    return int((b - a).total_seconds() // 60)


def detect(occurrences: list[Occurrence], rules: Rules | None = None) -> list[Issue]:
    """检测一批具体发生之间的冲突。这是本模块的主函数。

    【为什么分两阶段，而不是一个循环搞定】
        第一版是一个循环全查，结果发现两个问题：
        ① 两件事重叠时，"赶路"检查还会再报一条"只有 -60 分钟" —— 负的时间摆在那里，
           说明它压根没意识到这两件事已经撞了。同一个事实报两遍就是噪声。
        ② 面试的"准备时间不够"被"时间重叠"抢先报了，因为两者被归成同一类。
           可"提前准备时间被占用"和"两件事撞车"是**不同的病因**，处理方式也不同：
           前者要往前挪准备时间，后者要改期。
        所以现在：先找出真正重叠的对，赶路检查跳过这些对，准备时间单独成一条。
    """
    rules = rules or Rules()
    issues: list[Issue] = []
    overlapped: set[tuple[str, str]] = set()

    # ── 1. 真正的时间重叠 + 前置准备时间相撞（两者分开关，因为病因不同）
    for i in range(len(occurrences)):
        for j in range(i + 1, len(occurrences)):
            first, second = occurrences[i], occurrences[j]
            if not _overlaps(first.busy_start, first.end, second.busy_start, second.end):
                continue
            if _overlaps(first.start, first.end, second.start, second.end):
                overlapped.add((first.event.id, second.event.id))
                issues.append(Issue(
                    kind="overlap", severity="hard", day=first.day,
                    events=[first.event.id, second.event.id],
                    message=(f"时间重叠：{first.describe()} 与 {second.describe()} 撞在一起，"
                             f"重叠 {_minutes(max(first.start, second.start), min(first.end, second.end))} 分钟。"),
                ))
            elif first.event.prep_min or second.event.prep_min:
                later, earlier = (second, first) if second.start >= first.start else (first, second)
                issues.append(Issue(
                    kind="prep", severity="hard", day=first.day,
                    events=[first.event.id, second.event.id],
                    message=(f"准备时间不够：{later.title} 要提前 {later.event.prep_min} 分钟准备"
                             f"（{later.busy_start:%H:%M} 就得开始），"
                             f"可那段时间被「{earlier.title} {earlier.start:%H:%M}-{earlier.end:%H:%M}」占着，"
                             f"两者重叠 {_minutes(max(later.busy_start, earlier.start), min(later.start, earlier.end))} 分钟。"),
                ))

    # ── 2. 赶路时间：只看相邻两件事，且跳过已经报过重叠的那些对
    ordered = sorted(occurrences, key=lambda o: o.start)
    for first, second in zip(ordered, ordered[1:]):
        if first.day != second.day:
            continue
        if (first.event.id, second.event.id) in overlapped or (second.event.id, first.event.id) in overlapped:
            continue      # 已经报过重叠了，别再补一条"赶不上"
        needed = rules.travel_between(first.location, second.location)
        gap = _minutes(first.end, second.start)
        if gap < needed:
            where = f"{first.location or '未填地点'} → {second.location or '未填地点'}"
            issues.append(Issue(
                kind="travel", severity="hard", day=first.day,
                events=[first.event.id, second.event.id],
                message=(f"赶不上：{first.title} {first.end:%H:%M} 结束，{second.title} "
                         f"{second.start:%H:%M} 开始，只有 {gap} 分钟，但 {where} 需要 {needed} 分钟。"),
            ))

    # ── 3. 一天塞太多（只警告，不阻止）
    by_day: dict[date, list[Occurrence]] = {}
    for occ in occurrences:
        by_day.setdefault(occ.day, []).append(occ)
    for day, items in sorted(by_day.items()):
        if len(items) > rules.max_events_per_day:
            busy = sum(o.event.duration_min for o in items)
            issues.append(Issue(
                kind="overload", severity="warn", day=day,
                events=[o.event.id for o in items],
                message=(f"{day:%m-%d}（{WEEKDAY_NAMES[day.isoweekday()]}）排了 {len(items)} 件事，"
                         f"共 {busy // 60} 小时 {busy % 60} 分钟，超过设定的 {rules.max_events_per_day} 件上限。"),
            ))

    return issues


def issues_touching(issues: list[Issue], event_id: str) -> list[Issue]:
    """只留下和某个事件有关的冲突。

    【为什么必须有这个过滤】
        实测发现：裁决"把班会改到周三"时，输出里混进了库里**原本就存在**的
        "周四体育和班会撞了" —— 用户会以为是这次改动引起的。
        工具把"你这次提议的问题"和"数据里已有的问题"混在一起，是最招人烦的毛病之一。
        所以：裁决一个提议，就只报和这个提议有关的冲突。
        （库里的历史问题由 `verify` 命令单独负责，各管一摊。）
    """
    return [i for i in issues if event_id in i.events]


def check_range(events: list[Event], start_day: date, end_day: date,
                rules: Rules | None = None) -> list[Issue]:
    """便捷入口：给事件和日期区间，直接返回冲突列表。"""
    return detect(expand(events, start_day, end_day), rules)


# ═══════════════════════════ 找空档 ═══════════════════════════

def busy_intervals(occurrences: list[Occurrence], rules: Rules) -> list[tuple[datetime, datetime]]:
    """把一天里所有"忙"的时间段合并成互不相交的区间。

    包含两件事：
      - 前置准备时间也算忙
      - 两件事之间的赶路时间也算忙（否则"空档"会给出一个物理上赶不到的时间）
    """
    raw: list[tuple[datetime, datetime]] = []
    ordered = sorted(occurrences, key=lambda o: o.busy_start)
    for index, occ in enumerate(ordered):
        raw.append((occ.busy_start, occ.end))
        if index + 1 < len(ordered):
            nxt = ordered[index + 1]
            needed = rules.travel_between(occ.location, nxt.location)
            if needed > 0:
                raw.append((occ.end, min(occ.end + timedelta(minutes=needed), nxt.start)))

    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(raw):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_gaps(occurrences: list[Occurrence], day: date,
              rules: Rules | None = None, min_minutes: int = 30) -> list[tuple[datetime, datetime]]:
    """找出这一天里长度 >= min_minutes 的空档。"""
    rules = rules or Rules()
    window_start = datetime.combine(day, rules.day_start)
    window_end = datetime.combine(day, rules.day_end)
    day_items = [o for o in occurrences if o.day == day]

    gaps: list[tuple[datetime, datetime]] = []
    cursor = window_start
    for busy_start, busy_end in busy_intervals(day_items, rules):
        if busy_start > cursor and _minutes(cursor, min(busy_start, window_end)) >= min_minutes:
            gaps.append((cursor, min(busy_start, window_end)))
        cursor = max(cursor, busy_end)
    if _minutes(cursor, window_end) >= min_minutes:
        gaps.append((cursor, window_end))
    return gaps


# ═══════════════════════════ 检查一个"提议" ═══════════════════════════

@dataclass
class Verdict:
    """对一个提议的裁决结果。这是交给模型/界面看的完整答复。"""

    ok: bool
    issues: list[Issue]
    alternatives: list["Candidate"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": [
                {"kind": i.kind, "severity": i.severity, "message": i.message,
                 "events": i.events, "day": i.day.isoformat() if i.day else None}
                for i in self.issues
            ],
            "alternatives": [c.to_dict() for c in self.alternatives],
        }

    def text(self) -> str:
        lines = [f"{'✅ 可以这样安排' if self.ok else '❌ 这样安排不行'}"]
        lines += [f"   {issue}" for issue in self.issues]
        if self.alternatives:
            lines.append("   建议改到：")
            lines += [f"   → {c.describe()}" for c in self.alternatives]
        return "\n".join(lines)


@dataclass
class Candidate:
    """一个候选时间。

    tier 是"改动有多大"的档位，越小越好：
        0 = 同一天、时刻也差不多   （最省事：只动时间，不惊动别人）
        1 = 换了天、但时刻不变     （只挪日期，时间照旧）
        2 = 换了天、时刻也变了     （改动最大）
    """

    start: datetime
    end: datetime
    preferred: bool = False
    reason: str = ""
    tier: int = 2

    def describe(self) -> str:
        tail = f"（{self.reason}）" if self.reason else ""
        return f"{self.start:%m-%d} {WEEKDAY_NAMES[self.start.isoweekday()]} {self.start:%H:%M}-{self.end:%H:%M}{tail}"

    def to_dict(self) -> dict:
        return {
            "start": self.start.strftime("%Y-%m-%dT%H:%M"),
            "end": self.end.strftime("%Y-%m-%dT%H:%M"),
            "preferred": self.preferred,
            "reason": self.reason,
            "tier": self.tier,
        }


# ═══════════════════════════ 提议替代方案 ═══════════════════════════

# 候选时间的对齐粒度：按 15 分钟一跳，符合人的直觉（没人会说"19:07 开会"）
SLOT_STEP_MIN = 15


def alternatives(events: list[Event], candidate: Event,
                 rules: Rules | None = None, lookahead_days: int = 7,
                 limit: int = 3) -> list[Candidate]:
    """给一个注定冲突的提议找替代时间。

    【打分顺序，从好到坏 —— 这里的设计改过一版，值得记】
        档位 0：同一天、时刻也差不多       —— 只动时间，不惊动任何人
        档位 1：换一天、但时刻不变         —— 只挪日期，别人不用重新记时间
        档位 2：换一天、时刻也变           —— 改动最大，排最后

        第一版把"同一天优先"和"同时间优先"混成了一个 preferred 布尔值，结果：
        原定"周三 19:30"，程序推荐"周五 08:00"排在"周五 19:30"前面 ——
        对一个已经约好的会来说，"周五同一时刻"明显比"周五早上八点"更容易接受。
        原因不是排序代码写错了，而是**优先级本身没定义清楚**。
        把"改动有多大"显式写成三档之后，排序就变成一件不需要动脑的事。

        另外，同一档内部按"影响面"排：先按日期早晚（越早越不拖延），
        再按时刻与原定时刻的距离（越接近越省事）。
    """
    rules = rules or Rules()
    others = [e for e in events if e.id != candidate.id]
    base_day = candidate.start.date()
    end_day = base_day + timedelta(days=lookahead_days)
    busy = expand(others, base_day - timedelta(days=1), end_day + timedelta(days=1))

    found: list[Candidate] = []
    seen: set[str] = set()

    def try_slot(day: date, at: datetime, reason: str, tier: int) -> bool:
        start = at.replace(second=0, microsecond=0)
        end = start + timedelta(minutes=candidate.duration_min)
        key = start.isoformat()
        if key in seen:
            return False
        seen.add(key)

        # 别推荐一个"那天已经排爆了"的时间。
        # 这是实测出来的：不加这条时，程序会给出 22:15 这种深夜时段，
        # 而那天本来就排了 7 件事 —— 数学上没冲突，但人不会接受。
        day_count = len({o.event.id for o in busy if o.day == start.date()})
        if day_count >= rules.max_events_per_day:
            return False

        probe = Occurrence(event=candidate, day=start.date(), start=start, end=end,
                           busy_start=start - timedelta(minutes=candidate.prep_min))
        day_items = [o for o in busy if o.day == start.date()] + [probe]
        if any(i.severity == "hard" for i in detect(day_items, rules)):
            return False

        found.append(Candidate(start=start, end=end, preferred=tier == 0,
                               reason=reason, tier=tier))
        return True

    # ── 档位 0：同一天，从近到远地试
    try_slot(base_day, candidate.start, "原定时刻", 0)
    for offset in range(SLOT_STEP_MIN, 12 * 60 + 1, SLOT_STEP_MIN):
        for direction, label in ((1, "同一天推迟"), (-1, "同一天提前")):
            moment = candidate.start + timedelta(minutes=direction * offset)
            if moment.date() != base_day:
                continue
            if rules.day_start <= moment.time() <= rules.day_end:
                try_slot(base_day, moment, label, 0)

    # ── 档位 1：换一天、但时刻不变
    for delta in range(1, lookahead_days + 1):
        day = base_day + timedelta(days=delta)
        try_slot(day, datetime.combine(day, candidate.start.time()), "只挪日期", 1)

    # ── 档位 2：换一天、时刻也变
    for delta in range(1, lookahead_days + 1):
        day = base_day + timedelta(days=delta)
        for moment in _day_slots(day, candidate, rules):
            try_slot(day, moment, "换一天", 2)

    # 先按"改动大小"，再按日期早晚，最后按"离原定时刻多近"
    origin = candidate.start
    found.sort(key=lambda c: (c.tier, c.start.date(), abs((c.start - origin).total_seconds())))
    return found[:limit]


def _day_slots(day: date, candidate: Event, rules: Rules) -> list[datetime]:
    """某一天里所有候选时刻，从上午到晚上。"""
    slots: list[datetime] = []
    moment = datetime.combine(day, rules.day_start)
    stop = datetime.combine(day, rules.day_end)
    while moment <= stop:
        slots.append(moment)
        moment += timedelta(minutes=SLOT_STEP_MIN)
    return slots


def check_candidate(events: list[Event], candidate: Event,
                    rules: Rules | None = None) -> Verdict:
    """裁决一个提议的时间。这是 agent 最常用的那个工具。"""
    rules = rules or Rules()
    candidate.validate()
    others = [e for e in events if e.id != candidate.id]

    # 注意窗口的取法：要往前多铺一天、往后多铺一天。
    # 因为"前置准备时间"可能落在起始日的前一天（周一早八点的面试，准备时间在周日），
    # 如果只展开 [start_day, end_day] 就会漏掉这类冲突。
    window_start = candidate.start.date() - timedelta(days=1)
    window_end = candidate.end_date() + timedelta(days=1)

    probe = occurrences_of(candidate, window_start, window_end)
    everything = expand(others, window_start, window_end)
    # 只报和"这个提议"有关的冲突，别把库里的历史问题倒给用户
    issues = issues_touching(detect(everything + probe, rules), candidate.id)
    hard = [i for i in issues if i.severity == "hard"]

    verdict = Verdict(ok=not hard, issues=issues)
    if hard:
        verdict.alternatives = alternatives(others, candidate, rules)
    return verdict


def make_event(title: str, start: str, duration_min: int = 60, event_id: str = "",
               **kwargs) -> Event:
    """从字符串建事件的便捷函数 —— 供 CLI 和将来的语言入口使用。"""
    from .model import _parse_dt   # 局部导入，避免把下划线函数暴露成公开 API

    payload = {
        "title": title,
        "start": _parse_dt(start, "start"),
        "duration_min": duration_min,
        **kwargs,
    }
    if event_id:
        payload["id"] = event_id
    if "weekdays" in payload and isinstance(payload["weekdays"], str):
        payload["weekdays"] = [int(x) for x in payload["weekdays"].replace("，", ",").split(",") if x.strip()]
    if payload.get("until"):
        from .model import _parse_date
        payload["until"] = _parse_date(payload["until"], "until")
    else:
        payload["until"] = None
    event = Event(**payload)
    event.validate()
    event.ensure_id()
    return event
