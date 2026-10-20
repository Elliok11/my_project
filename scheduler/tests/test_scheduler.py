"""内核的单元测试

【为什么要写测试，而不是"跑一遍看着对"】
    这是整个项目里唯一必须绝对可靠的一层 —— 模型可以错，裁判不能错。
    而且排程逻辑有个特点：**错误的结果看起来往往是合理的**。
    比如"周四晚上有空"这句话，不写测试你根本发现不了它漏掉了体育课。

    跑：python -m pytest scheduler/tests -q
    （没有 pytest 也能跑：python scheduler/tests/test_scheduler.py）
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scheduler import ics
from scheduler.conflict import (
    Rules,
    alternatives,
    check_candidate,
    detect,
    expand,
    find_gaps,
    make_event,
    occurrences_of,
)
from scheduler.model import Event, ValidationError, make_id
from scheduler.store import Store

# 一批固定的测试日期，避免"测试在周三通过、在周日失败"这种鬼问题
MONDAY = date(2026, 3, 2)          # 2026-03-02 是周一
WED = MONDAY + timedelta(days=2)   # 周三
THU = MONDAY + timedelta(days=3)   # 周四


def _at(day: date, hhmm: str) -> str:
    return f"{day.isoformat()}T{hhmm}"


# ═══════════════════════════ 模型校验 ═══════════════════════════

def test_接受_range_格式的起始日期():
    event = make_event("高等数学", _at(MONDAY, "08:00"), 110, location="东区 A301")
    assert event.start == datetime(2026, 3, 2, 8, 0)


def test_拒绝_乱七八糟的时间格式():
    for bad in ("下周三下午", "2026/03/02 8点", ""):
        try:
            make_event("x", bad, 60)
        except ValidationError:
            continue
        raise AssertionError(f"{bad!r} 本该被拒绝")


def test_weekly_必须给出_until_否则会无限重复():
    try:
        make_event("例会", _at(WED, "19:00"), 90,
                   recurrence="weekly", weekdays=[3])
    except ValidationError as exc:
        assert "until" in str(exc)
        return
    raise AssertionError("没有 until 的 weekly 事件本该被拒绝")


def test_weekly_的起始日必须落在指定的星期上():
    # 3 月 2 日是周一，却声称"每周三"，这是笔误，必须报错
    try:
        make_event("例会", _at(MONDAY, "19:00"), 90,
                   recurrence="weekly", weekdays=[3], until="2026-06-30")
    except ValidationError as exc:
        assert "周三" in str(exc) or "weekdays" in str(exc)
        return
    raise AssertionError("起始日不在 weekdays 里，本该被拒绝")


def test_时长不能是零或负数():
    for bad in (0, -30):
        try:
            make_event("x", _at(MONDAY, "08:00"), bad)
        except ValidationError:
            continue
        raise AssertionError(f"时长 {bad} 本该被拒绝")


def test_id_是内容决定的_所以重复导出不会产生重复日程():
    """这条是 .ics 的地基：UID 变了，手机日历就会当成新事件，越导越多。"""
    a = make_event("班会", _at(THU, "19:00"), 60, location="东区")
    b = make_event("班会", _at(THU, "19:00"), 60, location="东区")
    assert a.id == b.id
    assert a.id == make_id("班会", datetime(2026, 3, 5, 19, 0), "东区")


# ═══════════════════════════ 周期展开 ═══════════════════════════

def test_每周重复_只在_until_之前展开():
    event = make_event("社团例会", _at(WED, "19:00"), 90, recurrence="weekly",
                       weekdays=[3], until="2026-03-25")
    occs = occurrences_of(event, MONDAY, MONDAY + timedelta(days=60))
    days = [o.day for o in occs]
    # 3/4、3/11、3/18、3/25 共 4 次，3/25 之后就没了
    assert days == [date(2026, 3, 4), date(2026, 3, 11), date(2026, 3, 18), date(2026, 3, 25)]


def test_exdates_把指定的一次从系列里跳过():
    event = make_event("社团例会", _at(WED, "19:00"), 90, recurrence="weekly",
                       weekdays=[3], until="2026-03-25", exdates=["2026-03-11"])
    days = [o.day for o in occurrences_of(event, MONDAY, MONDAY + timedelta(days=30))]
    assert days == [date(2026, 3, 4), date(2026, 3, 18), date(2026, 3, 25)]


def test_多星期的重复规则():
    # 周一 10:00 的"体育"，同时每周四也排 —— 展开后应该周一周四各一次
    event = make_event("体育", _at(MONDAY, "10:00"), 60, recurrence="weekly",
                       weekdays=[1, 4], until="2026-03-15")
    days = [o.day for o in occurrences_of(event, MONDAY, MONDAY + timedelta(days=20))]
    assert days == [date(2026, 3, 2), date(2026, 3, 5), date(2026, 3, 9), date(2026, 3, 12)]


def test_一次性事件只出现一次():
    event = make_event("面试", _at(THU, "15:00"), 60)
    occs = occurrences_of(event, MONDAY, MONDAY + timedelta(days=30))
    assert len(occs) == 1


def test_前置准备时间会把忙碌区间往前推():
    event = make_event("面试", _at(THU, "15:00"), 60, prep_min=60)
    occ = occurrences_of(event, MONDAY, MONDAY + timedelta(days=7))[0]
    assert occ.start == datetime(2026, 3, 5, 15, 0)
    assert occ.busy_start == datetime(2026, 3, 5, 14, 0)


# ═══════════════════════════ 三类冲突 ═══════════════════════════

def test_抓到时间重叠():
    events = [
        make_event("体育", _at(THU, "19:00"), 90, location="体育馆"),
        make_event("班会", _at(THU, "19:30"), 60, location="东区"),
    ]
    issues = detect(expand(events, MONDAY, MONDAY + timedelta(days=7)))
    kinds = [i.kind for i in issues]
    assert "overlap" in kinds
    assert any(i.severity == "hard" for i in issues)


def test_赶路时间不够要报_但同一地点不报():
    tight = [
        make_event("在东区的课", _at(WED, "10:00"), 120, location="东区"),
        make_event("在西区的会", _at(WED, "12:05"), 60, location="西区"),
    ]
    issues = detect(expand(tight, MONDAY, MONDAY + timedelta(days=7)))
    assert any(i.kind == "travel" for i in issues), "跨校区只留 5 分钟，本该报赶不上"

    same = [
        make_event("在东区的课", _at(WED, "10:00"), 120, location="东区"),
        make_event("还在东区的会", _at(WED, "12:05"), 60, location="东区"),
    ]
    issues = detect(expand(same, MONDAY, MONDAY + timedelta(days=7)))
    assert not any(i.kind == "travel" for i in issues), "同一楼里换教室不该报赶不上"


def test_回归_重叠时不再重复报一条负数的赶不上():
    """这是第一次跑 demo 时发现的真 bug：
    两件事重叠，却还额外报一条"只有 -60 分钟"—— 同一个事实报两遍就是噪声。"""
    events = [
        make_event("体育", _at(THU, "19:00"), 90, location="体育馆"),
        make_event("班会", _at(THU, "19:30"), 60, location="东区"),
    ]
    issues = detect(expand(events, MONDAY, MONDAY + timedelta(days=7)))
    travels = [i for i in issues if i.kind == "travel"]
    assert not travels, f"重叠的情况下不该再报赶路：{[str(t) for t in travels]}"


def test_回归_准备时间冲突和普通重叠是不同的病():
    """面试 15:00 要提前 60 分钟准备 → 14:00 起忙；14:00-14:30 有另一件事。
    两者本体不重叠，但准备时间被占 —— 必须报 prep，且不能报成 overlap。"""
    events = [
        make_event("某件小事", _at(THU, "14:00"), 30, location="东区"),
        make_event("面试", _at(THU, "15:00"), 60, prep_min=60),
    ]
    issues = detect(expand(events, MONDAY, MONDAY + timedelta(days=7)))
    kinds = {i.kind for i in issues}
    assert "prep" in kinds, f"本该报准备时间冲突，实际：{kinds}"
    assert "overlap" not in kinds, "本体没重叠，不该报 overlap"


def test_一天排太多只警告不阻止():
    events = [
        make_event(f"第{i}件事", _at(WED, f"{8 + i:02d}:00"), 30, location="东区")
        for i in range(6)
    ]
    issues = detect(expand(events, MONDAY, MONDAY + timedelta(days=7)), Rules(max_events_per_day=5))
    overload = [i for i in issues if i.kind == "overload"]
    assert overload and overload[0].severity == "warn", "过载应该是警告，不是硬冲突"


def test_没有冲突时就是没有():
    events = [
        make_event("早课", _at(MONDAY, "08:00"), 100, location="东区"),
        make_event("午课", _at(MONDAY, "14:00"), 100, location="东区"),
    ]
    assert detect(expand(events, MONDAY, MONDAY + timedelta(days=7))) == []


# ═══════════════════════════ 找空档 ═══════════════════════════

def test_空档计算把赶路时间也算成忙():
    events = [
        make_event("课", _at(WED, "10:00"), 60, location="东区"),
        # 11:05 在西区 —— 只有 5 分钟，赶不到，所以 11:00-11:05 这段不该被算成空档
        make_event("会", _at(WED, "11:05"), 60, location="西区"),
    ]
    occs = expand(events, MONDAY, MONDAY + timedelta(days=7))
    gaps = find_gaps(occs, WED, Rules(), min_minutes=5)
    assert not any(s < datetime(2026, 3, 4, 11, 0) < e for s, e in gaps), "被赶路占掉的 5 分钟不该算空档"


def test_空档之间的间隔是准的():
    events = [make_event("课", _at(WED, "10:00"), 60, location="东区")]
    occs = expand(events, MONDAY, MONDAY + timedelta(days=7))
    gaps = find_gaps(occs, WED, Rules(day_start=datetime.min.time().replace(hour=8),
                                      day_end=datetime.min.time().replace(hour=22)), min_minutes=60)
    assert gaps[0] == (datetime(2026, 3, 4, 8, 0), datetime(2026, 3, 4, 10, 0))
    assert gaps[-1][1] == datetime(2026, 3, 4, 22, 0)


# ═══════════════════════════ 裁决与替代方案 ═══════════════════════════

def test_裁决_能放的提议返回_ok():
    events = [make_event("课", _at(WED, "10:00"), 60, location="东区")]
    verdict = check_candidate(events, make_event("新会", _at(WED, "19:00"), 60, location="东区"))
    assert verdict.ok and not verdict.issues


def test_裁决_冲突的提议给出替代方案():
    events = [make_event("体育", _at(THU, "19:00"), 90, location="体育馆")]
    verdict = check_candidate(events, make_event("班会", _at(THU, "19:30"), 60, location="东区"))
    assert not verdict.ok
    assert verdict.alternatives, "冲突时必须给出至少一个替代时间"
    for cand in verdict.alternatives:
        # 每一个替代方案都必须真的是空的 —— 否则就是在骗人
        probe = make_event("班会", cand.start.strftime("%Y-%m-%dT%H:%M"), 60, location="东区")
        assert check_candidate(events, probe).ok, f"替代方案 {cand.describe()} 其实也冲突"


def test_裁决_重复事件会检查到未来每一周():
    """只看起始日是不够的：起始日那天刚好没课，但下周三有课 —— 必须也能发现。"""
    events = [
        make_event("体育", _at(THU, "19:00"), 90, location="体育馆", recurrence="weekly",
                   weekdays=[4], until="2026-06-30"),
    ]
    # 第一周周四排满了？不，这里故意从"下周四"开始，也要抓到
    candidate = make_event("班会", _at(THU + timedelta(days=7), "19:30"), 60,
                           location="东区", recurrence="weekly", weekdays=[4],
                           until="2026-05-30")
    verdict = check_candidate(events, candidate)
    assert not verdict.ok, "第二周才撞上的重复事件也必须被发现"


def test_回归_裁决只报和这次提议有关的冲突_不报库里原有的问题():
    """实测发现的可用性 bug：
    裁决"把班会改到周三"时，输出里混进了库里**原本就存在**的"周四体育撞班会"，
    用户会以为是这次改动引起的。裁决就该只管这一次提议。"""
    events = [
        make_event("体育", _at(THU, "19:00"), 90, location="体育馆"),
        make_event("班会", _at(THU, "19:30"), 60, location="东区"),      # 库里本来就有冲突
    ]
    # 提议一件和上面两件都无关的新事，且它自己不冲突
    verdict = check_candidate(events, make_event("买书", _at(WED, "15:00"), 60, location="书店"))
    assert verdict.ok, "这件新事本身没冲突，不该被判不行"
    assert verdict.issues == [], f"不该把库里原有的冲突倒出来：{[str(i) for i in verdict.issues]}"


def test_替代方案不会推荐已经排爆的那一天():
    """实测发现：不加这条约束时，程序会推荐 22:15 这种深夜时段，
    而那天本来就排了 7 件事 —— 数学上不冲突，但人不会接受。"""
    busy_day = [
        make_event(f"第{i}件事", _at(WED, f"{9 + i:02d}:00"), 30, location="东区")
        for i in range(6)      # 周三已经 6 件，超过上限 5
    ]
    cands = alternatives(busy_day, make_event("插进来的会", _at(WED, "15:30"), 30, location="东区"),
                         lookahead_days=3)
    assert all(c.start.date() != WED for c in cands), \
        f"不该往已经排爆的周三塞：{[c.describe() for c in cands]}"


def test_替代方案优先选改动最小的():
    events = [
        make_event("体育", _at(THU, "19:00"), 90, location="体育馆"),
        make_event("讲座", _at(THU, "21:00"), 60, location="东区"),
    ]
    cands = alternatives(events, make_event("班会", _at(THU, "20:00"), 30, location="东区"),
                         lookahead_days=3)
    assert cands, "这么大的空档，应该能找到替代"
    assert cands[0].preferred, "同一天内的方案应该排在最前面（改动最小）"


# ═══════════════════════════ 存储 ═══════════════════════════

def test_存储_是追加式的_能看出改了什么(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "schedule.jsonl"
        store = Store(path)
        store.add(make_event("A", _at(MONDAY, "08:00"), 60))
        store.add(make_event("B", _at(MONDAY, "10:00"), 60))

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2, "每次添加只该追加一行，而不是重写整个文件"

        # 重开一次，数据要能完整恢复
        reloaded = Store(path)
        assert len(reloaded) == 2
        assert {e.title for e in reloaded.all()} == {"A", "B"}


def test_存储_删除是软删除_流水里仍留痕(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "schedule.jsonl"
        store = Store(path)
        event = store.add(make_event("临时会议", _at(MONDAY, "08:00"), 60))
        assert store.remove(event.id)
        assert len(Store(path)) == 0
        assert "remove" in path.read_text(encoding="utf-8")


# ═══════════════════════════ .ics 导出 ═══════════════════════════

def test_ics_行尾必须是_CRLF():
    content = ics.build_calendar([make_event("班会", _at(THU, "19:00"), 60)])
    assert content.endswith("\r\n")
    assert "\r\r\n" not in content, "不能出现双重回车"
    assert content.count("BEGIN:VEVENT") == 1


def test_ics_重复事件用_RRULE_并带上_BYDAY():
    event = make_event("社团例会", _at(WED, "19:00"), 90, recurrence="weekly",
                       weekdays=[3], until="2026-06-30")
    content = ics.build_calendar([event])
    assert "RRULE:FREQ=WEEKLY;BYDAY=WE;UNTIL=20260630T235959" in content


def test_ics_时间是浮动本地时间_不带_Z():
    """带 Z 会被当成 UTC，导入手机后可能差 8 小时 —— 这是最经典的 .ics 坑。"""
    content = ics.build_calendar([make_event("班会", _at(THU, "19:00"), 60)])
    assert "DTSTART:20260305T190000" in content
    assert "DTSTART:20260305T190000Z" not in content


def test_ics_中文标题的超长行会被正确折行():
    """对**每一种长度**都试一遍。

    为什么必须穷举：折行的 bug 只在"切点刚好落在字符边界上"时出现。
    只测一两个长度，很可能整片都躲开了出问题的那几个 —— 第一版就是这么漏过去的。
    而且这个 bug 很坏：字节总数一个不差，所以长度检查完全发现不了它。
    """
    for repeat in range(1, 90):
        title = "第" * repeat + "章"
        content = ics.build_calendar([make_event(title, _at(THU, "19:00"), 60)])

        for line in content.split("\r\n"):
            assert len(line.encode("utf-8")) <= 75, f"标题重复 {repeat} 次：有行超过 75 字节"

        # 折行是"物理换行"，展开后必须和没折行的内容一模一样 —— 一个字节都不能丢
        unfolded = content.split("\r\n")[0]
        assert title in content.replace("\r\n ", ""), f"标题重复 {repeat} 次：折行后标题丢了或坏了"


def test_ics_折行不会丢字节也不会坏字符():
    """直接盯住 _fold：折完再拼回去，必须和原始行逐字节相同。"""
    for repeat in range(1, 120):
        original = "SUMMARY:" + "第" * repeat
        folded = ics._fold(original)
        restored = folded.replace("\r\n ", "")
        assert restored == original, f"重复 {repeat} 次时折行损坏了内容"
        for part in folded.split("\r\n"):
            assert len(part.encode("utf-8")) <= 75, f"重复 {repeat} 次时折行后仍有超长行"


def test_ics_混排中英文与emoji也不出错():
    for text in ("人工智能协会二面（最终轮）", "AI协会面试🎯", "a中b文c混d排" * 20):
        content = ics.build_calendar([make_event(text, _at(THU, "19:00"), 60)])
        for line in content.split("\r\n"):
            assert len(line.encode("utf-8")) <= 75, f"{text!r} 折行后有超长行"


def test_ics_文本里的特殊字符被转义():
    content = ics.build_calendar([make_event("会议; 关于, 排程", _at(THU, "19:00"), 60)])
    assert "SUMMARY:会议\\; 关于\\, 排程" in content


def test_ics_没有事件时明确拒绝_而不是写个坏文件():
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        try:
            ics.write_calendar([], Path(folder) / "empty.ics")
        except ValueError:
            assert not (Path(folder) / "empty.ics").exists(), "被拒绝时不该留下半个文件"
            return
    raise AssertionError("空日历本该被拒绝")


def test_ics_SU_对应周日():
    """SU=周日 是最容易搞反的一个（isoweekday 里周日是 7）。"""
    event = make_event("周日活动", _at(date(2026, 3, 2) + timedelta(days=6), "10:00"), 60,
                       recurrence="weekly", weekdays=[7], until="2026-06-30")
    assert "BYDAY=SU" in ics.build_calendar([event])


# ═══════════════════════════ 手动跑（无 pytest 时） ═══════════════════════════

def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, func in tests:
        try:
            func()
            print(f"  ✅ {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed.append((name, exc))
            print(f"  ❌ {name}\n       {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
