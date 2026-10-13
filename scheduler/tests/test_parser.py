"""自然语言解析的测试

【为什么这一层特别需要测试】
    "下周四"到底算哪天，是一个**有唯一正确答案**的问题。
    没有测试的话，只能靠手动敲一句看一眼 —— 而人眼对日期错误极不敏感：
    2026-03-12 和 2026-03-19 看上去都"挺合理"。
    所以这里把 today 钉死，逐条断言。
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scheduler import parser
from scheduler.parser import (
    parse_text,
    parse_with_llm,
    resolve_date,
    to_event,
)
from scheduler.model import ValidationError

# 2026-03-04 是周三（用过的同学可以自己核：3/2 周一、3/3 周二、3/4 周三）
WED = date(2026, 3, 4)


# ═══════════════════════════ 日期解析（核心）═══════════════════════════

def test_今天明天后天():
    assert resolve_date({"type": "relative", "offset": 0}, WED) == date(2026, 3, 4)
    assert resolve_date({"type": "relative", "offset": 1}, WED) == date(2026, 3, 5)
    assert resolve_date({"type": "relative", "offset": 2}, WED) == date(2026, 3, 6)


def test_这周几_与_下周几():
    """注意"这周"的语义：今天已经过去的那天，说"这周X"其实指的是下周。

    今天 3/4 是周三，本周一是 3/2 —— 已经过去了，所以"这周一"应该是 3/9。
    这是人的真实语感，不是 bug。
    """
    # 本周还没到 / 就是今天的
    assert resolve_date({"type": "weekday", "weekday": 3, "week": "this"}, WED) == date(2026, 3, 4)
    assert resolve_date({"type": "weekday", "weekday": 5, "week": "this"}, WED) == date(2026, 3, 6)
    # 本周已经过去的（周一 3/2）→ 顺延到 3/9
    assert resolve_date({"type": "weekday", "weekday": 1, "week": "this"}, WED) == date(2026, 3, 9)
    # "下周X" 永远是下周，不受这个规则影响
    assert resolve_date({"type": "weekday", "weekday": 4, "week": "next"}, WED) == date(2026, 3, 12)
    assert resolve_date({"type": "weekday", "weekday": 1, "week": "after_next"}, WED) == date(2026, 3, 16)


def test_星期一_当天说这周一_就是今天():
    monday = date(2026, 3, 2)
    assert resolve_date({"type": "weekday", "weekday": 1, "week": "this"}, monday) == monday


def test_说这周三但今天已经周四_应该指下周():
    """人的口语："这周三"在周四说出来，指的一定是下周三 —— 不是昨天。"""
    thursday = date(2026, 3, 5)
    assert resolve_date({"type": "weekday", "weekday": 3, "week": "this"}, thursday) == date(2026, 3, 11)


def test_绝对日期_月日():
    assert resolve_date({"type": "absolute", "month": 3, "day": 20}, WED) == date(2026, 3, 20)
    assert resolve_date({"type": "absolute", "month": 6, "day": 1}, WED) == date(2026, 6, 1)
    # 已经过去的日期 → 顺延到明年（排日程时说的日期一定是"下一次"）
    assert resolve_date({"type": "absolute", "month": 1, "day": 5}, WED) == date(2027, 1, 5)
    assert resolve_date({"type": "absolute", "month": 3, "day": 1}, WED) == date(2027, 3, 1)


def test_不存在的日期返回_None_而不是崩掉():
    assert resolve_date({"type": "absolute", "month": 2, "day": 30}, WED) is None
    assert resolve_date({"type": "weekday", "weekday": 9, "week": "this"}, WED) is None
    assert resolve_date(None, WED) is None


# ═══════════════════════════ 时间解析 ═══════════════════════════

def test_24小时制():
    item = parse_text("10月20号 15:00 面试", WED)[0]
    assert item.start == datetime(2026, 10, 20, 15, 0)


def test_中文点数词():
    item = parse_text("明天晚上七点在楼下集合", WED)[0]
    assert item.start == datetime(2026, 3, 5, 19, 0)


def test_下午三点_是_15点():
    item = parse_text("后天下午三点开会", WED)[0]
    assert item.start == datetime(2026, 3, 6, 15, 0)


def test_没写时段的三点_按下午算():
    """口语里"三点开会"指的一定不是凌晨三点。"""
    item = parse_text("周五三点交材料", WED)[0]
    assert item.start == datetime(2026, 3, 6, 15, 0)


def test_上午九点不需要修正():
    item = parse_text("周四上午九点体检", WED)[0]
    assert item.start == datetime(2026, 3, 5, 9, 0)


def test_点半():
    item = parse_text("下周三下午三点半组会", WED)[0]
    assert item.start == datetime(2026, 3, 11, 15, 30)


def test_房间号里的数字不该被当成时间():
    """真实课表里全是 "东区 A301""B202" 这种。把 301 当时间就全毁了。"""
    item = parse_text("周一 8:00 高等数学 东区 A301", WED)[0]
    # 3/4 是周三，这时说"周一"指的是 3/9（本周一已经过去了）
    assert item.start == datetime(2026, 3, 9, 8, 0), f"应该是 3/9 的 8:00，实际 {item.start}"
    assert item.location == "东区 A301"


# ═══════════════════════════ 时长与地点 ═══════════════════════════

def test_时长_小时与分钟():
    assert parse_text("明天下午两点面试，大概两小时", WED)[0].duration_min == 120
    assert parse_text("今晚八点组会，45分钟", WED)[0].duration_min == 45
    assert parse_text("周三下午三点开会 1小时", WED)[0].duration_min == 60


def test_半小时():
    """实测漏过："半小时"里的"半"不在中文数字表里，于是整个时长没认出来，
    还混进了标题（"班会 半小时准备"）。"""
    assert parse_text("今晚八点组会，半小时", WED)[0].duration_min == 30
    assert parse_text("明天下午两点面试，半个小时", WED)[0].duration_min == 30


def test_半小时的准备时间():
    item = parse_text("下周四晚上七点开班会，提前半小时准备", WED)[0]
    assert item.prep_min == 30, f"准备时间应该是 30 分钟，实际 {item.prep_min}"
    assert item.title == "班会", f"标题不该带'半小时准备'，实际 {item.title!r}"


def test_时间区间直接算出时长():
    """写了结束时间就等于给了时长，别白白丢掉用户已经说出口的信息。"""
    item = parse_text("10月20号 15:00-17:00 组会 西区活动室", WED)[0]
    assert item.duration_min == 120, f"15:00-17:00 应该是 120 分钟，实际 {item.duration_min}"
    assert item.title == "组会", f"标题不该残留结束时间，实际 {item.title!r}"
    assert item.location == "西区活动室"


def test_准备时间不算进事件时长():
    """'提前一小时准备'里的一小时是准备时间，不是事件时长。两者混了就是两个信息全错。"""
    item = parse_text("周五下午两点面试，提前一小时准备", WED)[0]
    assert item.prep_min == 60
    assert item.duration_min == 60, "事件时长应该还是默认的 60，不是那个准备时间"


def test_默认时长是一小时():
    assert parse_text("明天下午两点面试", WED)[0].duration_min == 60


def test_地点_在字后面():
    item = parse_text("下周四晚上七点在东区 A201 开班会", WED)[0]
    assert item.location == "东区 A201"
    assert item.title == "班会"


def test_地点_没有在字时靠常见地点词兜底():
    assert parse_text("周五 14:00 数据结构 东区 A305", WED)[0].location == "东区 A305"
    assert parse_text("明天十点体育馆体检", WED)[0].location == "体育馆"


# ═══════════════════════════ 标题 ═══════════════════════════

def test_标题剥掉了时间和日期():
    item = parse_text("下周四晚上七点在东区 A201 开班会", WED)[0]
    assert item.title == "班会", f"实际是 {item.title!r}"


def test_没写日期时如实报缺():
    item = parse_text("找个时间开班会", WED)[0]
    assert item.start is None
    assert "开始时间" in item.missing
    assert not item.ok


def test_没写事件名时如实报缺():
    item = parse_text("明天下午三点", WED)[0]
    assert "事件名称" in item.missing


# ═══════════════════════════ 重复规则 ═══════════════════════════

def test_每周三是重复事件():
    item = parse_text("每周三 19:00 社团例会 西区活动室", WED)[0]
    assert item.weekdays == [3]
    assert item.until, "重复事件必须自动补一个结束日期，否则会无限重复"
    assert item.start.isoweekday() == 3


def test_每周多个星期几():
    item = parse_text("每周一和周四 10:00 体育", WED)[0]
    assert item.weekdays == [1, 4]


def test_一次输入里多件事():
    items = parse_text("周三下午两点组会；周五上午十点面试", WED)
    assert len(items) == 2
    assert items[0].start == datetime(2026, 3, 4, 14, 0)
    assert items[1].start == datetime(2026, 3, 6, 10, 0)


# ═══════════════════════════ 类别猜测 ═══════════════════════════

def test_类别按关键词猜():
    assert parse_text("明天下午三点面试", WED)[0].category == "interview"
    assert parse_text("下周四晚上七点班会", WED)[0].category == "meeting"
    assert parse_text("周三 10:00 数据结构课", WED)[0].category == "course"
    assert parse_text("周五 19:00 社团招新", WED)[0].category == "activity"


# ═══════════════════════════ 变成内核事件 ═══════════════════════════

def test_能落库成内核认识的_Event():
    item = parse_text("下周四晚上七点在东区 A201 开班会", WED)[0]
    event = to_event(item)
    assert event.start == datetime(2026, 3, 12, 19, 0)
    assert event.location == "东区 A201"
    assert event.source == "text", "来源必须记成 text —— 第 5 天的 A/B 实验要靠它分组成绩"
    assert event.recurrence == "one_off"


def test_重复事件落库时带上_weekdays_和_until():
    item = parse_text("每周三 19:00 社团例会 西区活动室", WED)[0]
    event = to_event(item)
    assert event.recurrence == "weekly"
    assert event.weekdays == [3]
    assert event.until is not None


def test_没有开始时间的解析结果不许落库():
    item = parse_text("找个时间开班会", WED)[0]
    try:
        to_event(item)
    except ValidationError:
        return
    raise AssertionError("缺开始时间的事件本该被拒绝落库")


# ═══════════════════════════ 模型解析（用假的 ask，不依赖 Ollama）═══════════════════════════

def test_模型抽取_日期仍然由代码算():
    """关键：模型只给"零件"，日期必须由 resolve_date 算出来。

    如果哪天有人改成"让模型直接给日期字符串"，这条测试会抓出来 ——
    因为模型给的是零件，不是日期。
    """
    def fake_ask(prompt: str) -> str:
        return ('{"title":"社团例会","date":{"type":"weekday","weekday":4,"week":"next"},'
                '"time":{"hour":19,"minute":0},"duration_min":90,'
                '"location":"西区活动室","recurring":false}')

    item = parse_with_llm("下周四七点社团例会", WED, ask=fake_ask)[0]
    assert item.title == "社团例会"
    assert item.start == datetime(2026, 3, 12, 19, 0), f"日期应由代码算，实际 {item.start}"
    assert item.duration_min == 90
    assert item.location == "西区活动室"


def test_模型输出带废话和代码块也能抠出来():
    def messy_ask(prompt: str) -> str:
        return ('好的，我来分析一下：\n```json\n'
                '{"title":"买书","date":{"type":"relative","offset":1},'
                '"time":{"hour":15,"minute":30},"duration_min":30,"location":"书店"}\n'
                '```\n以上是我的抽取结果。')

    item = parse_with_llm("明天下午三点半去买书", WED, ask=messy_ask)[0]
    assert item.title == "买书"
    assert item.start == datetime(2026, 3, 5, 15, 30)


def test_模型给出垃圾时退回规则解析():
    def bad_ask(prompt: str) -> str:
        return "我不知道该怎么抽取这个。"

    item = parse_with_llm("明天下午三点在楼下集合", WED, ask=bad_ask)[0]
    assert item.start == datetime(2026, 3, 5, 15, 0), "模型没用时，规则解析应该顶上"
    assert "退回规则解析" in item.note


def test_模型和规则互补_模型漏了标题就用规则的():
    def half_ask(prompt: str) -> str:
        # 模型只给了时间，没给标题
        return '{"title":"","date":{"type":"relative","offset":1},"time":{"hour":9,"minute":0}}'

    item = parse_with_llm("明天上午九点体检", WED, ask=half_ask)[0]
    assert item.title == "体检", "模型没给标题时，应该用规则解析的结果补上"
    assert item.start == datetime(2026, 3, 5, 9, 0)


# ═══════════════════════════ 手动跑 ═══════════════════════════

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
