"""命令行入口

    python -m scheduler demo                      # 用内置样例跑一遍，看它怎么抓冲突
    python -m scheduler week                      # 看本周完整安排（含重复事件展开）
    python -m scheduler verify                    # 扫未来 4 周，列出所有冲突（自检）
    python -m scheduler add "组会" 2026-03-05T19:00 -d 120 -l 东区
    python -m scheduler add "社团例会" 2026-03-04T19:00 --weekly 3 --until 2026-06-30
    python -m scheduler check "班会" 2026-03-05T19:00 -d 90
    python -m scheduler gaps 2026-03-05
    python -m scheduler ics -o output/schedule.ics

【为什么先做 CLI，不先做网页】
    因为内核要能被测试、能被脚本调用、能被 agent 当工具调。
    网页只是最后一个壳。顺序反了的话，你会把时间花在调 CSS 上。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

# ── Windows 控制台默认是 GBK，输出中文/emoji 会直接抛 UnicodeEncodeError。
#    在 import 阶段就把它顶成 UTF-8，避免"代码没错但一跑就崩"。
#    （这也是踩坑记录里"中文路径"那类问题的同源 cousins：编码只在出问题时才被想起）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")          # Python 3.7+
    except (AttributeError, ValueError):               # 被重定向到不支持 reconfigure 的对象
        pass

from . import conflict, ics
from .conflict import Rules, check_candidate, detect, expand, find_gaps, make_event
from .model import (
    CATEGORIES,
    WEEKDAY_NAMES,
    Event,
    ValidationError,
)
from .store import Store

BAR = "─" * 68


def _today(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def _week_bounds(anchor: date) -> tuple[date, date]:
    """返回 anchor 所在周的周一和周日。"""
    monday = anchor - timedelta(days=anchor.isoweekday() - 1)
    return monday, monday + timedelta(days=6)


# ═══════════════════════════ 各子命令 ═══════════════════════════

def cmd_add(args, store: Store) -> int:
    try:
        event = make_event(
            title=args.title,
            start=args.start,
            duration_min=args.duration,
            event_id=args.id or "",
            location=args.location or "",
            category=args.category,
            consequence=args.consequence,
            recurrence="weekly" if args.weekly else "one_off",
            weekdays=[int(x) for x in (args.weekly or "").replace("，", ",").split(",") if x.strip()],
            until=args.until,
            prep_min=args.prep,
            source="manual",
            note=args.note or "",
        )
    except ValidationError as exc:
        print(f"❌ 事件不合法：{exc}")
        return 2

    # 加之前先裁决 —— 这是"让代码裁决"原则落实到 CLI 上的样子
    verdict = check_candidate(store.all(), event)
    store.add(event)
    print(f"✅ 已加入：{event.describe()}")
    print(verdict.text())
    if not verdict.ok:
        print(f"\n   （已按你的要求写进去了。要改的话可以删掉重加：python -m scheduler rm {event.id}）")
    return 0


def cmd_rm(args, store: Store) -> int:
    if store.remove(args.id):
        print(f"✅ 已删除 {args.id}")
        return 0
    print(f"❌ 没有 id 为 {args.id} 的事件。用 `python -m scheduler list` 看全部。")
    return 1


def cmd_list(args, store: Store) -> int:
    events = store.all()
    if not events:
        print("（还是空的。可以先跑 `python -m scheduler demo` 看效果）")
        return 0
    print(f"共 {len(events)} 条日程规则：\n")
    for event in events:
        print("  " + event.describe())
    return 0


def cmd_week(args, store: Store) -> int:
    anchor = _today(args.date)
    monday, sunday = _week_bounds(anchor)
    occs = expand(store.all(), monday, sunday)
    issues = detect(occs, Rules())
    hard_days = {i.day for i in issues if i.severity == "hard"}

    print(f"\n📅 {monday} ~ {sunday}（第 {anchor.isocalendar().week} 周）")
    print(BAR)
    if not occs:
        print("  这一周没有任何安排。")
    for offset in range(7):
        day = monday + timedelta(days=offset)
        items = [o for o in occs if o.day == day]
        flag = "  ⚠️ 有冲突" if day in hard_days else ""
        head = f"{WEEKDAY_NAMES[day.isoweekday()]} {day:%m-%d}"
        if not items:
            print(f"\n{head}   ——{flag}")
            continue
        print(f"\n{head}{flag}")
        for occ in items:
            tag = f"[{occ.event.category}]"
            where = f"  @{occ.location}" if occ.location else ""
            prep = f"  (需提前 {occ.event.prep_min} 分钟)" if occ.event.prep_min else ""
            print(f"   {occ.start:%H:%M}-{occ.end:%H:%M}  {tag} {occ.title}{where}{prep}")
    print(f"\n{BAR}")
    if issues:
        print(f"本周有 {len([i for i in issues if i.severity == 'hard'])} 条硬冲突、"
              f"{len([i for i in issues if i.severity == 'warn'])} 条提醒：\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 本周没有冲突。")
    return 0


def cmd_verify(args, store: Store) -> int:
    start = _today(args.start) if args.start else date.today()
    end = start + timedelta(days=args.days)
    issues = conflict.check_range(store.all(), start, end, Rules())
    print(f"\n🔍 扫描 {start} ~ {end}（{args.days} 天，{len(store)} 条规则）")
    print(BAR)
    if not issues:
        print("✅ 没有发现任何冲突。")
        return 0
    hard = [i for i in issues if i.severity == "hard"]
    warn = [i for i in issues if i.severity == "warn"]
    for issue in hard + warn:
        print(f"  {issue}")
    print(BAR)
    print(f"合计：硬冲突 {len(hard)} 条，提醒 {len(warn)} 条。")
    return 1 if hard else 0


def cmd_check(args, store: Store) -> int:
    try:
        candidate = make_event(
            title=args.title, start=args.start, duration_min=args.duration,
            location=args.location or "", prep_min=args.prep,
            category=args.category, source="manual",
        )
    except ValidationError as exc:
        print(f"❌ 这个提议本身不合法：{exc}")
        return 2
    verdict = check_candidate(store.all(), candidate, Rules())
    print(f"\n📝 提议：{candidate.describe()}\n")
    print(verdict.text())
    return 0 if verdict.ok else 1


def cmd_gaps(args, store: Store) -> int:
    day = _today(args.date)
    occs = expand(store.all(), day, day)
    gaps = find_gaps(occs, day, Rules(), min_minutes=args.min)
    print(f"\n🕳  {day}（{WEEKDAY_NAMES[day.isoweekday()]}）长度 ≥ {args.min} 分钟的空档：\n")
    if not gaps:
        print("  没有这样的空档。")
    for start, end in gaps:
        minutes = int((end - start).total_seconds() // 60)
        print(f"   {start:%H:%M} - {end:%H:%M}   （{minutes // 60} 小时 {minutes % 60} 分钟）")
    if occs:
        print(f"\n   当天已有 {len(occs)} 件事。")
    return 0


def cmd_ics(args, store: Store) -> int:
    events = store.all()
    try:
        count = ics.write_calendar(events, args.output, args.name)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1
    print(f"✅ 已导出 {count} 条日程 → {args.output}")
    print("   手机导入方法：把这个文件发到手机（微信/邮件都行）→ 点开 → 系统会问是否加入日历。")
    print("   iPhone 也可以放到 iCloud 云盘后用「日历」App 订阅。")
    return 0


def cmd_demo(args, store: Store) -> int:
    """把一组真实校园场景灌进去，直接看它抓冲突。

    这组样例是精心设计的：每一条都对应一个真实的坑。
    """
    from pathlib import Path

    demo_store = Store(Path(store.path).with_name("schedule_demo.jsonl"))
    # 【踩过的坑】第一版没删旧文件，结果 demo 跑第二遍时，
    # 上一版的"线性代数 14:00"和这一版的"线性代数 12:00"同时存在 —— 看起来像 bug，其实是脏数据。
    # 演示数据必须每次重建，否则它自己就会变成一个"看起来有冲突"的假象。
    if demo_store.path.exists():
        demo_store.path.unlink()
    demo_store = Store(demo_store.path)
    for event in _demo_events():
        demo_store.add(event)

    rules = Rules()   # 用和检测同一套规则，保证打印出来的阈值和实际判定一致
    print(f"\n🎬 已生成演示数据 → {demo_store.path}（{len(demo_store)} 条规则，演示前已重置）")
    print("   下面这些都故意设计成有问题的，检验内核抓不抓得住：")
    print(BAR)
    for event in demo_store.all():
        print("  " + event.describe())

    print(f"\n{BAR}\n🔍 未来 4 周的冲突扫描：\n")
    start = date.today()
    issues = conflict.check_range(demo_store.all(), start, start + timedelta(days=28), rules)
    for issue in issues:
        print(f"  {issue}")

    print(f"\n{BAR}")
    if not issues:
        print("⚠️  一条都没抓到 —— 演示数据应该至少能触发 3 类冲突，说明检测逻辑出问题了。")
        return 1
    kinds = sorted({i.kind for i in issues})
    print(f"✅ 抓到 {len(issues)} 条，覆盖 {len(kinds)} 类：{'、'.join(kinds)}")
    print("\n💡 试一试（把 demo 库当数据源）：")
    print("   python -m scheduler -s data/schedule_demo.jsonl week")
    print("   python -m scheduler -s data/schedule_demo.jsonl gaps <某天>")
    print("   python -m scheduler -s data/schedule_demo.jsonl check \"新会议\" 2026-10-14T19:30")
    return 0


def _demo_events() -> list[Event]:
    """演示数据：一个典型大一学生的真实处境。

    刻意埋了 4 个坑，每一条都对应一个真实的、日历 App 处理不了的场景：
      1. 周三 10:00 在东区、13:00 在西区 —— 中间 1 小时，够赶路。**故意是"不该报"的**，
         用来验证赶路检查不会误报（只会抓真问题，才值得信任）
      2. 周四 19:00 体育 与 19:30 班会 —— 硬重叠（课表 App 只会画成两个叠在一起的色块）
      3. 周五 15:00 面试要提前 60 分钟准备（14:00 起忙），而 14:00 有课 —— 准备时间被压
      4. 周三排了 6 件事（含 07:00 的晨跑）—— 精力过载
         （只警告：一天排几件算多，只有人知道，工具不该替人决定）
    """
    today = date.today()
    monday = today - timedelta(days=today.isoweekday() - 1)
    until = monday + timedelta(days=17 * 7)

    def at(weekday: int, hhmm: str) -> str:
        day = monday + timedelta(days=weekday - 1)
        return f"{day.isoformat()}T{hhmm}"

    return [
        # ── 固定课程（每周重复，一整个学期）
        make_event("高等数学", at(1, "08:00"), 110, location="东区 A301",
                   category="course", recurrence="weekly", weekdays=[1], until=until),
        make_event("人工智能导论", at(1, "14:00"), 100, location="东区 B202",
                   category="course", recurrence="weekly", weekdays=[1], until=until),
        # 坑 1：周三 10:00 在东区，13:00 在西区，只有 1 小时但需要 20 分钟 —— 还够
        make_event("数据结构", at(3, "10:00"), 120, location="东区 A305",
                   category="course", recurrence="weekly", weekdays=[3], until=until),
        make_event("英语口语", at(3, "13:00"), 90, location="西区 C101",
                   category="course", recurrence="weekly", weekdays=[3], until=until),
        # 坑 2：周四 19:00 这两个撞在一起
        make_event("大学体育", at(4, "19:00"), 90, location="体育馆",
                   category="course", recurrence="weekly", weekdays=[4], until=until),
        make_event("班会", at(4, "19:30"), 60, location="东区 A201",
                   category="meeting", recurrence="weekly", weekdays=[4], until=until,
                   consequence="normal"),
        # 坑 4：周三再加几件事，凑到 6 件
        make_event("社团例会", at(3, "19:00"), 90, location="西区活动室",
                   category="activity", recurrence="weekly", weekdays=[3], until=until),
        make_event("英语角", at(3, "21:00"), 60, location="西区活动室",
                   category="activity", recurrence="weekly", weekdays=[3], until=until),
        make_event("选修：化学与社会", at(3, "15:30"), 30, location="东区 D105",
                   category="course", recurrence="weekly", weekdays=[3], until=until),
        # 第 6 件：不制造重叠，只为了触发"一天排太多"的警告
        make_event("晨跑打卡", at(3, "07:00"), 40, location="操场",
                   category="personal", recurrence="weekly", weekdays=[3], until=until),
        # 坑 3：面试 15:00 开始、要提前 60 分钟准备 → 14:00 就进入忙碌。
        #      所以线性代数刻意放在 12:00-13:40：不与面试本身重叠，
        #      但正好吃掉准备时间 —— 这样"准备时间"这条才能被单独检验出来。
        make_event("人工智能协会二面", at(5, "15:00"), 60, location="线上",
                   category="interview", prep_min=60, consequence="high"),
        make_event("线性代数", at(5, "12:00"), 100, location="东区 A301",
                   category="course", recurrence="weekly", weekdays=[5], until=until),
        # 一次性事件：报名截止
        make_event("大创报名截止", f"{monday.isoformat()}T23:00", 30,
                   category="other", consequence="high", note="错过要再等一年"),
    ]


# ═══════════════════════════ 参数解析 ═══════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scheduler",
        description="课表/日程内核：冲突检测、找空档、导出日历。不依赖任何大模型。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--store", default=None,
                        help="事件库路径（默认 data/schedule.jsonl）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="生成一组演示数据并跑一遍冲突扫描")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("list", help="列出所有日程规则")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("week", help="看某一周的完整安排（重复事件已展开）")
    p.add_argument("date", nargs="?", help="这一周内的任意一天，默认今天")
    p.set_defaults(func=cmd_week)

    p = sub.add_parser("verify", help="扫描一段时间内的所有冲突")
    p.add_argument("start", nargs="?", help="开始日期，默认今天")
    p.add_argument("--days", type=int, default=28, help="扫描多少天，默认 28")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("gaps", help="找某一天的空档")
    p.add_argument("date", nargs="?", help="哪一天，默认今天")
    p.add_argument("--min", type=int, default=30, help="最短空档时长（分钟），默认 30")
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("check", help="裁决一个提议的时间能不能放")
    p.add_argument("title", help="这件事叫什么")
    p.add_argument("start", help="开始时间，如 2026-03-05T19:00")
    p.add_argument("-d", "--duration", type=int, default=60, help="时长（分钟），默认 60")
    p.add_argument("-l", "--location", default="", help="地点")
    p.add_argument("--prep", type=int, default=0, help="需要提前多少分钟准备")
    p.add_argument("--category", default="other", choices=CATEGORIES)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("add", help="添加一条日程")
    p.add_argument("title", help="这件事叫什么")
    p.add_argument("start", help="开始时间，如 2026-03-05T19:00")
    p.add_argument("-d", "--duration", type=int, default=60, help="时长（分钟），默认 60")
    p.add_argument("-l", "--location", default="", help="地点")
    p.add_argument("--prep", type=int, default=0, help="需要提前多少分钟准备")
    p.add_argument("--category", default="other", choices=CATEGORIES)
    p.add_argument("--consequence", default="normal", choices=["low", "normal", "high"])
    p.add_argument("--note", default="", help="备注")
    p.add_argument("--id", default="", help="指定 id（一般不用填）")
    p.add_argument("--weekly", default="",
                   help="每周重复，填星期几，如 3 表示每周三；多个用逗号：1,3")
    p.add_argument("--until", default=None, help="重复到哪天为止，如 2026-06-30")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("rm", help="删除一条日程")
    p.add_argument("id", help="事件 id（用 list 查看）")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("ics", help="导出 .ics 日历文件")
    p.add_argument("-o", "--output", default="output/schedule.ics", help="输出路径")
    p.add_argument("--name", default="我的日程", help="日历名称")
    p.set_defaults(func=cmd_ics)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = Store(args.store) if args.store else Store()
    try:
        return args.func(args, store)
    except ValidationError as exc:
        print(f"❌ {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
