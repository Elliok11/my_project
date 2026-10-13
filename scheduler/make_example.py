"""生成本仓库里可以提交的**样例数据**（不含任何个人真实信息）。

为什么单独放在 scheduler/ 下：`data/` 整个被 .gitignore 排除了
（里面是真实课表和照片），但面试官需要能直接跑起来看效果。
所以必须放在**会被提交**的目录下。

重新生成：
    python -m scheduler.make_example

【样例数据用固定日期，不用 date.today()】
    第一版直接复用了 CLI 的 demo 数据（锚定"本周"），结果生成的文件里
    写满了运行当天的日期 —— 提交进仓库后，过两个月再有人来看，
    满屏都是过去的日期，还以为是坏数据。
    示例数据是**给人读、给文档引用**的，所以必须钉死在周一到周日这一周里。
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from .conflict import Rules, check_range, make_event

OUT = Path(__file__).resolve().parent / "examples" / "schedule_demo.jsonl"

# 固定锚点：2026-03-02 是周一
ANCHOR = date(2026, 3, 2)
UNTIL = "2026-06-30"


def _at(weekday: int, hhmm: str) -> str:
    """这一周的星期几（1=周一）的某个时刻。"""
    return f"{(ANCHOR + timedelta(days=weekday - 1)).isoformat()}T{hhmm}"


def build() -> list:
    """一份"典型大一学生"的样例课表。

    刻意埋了 4 个坑，每一个都对应一个真实日历 App 处理不了的场景：

      ① 周三 10:00 在东区、13:00 在西区 —— 中间 1 小时，够赶路。
         **故意是"不该报"的**：用来证明检查器只抓真问题、不制造噪声。
         一个只会说"你有冲突"的工具没人敢用，一个能分清"这个不用管"的才敢用。

      ② 周四 19:00 体育 与 19:30 班会 —— 硬重叠。
         课表 App 到这里只会画两个叠在一起的色块。

      ③ 周五 15:00 面试（要提前 60 分钟准备 → 14:00 起忙），而 14:00 有线代课
         —— 两件事本身不重叠，重叠的是**准备时间**。

      ④ 周三排了 6 件事 —— 精力过载。只给警告：
         一天排几件算多，只有人自己知道，工具不该替人做决定。
    """
    return [
        # ══════ 每周重复的固定课程 ══════
        make_event("高等数学", _at(1, "08:00"), 110, location="东区 A301",
                   category="course", recurrence="weekly", weekdays=[1], until=UNTIL),
        make_event("人工智能导论", _at(1, "14:00"), 100, location="东区 B202",
                   category="course", recurrence="weekly", weekdays=[1], until=UNTIL),
        make_event("数据结构", _at(3, "10:00"), 120, location="东区 A305",
                   category="course", recurrence="weekly", weekdays=[3], until=UNTIL),
        make_event("英语口语", _at(3, "13:00"), 90, location="西区 C101",
                   category="course", recurrence="weekly", weekdays=[3], until=UNTIL),
        make_event("大学体育", _at(4, "19:00"), 90, location="体育馆",
                   category="course", recurrence="weekly", weekdays=[4], until=UNTIL),
        make_event("线性代数", _at(5, "14:00"), 100, location="东区 A301",
                   category="course", recurrence="weekly", weekdays=[5], until=UNTIL),
        # ══════ 校园活动：周期性、但有终点（这是普通日历最不擅长的）══════
        make_event("社团例会", _at(3, "19:00"), 90, location="西区活动室",
                   category="activity", recurrence="weekly", weekdays=[3], until=UNTIL),
        make_event("英语角", _at(3, "21:00"), 60, location="西区活动室",
                   category="activity", recurrence="weekly", weekdays=[3], until=UNTIL),
        make_event("晨跑打卡", _at(3, "07:00"), 40, location="操场",
                   category="personal", recurrence="weekly", weekdays=[3], until=UNTIL),
        # 周三的第 6 件：既让"精力过载"这条能被触发，
        # 又刻意落在 15:30 这个真实空档里 —— 它本身不该引发任何冲突。
        # （15:30 距离英语口语 14:30 结束有 1 小时，距离社团例会 19:00 还有 3 个半小时）
        make_event("选修：化学与社会", _at(3, "15:30"), 30, location="东区 D105",
                   category="course", recurrence="weekly", weekdays=[3], until=UNTIL),
        make_event("班会", _at(4, "19:30"), 60, location="东区 A201",
                   category="meeting", recurrence="weekly", weekdays=[4], until=UNTIL),
        # ══════ 一次性事件 ══════
        make_event("人工智能协会二面", _at(5, "15:00"), 60, location="线上",
                   category="interview", prep_min=60, consequence="high"),
        make_event("大创报名截止", f"{ANCHOR.isoformat()}T23:00", 30,
                   category="other", consequence="high", note="错过要再等一年"),
    ]


# 这份样例必须能触发下面这些类型的问题，否则文档就成了空话。
EXPECTED_KINDS = {"overlap", "overload"}


def _self_check(events: list) -> None:
    """生成完自检一遍。

    如果不自检，某天改了数据之后它可能悄悄变成一份"没有任何冲突的无聊样例"，
    而 README 还在说"看，它抓到了冲突" —— 文档与数据脱节是最坑后来人的。
    """
    start = min(e.start.date() for e in events)
    issues = check_range(events, start, start + timedelta(days=21), Rules())
    kinds = {i.kind for i in issues}
    missing = EXPECTED_KINDS - kinds
    if missing:
        raise SystemExit(f"❌ 样例数据本该能触发 {missing}，实际只有 {kinds}。请检查数据是否被改动。")
    return kinds


def _check_raw_file(anchor: date) -> None:
    """直接读磁盘上的 JSONL 检查日期锚点。

    【为什么要这一步】
        实测踩到：自检失败时脚本在"写文件之前"就退出了，于是旧文件留在原地，
        看起来一切正常 —— 但里面的日期还是上一版的。后来用 `verify` 扫的时候
        怎么扫都扫不到冲突，白查了一轮。
        教训：**只检查"将要写什么"是不够的，还要检查"磁盘上实际留下了什么"。**
        所以这里绕开数据模型直接读原始 JSON，专门盯"锚点日期对不对"。
    """
    if not OUT.exists():
        return
    for lineno, line in enumerate(OUT.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        start = record["start"]
        day = date.fromisoformat(start[:10])
        if not (anchor <= day <= anchor + timedelta(days=6)):
            raise SystemExit(
                f"❌ {OUT.name} 第 {lineno} 行的日期 {start} 不在锚点周 "
                f"{anchor} ~ {anchor + timedelta(days=6)} 内 —— 文件是旧的，或者锚点被改过。"
            )


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    events = build()
    kinds = _self_check(events)
    _check_raw_file(ANCHOR)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for event in events:
            fh.write(json.dumps({"op": "add", **event.to_dict()}, ensure_ascii=False) + "\n")
    _check_raw_file(ANCHOR)          # 写完再验一遍，确认真的落盘了
    rel = OUT.relative_to(Path(__file__).resolve().parent.parent)
    print(f"✅ 已生成 {len(events)} 条样例日程 → {rel}")
    print(f"   自检通过：能触发 {', '.join(sorted(kinds))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
