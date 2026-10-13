"""事件存储 —— 追加式 JSONL（一行一个事件）

【为什么不是一个普通的 JSON 文件，也不是数据库】
    1. 不用 SQLite：几 KB 的数据，用数据库是给自己找麻烦，而且没法用 git 看 diff。
    2. 不用单个 JSON 文件：那样每次改动都会把整个文件重写一遍，
       git diff 会变成"整个文件都变了"，看不出你到底改了什么。
    3. 用追加式 JSONL：每次操作加一行，**只新增、不重写**。
       → `git log -p data/schedule.jsonl` 能清楚看到"哪天加了哪件事"
       → 天然是一份操作流水，出错了能手工改

【冲突的关键词】
    "课程表"这个词自带一个陷阱：它既指"我有哪些课"（数据），
    也指"我每周的具体安排"（数据的展开）。本模块只管前者。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .model import Event, ValidationError

DEFAULT_STORE = Path(__file__).resolve().parent.parent / "data" / "schedule.jsonl"


class Store:
    """事件库。内存里保存一个 id → Event 的字典，磁盘上是一行行 JSON。"""

    def __init__(self, path: str | Path = DEFAULT_STORE):
        self.path = Path(path)
        self._events: dict[str, Event] = {}
        self.reload()

    # ────────────────────────── 读写 ──────────────────────────

    def reload(self) -> None:
        """从磁盘重放整个流水。任何一行坏掉都明确报错 —— 不静默跳过。"""
        self._events = {}
        if not self.path.exists():
            return
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{self.path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
            if record.get("op") == "remove":
                self._events.pop(record.get("id", ""), None)
                continue
            event = Event.from_dict(record)
            event.ensure_id()
            self._events[event.id] = event

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ────────────────────────── 增删查 ──────────────────────────

    def add(self, event: Event) -> Event:
        event.ensure_id()
        event.validate()
        self._events[event.id] = event
        self._append({"op": "add", **event.to_dict()})
        return event

    def remove(self, event_id: str) -> bool:
        if event_id not in self._events:
            return False
        del self._events[event_id]
        self._append({"op": "remove", "id": event_id})
        return True

    def get(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def all(self) -> list[Event]:
        """按开始时间排序返回全部事件。排序是为了输出稳定，便于测试和 diff。"""
        return sorted(self._events.values(), key=lambda e: (e.start, e.title))

    def active_on(self, day: date) -> list[Event]:
        """在这一天"有效"的事件（一次性事件只在当天有效，weekly 要看范围）。"""
        result = []
        for event in self.all():
            if event.recurrence == "weekly":
                if event.start.date() <= day <= event.end_date():
                    result.append(event)
            elif event.start.date() == day:
                result.append(event)
        return result

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self.all())
