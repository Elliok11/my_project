"""日程 API 的测试（不启动服务器，用 Flask 自带的测试客户端）

【为什么后端也要测】
    前端的 bug 你看一眼就知道不对（页面白了、按钮没反应）。
    后端的 bug 是**静默的**：接口返回 200 和一坨看起来正常的 JSON，
    但字段名拼错了、日期算偏了一天 —— 界面照常显示，只是显示错了。
    所以接口必须有断言。

    跑：python scheduler/tests/test_api.py
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# 用真实路径 import web/app.py（文件名不叫 app，不能直接 import）
_spec = importlib.util.spec_from_file_location("web_app", ROOT / "web" / "app.py")
web_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(web_app)

import scheduler.store as store_mod          # noqa: E402


def _isolated_store(monkeypatch_path: Path):
    """把日程库指到一个临时文件，别污染用户真实的 data/schedule.jsonl。"""
    original = web_app._schedule_store

    def fake():
        return store_mod.Store(monkeypatch_path)

    web_app._schedule_store = fake
    return original


def _with_client(func):
    """给测试提供一个干净的临时日程库。"""
    import tempfile

    def wrapper():
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "schedule.jsonl"
            original = _isolated_store(path)
            try:
                web_app.app.config["TESTING"] = True
                func(web_app.app.test_client())
            finally:
                web_app._schedule_store = original

    return wrapper


@_with_client
def test_周视图返回七天的骨架(client):
    resp = client.get("/api/schedule/week?date=2026-03-04")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["monday"] == "2026-03-02", data["monday"]
    assert data["sunday"] == "2026-03-08"
    assert len(data["days"]) == 7
    assert data["days"][0]["weekday"] == 1
    assert data["days"][6]["weekday"] == 7


@_with_client
def test_周视图的日期格式错的_要报可读的错(client):
    resp = client.get("/api/schedule/week?date=下周三")
    assert resp.status_code == 400
    assert "日期格式" in resp.get_json()["error"]


@_with_client
def test_解析接口_只解析不落库(client):
    from scheduler.store import Store

    resp = client.post("/api/schedule/parse",
                       json={"text": "下周四晚上七点在东区 A201 开班会"})
    assert resp.status_code == 200
    items = resp.get_json()["items"]
    assert len(items) == 1
    assert items[0]["title"] == "班会"
    assert items[0]["location"] == "东区 A201"
    assert items[0]["ok"] is True

    # 关键断言：解析之后库里应该**还是空的** —— 用户没确认就不能写进去
    week = client.get("/api/schedule/week").get_json()
    assert week["total"] == 0, "解析阶段不该落库"


@_with_client
def test_解析接口_空输入要报错(client):
    resp = client.post("/api/schedule/parse", json={"text": "   "})
    assert resp.status_code == 400


@_with_client
def test_添加接口_落库并返回冲突结果(client):
    resp = client.post("/api/schedule/add", json={"items": [{
        "title": "班会", "start": "2026-03-05T19:00", "duration_min": 60,
        "location": "东区 A201", "category": "meeting",
    }]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total"] == 1
    assert data["added"][0]["ok"] is True, data["added"][0]

    # 再加一件同时间的，应该被报出冲突
    resp = client.post("/api/schedule/add", json={"items": [{
        "title": "临时会", "start": "2026-03-05T19:30", "duration_min": 60,
        "location": "西区",
    }]})
    added = resp.get_json()["added"][0]
    assert added["ok"] is False, "重叠的时间本该被判为冲突"
    assert added["issues"], "冲突应该有说明文字"


@_with_client
def test_添加接口_非法数据进不了库(client):
    resp = client.post("/api/schedule/add", json={"items": [
        {"title": "没时间的会"},
        {"title": "时长是负的", "start": "2026-03-05T10:00", "duration_min": -5},
    ]})
    data = resp.get_json()
    assert len(data["rejected"]) == 2, data
    assert data["total"] == 0, "被拒绝的条目不该进库"


@_with_client
def test_检查接口_冲突时给出替代方案(client):
    client.post("/api/schedule/add", json={"items": [{
        "title": "体育", "start": "2026-03-05T19:00", "duration_min": 90,
        "location": "体育馆",
    }]})
    resp = client.post("/api/schedule/check", json={
        "title": "班会", "start": "2026-03-05T19:30", "duration_min": 60, "location": "东区",
    })
    verdict = resp.get_json()["verdict"]
    assert verdict["ok"] is False
    assert verdict["issues"], "应该报出冲突"
    assert verdict["alternatives"], "冲突时必须给出替代时间"
    for cand in verdict["alternatives"]:
        assert cand["start"] and cand["end"]


@_with_client
def test_重复事件落库时自动补上结束日期(client):
    resp = client.post("/api/schedule/add", json={"items": [{
        "title": "社团例会", "start": "2026-03-04T19:00", "duration_min": 90,
        "weekdays": [3], "location": "西区活动室",
    }]})
    added = resp.get_json()["added"][0]
    assert added["recurrence"] == "weekly"
    assert added["until"], "重复事件必须有结束日期，否则会无限重复"


@_with_client
def test_删除接口(client):
    client.post("/api/schedule/add", json={"items": [{
        "title": "临时", "start": "2026-03-05T10:00", "duration_min": 30,
    }]})
    event_id = client.get("/api/schedule/week?date=2026-03-05").get_json()["days"][3]["items"][0]["id"]

    assert client.post("/api/schedule/delete", json={"id": event_id}).status_code == 200
    assert client.post("/api/schedule/delete", json={"id": event_id}).status_code == 404
    assert client.get("/api/schedule/week?date=2026-03-05").get_json()["total"] == 0


@_with_client
def test_ics_导出(client):
    client.post("/api/schedule/add", json={"items": [{
        "title": "班会", "start": "2026-03-05T19:00", "duration_min": 60, "location": "东区",
    }]})
    resp = client.get("/api/schedule/ics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert body.startswith("BEGIN:VCALENDAR")
    assert "DTSTART:20260305T190000" in body
    assert "SUMMARY:班会" in body
    assert "attachment" in resp.headers["Content-Disposition"]


@_with_client
def test_ics_空日程时明确拒绝(client):
    resp = client.get("/api/schedule/ics")
    assert resp.status_code == 400
    assert "空的" in resp.get_json()["error"]


@_with_client
def test_后端清单接口(client):
    data = client.get("/api/schedule/backends").get_json()
    assert "available" in data and "how_to" in data
    # 没装 OCR / 没配多模态时不报错，只是列表为空 —— 界面据此禁用按钮
    assert isinstance(data["available"], list)


@_with_client
def test_照片接口_后端不可用时明确说清楚_而不是假装成功(client):
    import io

    resp = client.post("/api/schedule/photo",
                       data={"image": (io.BytesIO(b"not-an-image"), "kebiao.jpg"),
                             "backend": "ocr"},
                       content_type="multipart/form-data")
    # 装了 OCR 的话这里可能是 200；没装则必须是 501 + 怎么装的说明
    if resp.status_code == 501:
        data = resp.get_json()
        assert "how_to" in data, "不可用时必须告诉用户怎么开启，而不是只报一句错"
        assert "ocr" in data["how_to"]


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
            failed.append(name)
            print(f"  ❌ {name}\n       {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
