"""本地小助手 —— Web 界面后端（Flask）

【两个页面】
    ① 对话     → 本地大模型（Ollama + qwen2.5:1.5b），流式输出
    ② 日程表   → 自然语言 / 手动录入，冲突检测、周视图、导出 .ics

【启动】
    python web/app.py
    然后浏览器打开 http://127.0.0.1:5000

【设计说明】
    - 日程内核不调用模型：解析 → 预览 → 用户确认 → 才落库
    - Vue 用本地文件（static/vue.global.prod.js），不依赖 CDN
    - 不保存任何上传文件：日程只存结构化字段
"""
import json
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# ── 路径 ──
ROOT = Path(__file__).resolve().parent.parent
# 把项目根目录加进 sys.path，才能 `from scheduler.xxx import ...`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 配置 ──
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"

app = Flask(__name__)


# ────────────────────── 统一错误处理 ──────────────────────
# 关键：Flask 默认对 413/500 返回 HTML 错误页，而前端是拿 JSON.parse 去解析的，
# 结果会报出很误导人的 "Unexpected token '<'"。
# 所以这里把所有错误统一改成返回 JSON，前端就能显示可读的原因。
@app.errorhandler(HTTPException)
def handle_http_error(exc: HTTPException):
    return jsonify({"error": f"{exc.code} {exc.name}：{exc.description}"}), exc.code


@app.errorhandler(Exception)
def handle_unexpected(exc: Exception):
    traceback.print_exc()
    return jsonify({"error": f"服务器内部错误：{exc}"}), 500


def ask_ollama(prompt: str, stream: bool = False):
    """调用本地大模型。stream=True 时返回逐块生成器。"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": stream}
    if stream:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        def gen():
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    yield piece

        return gen()

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


# ────────────────────── 页面 ──────────────────────
@app.get("/")
def index():
    # 直接以静态文件方式返回前端页面（不走 Jinja2 模板渲染）。
    # 原因：Flask 的 Jinja2 和 Vue 都用 {{ }} 做插值，会互相冲突。
    return app.send_static_file("index.html")


@app.get("/api/status")
def api_status():
    """给前端做健康检查：Ollama 通不通、模型在不在。"""
    try:
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        ollama_ok = True
    except Exception:
        models, ollama_ok = [], False

    return jsonify({
        "ollama": ollama_ok,
        "models": models,
        "model": MODEL_NAME,
    })


# ────────────────────── ① 对话 ──────────────────────
@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    system = (data.get("system") or "").strip()
    if not prompt:
        return jsonify({"error": "问题不能为空"}), 400
    if system:
        prompt = f"{system}\n\n{prompt}"

    def generate():
        try:
            yield from ask_ollama(prompt, stream=True)
        except requests.exceptions.ConnectionError:
            yield "\n\n[错误] 连不上本地 Ollama。请先运行：ollama serve"
        except Exception as exc:                       # noqa: BLE001
            yield f"\n\n[错误] {exc}"

    return app.response_class(generate(), mimetype="text/plain; charset=utf-8")


# ══════════════════════════════════════════════════════════════════
#  ② 日程表（scheduler）
#
#  设计：解析器只抽结构化字段 → 用户**肉眼确认** → 才落库。
#  日程一旦错了，人是按着它去开会的，所以中间必须有一道人工确认。
#  冲突检测与替代方案全部由 scheduler/conflict.py 裁决，不调用模型。
# ══════════════════════════════════════════════════════════════════

SCHEDULE_FILE = ROOT / "data" / "schedule.jsonl"


def _schedule_store():
    """日程库。放在 data/ 下（已被 .gitignore 排除，不会把个人日程提交上去）。

    每次请求都重新读文件 —— 几百条数据读一遍只要毫秒级，
    但换来一个重要的性质：用 CLI（`python -m scheduler ... add`）改完，
    网页刷新一下就能看到，不会读到进程启动时的旧快照。
    """
    from scheduler.store import Store

    return Store(SCHEDULE_FILE)


def _event_json(event) -> dict:
    return {
        "id": event.id,
        "title": event.title,
        "start": event.start.strftime("%Y-%m-%dT%H:%M"),
        "duration_min": event.duration_min,
        "location": event.location,
        "category": event.category,
        "recurrence": event.recurrence,
        "weekdays": list(event.weekdays),
        "until": event.until.isoformat() if event.until else "",
        "exdates": [d.isoformat() for d in event.exdates],
        "prep_min": event.prep_min,
        "consequence": event.consequence,
        "source": event.source,
        "note": event.note,
    }


@app.get("/api/schedule/week")
def api_schedule_week():
    """返回某一周的完整安排 + 这一周的冲突列表。

    前端只要给一个"这一周里的任意一天"，后端负责算出周一到周日。
    """
    from datetime import date as _date
    from datetime import timedelta

    from scheduler.conflict import Rules, detect, expand

    raw = (request.args.get("date") or "").strip()
    try:
        anchor = _date.fromisoformat(raw) if raw else _date.today()
    except ValueError:
        return jsonify({"error": f"日期格式不对：{raw}。请用 2026-03-02 这种格式。"}), 400

    monday = anchor - timedelta(days=anchor.isoweekday() - 1)
    sunday = monday + timedelta(days=6)
    rules = Rules()
    store = _schedule_store()
    events = store.all()
    occurrences = expand(events, monday, sunday)
    issues = detect(occurrences, rules)

    by_day: dict[str, list] = {}
    for occ in occurrences:
        by_day.setdefault(occ.day.isoformat(), []).append({
            **_event_json(occ.event),
            "occ_start": occ.start.strftime("%H:%M"),
            "occ_end": occ.end.strftime("%H:%M"),
        })

    return jsonify({
        "monday": monday.isoformat(),
        "sunday": sunday.isoformat(),
        "today": _date.today().isoformat(),
        "week": anchor.isocalendar().week,
        "days": [
            {
                "date": (monday + timedelta(days=offset)).isoformat(),
                "weekday": (monday + timedelta(days=offset)).isoweekday(),
                "items": by_day.get((monday + timedelta(days=offset)).isoformat(), []),
            }
            for offset in range(7)
        ],
        "issues": [
            {"kind": i.kind, "severity": i.severity, "message": i.message,
             "day": i.day.isoformat() if i.day else ""}
            for i in issues
        ],
        "total": len(occurrences),
    })


@app.post("/api/schedule/parse")
def api_schedule_parse():
    """把一段自然语言解析成日程候选。

    ⚠️ 这里**只解析、不落库**。解析结果必须经用户在界面上确认后才写进去 ——
       解析错了而用户没看见，比解析不出来更糟。
    """
    from scheduler.parser import parse_text

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "请先输入一句话，例如「下周四晚上七点在东区开班会」"}), 400

    items = parse_text(text)
    return jsonify({"items": [item.to_dict() for item in items]})


@app.post("/api/schedule/check")
def api_schedule_check():
    """落库前先裁决：这个时间到底行不行？行的话给冲突、不行的话给替代方案。"""
    from scheduler.conflict import Rules, check_candidate

    data = request.get_json(silent=True) or {}
    try:
        event = _event_from_payload(data)
    except Exception as exc:                                    # noqa: BLE001
        return jsonify({"error": f"这条日程不合法：{exc}"}), 400

    verdict = check_candidate(_schedule_store().all(), event, Rules())
    return jsonify({"verdict": verdict.to_dict()})


@app.post("/api/schedule/add")
def api_schedule_add():
    """确认落库。"""
    from scheduler.conflict import Rules, check_candidate

    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    if not items:
        return jsonify({"error": "没有要添加的日程"}), 400

    store = _schedule_store()
    added, rejected = [], []
    for payload in items:
        try:
            event = _event_from_payload(payload)
        except Exception as exc:                                # noqa: BLE001
            rejected.append({"title": payload.get("title", ""), "reason": str(exc)})
            continue
        verdict = check_candidate(store.all(), event, Rules())
        store.add(event)
        added.append({**_event_json(event), "ok": verdict.ok,
                      "issues": [i.message for i in verdict.issues]})

    return jsonify({"added": added, "rejected": rejected, "total": len(store)})


@app.post("/api/schedule/delete")
def api_schedule_delete():
    data = request.get_json(silent=True) or {}
    event_id = str(data.get("id") or "")
    if not event_id:
        return jsonify({"error": "缺少要删除的日程 id"}), 400
    if not _schedule_store().remove(event_id):
        return jsonify({"error": "没找到这条日程（可能已经被删了）"}), 404
    return jsonify({"ok": True})


@app.post("/api/schedule/edit")
def api_schedule_edit():
    """编辑日程。scope=single 只改这一次；默认 scope=series 改整个系列。

    重复日程在库里只有一条记录，没改过的那几次由 weekdays/until 展开；
    要单独改某一次，就把它从系列里摘出去（exdates），另存一条一次性事件。
    """
    from scheduler.conflict import Rules, check_candidate, occurrences_of

    data = request.get_json(silent=True) or {}
    event_id = str(data.get("id") or "")
    if not event_id:
        return jsonify({"error": "缺少要编辑的日程 id"}), 400

    store = _schedule_store()
    current = store.get(event_id)
    if not current:
        return jsonify({"error": "没找到这条日程（可能已经被删了）"}), 404

    if str(data.get("scope") or "series") == "single" and current.recurrence == "weekly":
        return _edit_one_occurrence(store, current, data)

    try:
        event = _event_from_payload({**data, "id": event_id})
    except Exception as exc:                                    # noqa: BLE001
        return jsonify({"error": f"这条日程不合法：{exc}"}), 400

    if current.recurrence == "weekly":
        occurrences = occurrences_of(current, event.start.date(), event.start.date())
        if any(occ.day == event.start.date() for occ in occurrences):
            event.start = current.start

    others = [item for item in store.all() if item.id != event_id]
    verdict = check_candidate(others, event, Rules())
    store.update(event)
    return jsonify({**_event_json(event), "ok": verdict.ok,
                    "issues": [i.message for i in verdict.issues]})


def _edit_one_occurrence(store, series, data):
    """只改系列里的这一次：系列加一条 exdate，另存一条一次性事件。"""
    from datetime import date as _date

    from scheduler.conflict import Rules, check_candidate, occurrences_of

    raw = str(data.get("occurrence") or "").strip()
    try:
        day = _date.fromisoformat(raw)
    except ValueError:
        return jsonify({"error": "缺少这次编辑对应的日期（occurrence）"}), 400
    if not any(occ.day == day for occ in occurrences_of(series, day, day)):
        return jsonify({"error": f"{day} 不在「{series.title}」这个系列里。"}), 400

    payload = {**data, "id": "", "weekdays": [], "until": "", "exdates": []}
    try:
        event = _event_from_payload(payload)
    except Exception as exc:                                    # noqa: BLE001
        return jsonify({"error": f"这条日程不合法：{exc}"}), 400
    event.id = f"{series.id}-{day.isoformat()}"

    series.exdates = sorted(set(series.exdates) | {day})
    try:
        series.validate()
    except Exception as exc:                                    # noqa: BLE001
        return jsonify({"error": f"这条日程不合法：{exc}"}), 400

    others = [item for item in store.all() if item.id != series.id]
    verdict = check_candidate(others + [series], event, Rules())
    store.update(series)
    store.add(event)
    return jsonify({**_event_json(event), "ok": verdict.ok,
                    "issues": [i.message for i in verdict.issues],
                    "series_id": series.id})


@app.get("/api/schedule/ics")
def api_schedule_ics():
    """导出 .ics 让手机日历导入。

    这一步是这个作品从"demo"变成"我真在用的工具"的分界线 ——
    导进手机之后，锁屏能看到、有人找你时能当场查。
    """
    from flask import Response

    from scheduler import ics as ics_mod

    store = _schedule_store()
    events = store.all()
    if not events:
        return jsonify({"error": "日程还是空的，先添加几条再导出"}), 400
    content = ics_mod.build_calendar(events, "我的日程")
    return Response(
        content,
        mimetype="text/calendar; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=schedule.ics"},
    )


def _event_from_payload(data: dict):
    """把前端传来的 JSON 变成内核的 Event。校验全部交给 Event.validate。"""
    from datetime import date as _date

    from scheduler.conflict import make_event

    start = str(data.get("start") or "").strip()
    if not start:
        raise ValueError("缺少开始时间")
    weekdays = data.get("weekdays") or []
    if isinstance(weekdays, str):
        weekdays = [int(x) for x in weekdays.replace("，", ",").split(",") if x.strip()]
    until = str(data.get("until") or "").strip() or None
    exdates = data.get("exdates") or []
    recurrence = "weekly" if weekdays else "one_off"
    if recurrence == "weekly" and not until:
        # 重复事件必须有个终点，否则会无限重复。没给就默认一学期（20 周）。
        until = (_date.fromisoformat(start[:10]) + timedelta(weeks=20)).isoformat()

    return make_event(
        title=str(data.get("title") or "").strip(),
        start=start,
        duration_min=int(data.get("duration_min") or 60),
        event_id=str(data.get("id") or ""),
        location=str(data.get("location") or "").strip(),
        category=str(data.get("category") or "other"),
        prep_min=int(data.get("prep_min") or 0),
        consequence=str(data.get("consequence") or "normal"),
        recurrence=recurrence,
        weekdays=weekdays,
        until=until,
        exdates=exdates,
        source=str(data.get("source") or "manual"),
        note=str(data.get("note") or ""),
    )


if __name__ == "__main__":
    print("=" * 56)
    print("  本地小助手  ——  http://127.0.0.1:5000")
    print("=" * 56)
    print("  提示：如果【对话】页面用不了，先在另一个窗口运行 ollama serve")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
