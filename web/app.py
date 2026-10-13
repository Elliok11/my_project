"""相册 AI 工具箱 —— Web 界面后端（Flask）

【三个页面 = 项目里的三种能力】
    ① 对话       → 本地大模型（Ollama + qwen2.5:1.5b），流式输出
    ② 看图说话   → YOLO 检测框 + 大模型用自然语言描述
    ③ 相册分类   → 自己训练的分类模型（5 类 + 置信度兜底到「其他」）

【启动】
    python web/app.py
    然后浏览器打开 http://127.0.0.1:5000

【设计说明】
    - 全部复用 src/ 下已有的模块（detector.py 等），不重复造轮子
    - Vue 用本地文件（static/vue.global.prod.js），不依赖 CDN，演示更稳
    - 上传的图片用完即删，不留垃圾
"""
import base64
import io
import json
import re
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.exceptions import HTTPException

# ── 复用项目里已有的模块 ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import detect, format_for_llm  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ── 配置 ──
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
DET_WEIGHTS = str(ROOT / "weights" / "best.pt")        # 自己训练的检测模型
CLS_WEIGHTS = str(ROOT / "weights" / "cls_best.pt")    # 自己训练的分类模型
CLS_NAMES = {
    "screenshot": "截图",
    "document": "文档",
    "scenery": "风景",
    "meme": "表情包",
    "food_cn": "美食",
}
OTHER = "其他"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ORGANIZE_IMAGES = 300   # 超过这个数量的文件夹不建议"整体整理"（会复制大量文件、耗时很久）

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RUNS_DIR = UPLOAD_DIR / "runs"          # 每次上传的图片暂存处（按 run_id 分目录）
OUTPUT_DIR = ROOT / "output"            # 「保存结果」时落盘的位置

# 每次启动清空上次的暂存（避免磁盘越积越多）；
# 如果上次有没保存的结果，会在这里被清掉——界面上会有提示。
shutil.rmtree(RUNS_DIR, ignore_errors=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024     # 单次请求上限 256MB

_cls_model = None


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


def get_cls_model() -> YOLO:
    """分类模型只加载一次（加载一次几秒，每次请求都加载会慢得没法用）。"""
    global _cls_model
    if _cls_model is None:
        _cls_model = YOLO(CLS_WEIGHTS)
    return _cls_model


# ────────────────────── 工具函数 ──────────────────────
def save_upload(file_storage) -> Path:
    suffix = Path(file_storage.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".jpg"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    file_storage.save(path)
    return path


def to_data_url(path: Path, max_size: int = 0) -> str:
    """图片转 data URL。max_size > 0 时生成缩略图（减小传输体积）。"""
    if max_size > 0:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
        data = buf.getvalue()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


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
        "det_weights": Path(DET_WEIGHTS).exists(),
        "cls_weights": Path(CLS_WEIGHTS).exists(),
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


# ────────────────────── ② 看图说话 ──────────────────────
@app.post("/api/vision")
def api_vision():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "没有收到图片"}), 400
    question = (request.form.get("question") or "这张图里有什么？").strip()

    path = save_upload(file)
    try:
        boxes = detect(str(path), weights=DET_WEIGHTS, conf=0.35)
        w, h = image_size(path)
        image_url = to_data_url(path, max_size=900)

        detail = format_for_llm(boxes)
        prompt = (
            "下面是一张图片经过目标检测模型得到的结果（只有物体类别和数量，没有任何场景信息）：\n"
            f"{detail}\n\n"
            "请用两三句自然流畅的中文描述这张图片里可能是什么场景。"
            "要求：直接给出描述，不要复述上面的清单，不要出现任何数字或置信度，"
            "不要重复列举同一种物体。\n"
            f"最后回答用户的问题：{question}"
        )
        description = ask_ollama(prompt, stream=False)
        return jsonify({
            "image": image_url,
            "width": w,
            "height": h,
            "boxes": boxes,
            "description": description,
        })
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "连不上本地 Ollama，请先运行 ollama serve"}), 503
    finally:
        path.unlink(missing_ok=True)


# ────────────────────── ③ 相册分类 ──────────────────────
@app.post("/api/classify")
def api_classify():
    """对上传的图片逐张跑分类模型。

    注意两点：
      1. 这里**不**做置信度过滤，只返回"模型最像哪一类 + 置信度"。
         阈值判断放在前端做，这样拖动阈值滑块时能立刻重算，不用重新上传。
      2. 上传的图片会**暂存**在 web/uploads/runs/<run_id>/ 下，
         等用户点「保存结果」时再按最终分类复制到 output/ 目录。
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "没有收到图片"}), 400

    run_id = (request.form.get("run_id") or "").strip()
    # 只接受字母/数字/下划线/短横线：run_id 会被拼进文件路径，必须防目录穿越
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", run_id):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = get_cls_model()
    results = []

    for file in files:
        path = run_dir / f"{uuid.uuid4().hex}{Path(file.filename or 'x.jpg').suffix.lower() or '.jpg'}"
        file.save(path)
        pred = model.predict(str(path), imgsz=224, verbose=False)[0]
        label = model.names[int(pred.probs.top1)]
        results.append({
            "name": file.filename,
            "stored_name": path.name,                       # 供「保存结果」时定位文件
            "thumb": to_data_url(path, max_size=180),
            "raw_category": CLS_NAMES.get(label, label),
            "conf": round(float(pred.probs.top1conf), 3),
        })

    return jsonify({"run_id": run_id, "total": len(results), "results": results})


@app.post("/api/save")
def api_save():
    """把本次上传的图片按【用户当前阈值下的分类】复制到 output/ 目录。

    前端会把「文件名 → 分类」的对应关系发过来，
    所以保存下来的结果和用户在界面上看到的一致。
    """
    data = request.get_json(silent=True) or {}
    run_id = str(data.get("run_id") or "")
    items = data.get("items") or []
    if not run_id or not items:
        return jsonify({"error": "没有要保存的内容"}), 400

    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        return jsonify({"error": "暂存的图片已不存在（可能服务重启过），请重新上传"}), 410

    out_dir = OUTPUT_DIR / f"web_{run_id}"
    saved: dict[str, int] = {}

    for item in items:
        src = run_dir / str(item.get("stored_name", ""))
        category = str(item.get("category") or "其他")
        if not src.exists():
            continue
        dst_dir = out_dir / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        saved[category] = saved.get(category, 0) + 1

    # 保存完就清理暂存，避免磁盘堆积
    shutil.rmtree(run_dir, ignore_errors=True)

    return jsonify({
        "path": str(out_dir),
        "saved": [{"category": k, "count": v} for k, v in saved.items()],
        "total": sum(saved.values()),
    })


@app.get("/api/agent/tools")
def api_agent_tools():
    """给前端展示用的工具清单。"""
    from agent import TOOLS
    return jsonify([{"name": name, **spec} for name, spec in TOOLS.items()])


@app.get("/api/agent/suggestions")
def api_agent_suggestions():
    """扫描项目里的 data/ 目录，生成「你现在可以让我做什么」的建议。

    关键设计：**完全靠文件扫描生成，一次都不调用大模型**。因此：
      - 列出来的目标都是**真实存在**的（模型会编造不存在的路径，扫描不会）
      - 毫秒级响应，不占 CPU
      - 用户往 data/ 里放了新文件夹，刷新页面就自动出现新建议
    """
    data_dir = ROOT / "data"
    if not data_dir.is_dir():
        return jsonify([])

    folders: list[tuple[Path, list[Path]]] = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            folders.append((folder, images))

    items: list[dict] = []

    # ① 优先推荐"适合整理"的文件夹（5~300 张），张数多的排前面
    album_like = [(f, i) for f, i in folders if 5 <= len(i) <= MAX_ORGANIZE_IMAGES]
    album_like.sort(key=lambda pair: -len(pair[1]))
    for folder, images in album_like[:3]:
        rel = folder.relative_to(ROOT).as_posix()
        items.append({
            "icon": "🗂️",
            "title": f"整理 {rel}",
            "detail": f"{len(images)} 张照片 · 自动分类归档并生成报告",
            "question": f"帮我整理一下 {rel} 这个文件夹",
        })

    # ② 张数很少的文件夹：建议看看其中的单张图
    for folder, images in folders:
        if len(images) <= 5:
            for image in images[:2]:
                rel = image.relative_to(ROOT).as_posix()
                items.append({
                    "icon": "🔍",
                    "title": f"看看 {image.name} 里有什么",
                    "detail": "用检测模型识别画面中的物体",
                    "question": f"看看 {rel} 里有什么",
                })

    # ③ 图片特别多的文件夹：不建议整体整理，先数一数
    for folder, images in folders:
        if len(images) > MAX_ORGANIZE_IMAGES:
            rel = folder.relative_to(ROOT).as_posix()
            items.append({
                "icon": "🔢",
                "title": f"数一数 {rel}",
                "detail": f"{len(images)} 张 · 数量较多，不建议整体整理",
                "question": f"{rel} 里有多少张照片",
            })

    return jsonify(items[:6])


@app.post("/api/agent")
def api_agent():
    """Agent 接口：流式返回每一步（思考 / 工具调用 / 工具结果 / 最终回答）。

    用 NDJSON（每行一个 JSON 对象）而不是 SSE —— 前端解析更简单，也不必用 EventSource。
    这样界面上能**实时看到**模型调用了什么工具、拿到了什么结果。
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "请输入你想让 Agent 做的事"}), 400

    from agent import run_agent

    def generate():
        try:
            for event in run_agent(question):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            yield json.dumps({"type": "error", "text": f"Agent 出错：{exc}"},
                             ensure_ascii=False) + "\n"

    return app.response_class(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    print("=" * 56)
    print("  相册 AI 工具箱  ——  http://127.0.0.1:5000")
    print("=" * 56)
    print("  提示：如果【对话】页面用不了，先在另一个窗口运行 ollama serve")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
