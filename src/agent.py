"""最小 Agent Harness —— 让 1.5B 的小模型也能调用真实工具

【设计原则（参照 Pi 的分层）】
    1. 工具少而自解释 —— 只有 3 个，模型不用纠结选哪个
    2. 把难度放在工具里 —— 复杂活由工具内部干完，模型只负责"选对工具、填对参数"
    3. 分层清晰：
         模型接口层  → ask_llm()           （对应 pi-ai）
         agent 核心  → run_agent()         （对应 pi-agent-core）
         工具层      → TOOLS + execute()   （对应 pi-coding-agent，复用项目已有的检测/分类能力）

【针对小模型的 4 个弱点，harness 做这 4 件事】
    这是先用探针实验验证出来的问题，再针对性补的：

    探针发现的问题                      →  harness 的对策
    ─────────────────────────────────────────────────────
    ① 把工具调用写成纯文本/带废话        →  容错解析：扫描平衡括号抠出 JSON
    ② 参数写错（如 /data/samples）       →  参数校验：缺参数/路径不存在就把错误回传，让它重试
    ③ 走一两步就停住                     →  步数控制 + 任务未完成时推进一次
    ④ 无法把工具结果传给下一步           →  结果显式注入：工具返回值直接拼进下一轮提示词

【单独运行（不开网页也能测）】
    python src/agent.py "帮我整理 data/my_photos"
    python src/agent.py "看看 data/samples/bus.jpg 里有什么"
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classify_album import CLS_NAMES, OTHER_DIR          # noqa: E402
from detector import detect, format_for_llm              # noqa: E402
from ultralytics import YOLO                             # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
CLS_WEIGHTS = str(ROOT / "weights" / "cls_best.pt")
DET_WEIGHTS = str(ROOT / "weights" / "best.pt")
DEFAULT_MIN_CONF = 0.6
MAX_STEPS = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 项目根目录：模型给的相对路径都相对它解析，避免"工作目录不同就找不到文件"
PROJECT_ROOT = ROOT


# ═══════════════════════════ 工具定义 ═══════════════════════════
# 这是给模型看的"说明书"，不是给它执行的代码。
# 描述必须短、直白、带例子 —— 1.5B 的模型读不懂绕的话。

TOOLS: dict[str, dict] = {
    "organize_album": {
        "desc": "把一整个文件夹里的照片按内容自动分好类：分类、复制到对应子文件夹、生成报告。"
                "用户说「整理照片」「把相册分类」时用它。",
        "required": ["source", "target"],
        "args": {
            "source": "要整理的文件夹路径，例如 data/my_photos",
            "target": "整理结果放哪里，例如 output/agent_sorted",
        },
    },
    "detect_objects": {
        "desc": "用目标检测模型看一张图片里有哪些物体。用户问「这张图里有什么」时用它。",
        "required": ["image"],
        "args": {"image": "图片路径，例如 data/samples/bus.jpg"},
    },
    "list_images": {
        "desc": "列出一个文件夹里有多少张图片。用户想先了解目录内容时用它。",
        "required": ["directory"],
        "args": {"directory": "文件夹路径，例如 data/my_photos"},
    },
}


# ═══════════════════════════ 模型接口层 ═══════════════════════════
def ask_llm(prompt: str, temperature: float = 0.2) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def build_prompt(question: str, history: list[dict]) -> str:
    tool_text = []
    for i, (name, spec) in enumerate(TOOLS.items(), 1):
        lines = [f"{i}. {name}", f"   作用：{spec['desc']}", "   参数："]
        for arg, desc in spec["args"].items():
            mark = "（必需）" if arg in spec["required"] else ""
            lines.append(f"     - {arg}{mark}：{desc}")
        tool_text.append("\n".join(lines))

    history_text = ""
    if history:
        parts = []
        for item in history:
            if item["role"] == "assistant":
                parts.append(f"你上一轮说：{item['content']}")
            else:
                parts.append(f"工具返回的结果：\n{item['content']}")
        history_text = "\n【之前的进展】\n" + "\n".join(parts) + "\n"

    return f"""你是一个会用工具的助手，请用中文回答。

【可用工具】
{chr(10).join(tool_text)}

【输出规则 —— 必须严格遵守】
- 要调用工具时，**只输出一行 JSON**，不要写任何解释、不要用代码块包裹：
  {{"tool": "工具名", "args": {{"参数名": "参数值"}}}}
- 一次只调用一个工具。
- 路径分隔符统一用 / ，不要用 \\ 。
- 任务全部完成后，用中文写一句话总结，**不要输出 JSON**。

【例子】
用户：帮我整理一下 data/my_photos 这个文件夹
你的输出：{{"tool": "organize_album", "args": {{"source": "data/my_photos", "target": "output/agent_sorted"}}}}
{history_text}
【用户的要求】
{question}

【你的输出】"""


# ═══════════════════════════ 容错解析 ═══════════════════════════
def _balanced_json_fragments(text: str) -> list[str]:
    """扫描文本，取出所有括号配平的 {...} 片段（能正确处理嵌套的大括号）。"""
    fragments, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                fragments.append(text[start:i + 1])
    return fragments


def parse_tool_call(raw: str) -> tuple[str, dict] | None:
    """从模型输出里抠出工具调用。

    小模型经常不守规矩：加代码块、前面写一堆废话、参数字符串写错。
    所以这里逐层容错，而不是直接 json.loads 整个输出。
    """
    cleaned = raw.replace("```json", "").replace("```", "")
    for fragment in _balanced_json_fragments(cleaned):
        try:
            obj = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "tool" not in obj:
            continue

        name = str(obj.get("tool", "")).strip()
        args = obj.get("args")
        if not isinstance(args, dict):
            # 模型可能把参数直接写在顶层（没有 args 包一层）
            args = {k: v for k, v in obj.items() if k != "tool"}
        return name, args
    return None


def looks_like_task(question: str) -> bool:
    """粗判用户是不是在提"要干活"的需求（用来决定要不要推进模型）。"""
    keywords = ["整理", "分类", "归档", "看看", "检测", "有什么", "多少张", "列出", "分一下"]
    return any(k in question for k in keywords)


def build_summary_prompt(question: str, history: list[dict]) -> str:
    """收尾专用的提示词。

    关键点：**这个提示词里没有任何工具说明**。
    小模型只要还能看到工具，就会忍不住继续调用 —— 所以收尾必须把工具"藏起来"，
    让它没有别的选择，只能好好写总结。
    """
    results = "\n".join(f"- {item['content']}" for item in history if item["role"] == "tool")
    return f"""用户的要求：{question}

你已经调用过工具，下面是工具返回的结果：
{results}

请用两三句自然流畅的中文，把结果告诉用户，就像助手在汇报工作。
要求：
- 不要输出 JSON
- 不要出现"工具""调用""参数"这些词
- 不要编造上面结果里没有的信息

你的回答："""


# ═══════════════════════════ 工具层（真正干活）═══════════════════════════
_cls_model = None


def _get_cls_model() -> YOLO:
    global _cls_model
    if _cls_model is None:
        _cls_model = YOLO(CLS_WEIGHTS)
    return _cls_model


def _resolve(path_str: str) -> Path:
    """把模型给的路径解析成绝对路径。"""
    path = Path(str(path_str).strip().strip('"').strip("'"))
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def tool_list_images(directory: str) -> tuple[bool, str, dict]:
    folder = _resolve(directory)
    if not folder.is_dir():
        return False, f"错误：目录不存在 —— {directory}。请你确认路径后重新调用。", {}
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        return True, f"目录 {directory} 里没有找到图片。", {"count": 0}
    by_ext = Counter(p.suffix.lower() for p in images)
    detail = "、".join(f"{ext} {n} 张" for ext, n in by_ext.most_common())
    return True, f"目录 {directory} 里共有 {len(images)} 张图片（{detail}）。", {
        "count": len(images),
        "samples": [p.name for p in images[:5]],
    }


def tool_detect_objects(image: str) -> tuple[bool, str, dict]:
    path = _resolve(image)
    if not path.is_file():
        return False, f"错误：图片不存在 —— {image}。请你确认路径后重新调用。", {}
    boxes = detect(str(path), weights=DET_WEIGHTS, conf=0.35)
    if not boxes:
        return True, f"这张图（{path.name}）里没有检测到明显的物体。", {"boxes": []}
    return True, f"在 {path.name} 里检测到：{format_for_llm(boxes)}。", {"boxes": boxes}


def tool_organize_album(source: str, target: str) -> tuple[bool, str, dict]:
    src = _resolve(source)
    dst = _resolve(target)
    if not src.is_dir():
        return False, f"错误：源目录不存在 —— {source}。请你确认路径后重新调用。", {}

    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        return False, f"错误：{source} 里没有图片，没法整理。", {}

    model = _get_cls_model()
    model.predict([str(images[0])], imgsz=224, verbose=False)   # 预热，避免首个计时不准
    start = time.time()

    counts: Counter = Counter()
    for image in images:
        pred = model.predict(str(image), imgsz=224, verbose=False)[0]
        label = model.names[int(pred.probs.top1)]
        conf = float(pred.probs.top1conf)
        category = OTHER_DIR if conf < DEFAULT_MIN_CONF else CLS_NAMES.get(label, label)
        folder = dst / category
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, folder / image.name)
        counts[category] += 1

    elapsed = time.time() - start
    summary_parts = "、".join(f"{cat} {n} 张" for cat, n in counts.most_common())
    report = dst / "report.md"
    report.write_text(
        "# 相册整理报告（由 Agent 自动生成）\n\n"
        f"- 源目录：`{source}`\n- 输出目录：`{target}`\n- 图片总数：{len(images)}\n"
        f"- 置信度阈值：{DEFAULT_MIN_CONF}（低于此值归入「{OTHER_DIR}」）\n"
        f"- 耗时：{elapsed:.1f} 秒\n\n## 分类统计\n\n| 分类 | 数量 |\n|---|---|\n"
        + "".join(f"| {cat} | {n} |\n" for cat, n in counts.most_common()),
        encoding="utf-8",
    )

    return True, (
        f"整理完成：共 {len(images)} 张，耗时 {elapsed:.1f} 秒。"
        f"分类结果 —— {summary_parts}。报告已写入 {target}/report.md。"
    ), {"total": len(images), "counts": dict(counts), "elapsed": round(elapsed, 1)}


def execute_tool(name: str, args: dict) -> tuple[bool, str, dict]:
    """执行工具。任何失败都返回可读的错误信息，让模型能看懂并重试。"""
    if name not in TOOLS:
        return False, (f"错误：没有叫 {name} 的工具。"
                       f"可用的工具只有：{'、'.join(TOOLS)}。请重新调用。"), {}

    missing = [a for a in TOOLS[name]["required"] if not str(args.get(a, "")).strip()]
    if missing:
        return False, (f"错误：调用 {name} 缺少必需参数 {'、'.join(missing)}。"
                       f"需要这些参数：{'、'.join(TOOLS[name]['args'])}。请重新调用。"), {}

    try:
        if name == "organize_album":
            return tool_organize_album(args["source"], args["target"])
        if name == "detect_objects":
            return tool_detect_objects(args["image"])
        return tool_list_images(args["directory"])
    except requests.exceptions.ConnectionError:
        return False, "错误：连不上本地大模型服务，请先运行 ollama serve。", {}
    except Exception as exc:                                    # noqa: BLE001
        return False, f"工具执行出错：{exc}", {}


# ═══════════════════════════ agent 核心循环 ═══════════════════════════
def run_agent(question: str, max_steps: int = MAX_STEPS):
    """跑一次 agent，逐步 yield 事件（供网页实时显示）。

    事件类型：start / thinking / tool_call / tool_result / nudge / repeat / summarizing / answer / error

    【为什么需要"重复调用检测"和"强制收尾"】
        实测发现 1.5B 模型能选对工具、填对参数，但**不会判断任务已经完成** ——
        它会用同样的参数反复调用同一个工具，直到步数耗尽，永远不给最终回答。
        对策：把"何时结束"这件事从模型手里拿走，交给 harness 判断 ——
        一旦发现重复调用，就强制进入"收尾阶段"（用不含工具说明的提示词逼它写总结）。
    """
    yield {"type": "start", "question": question, "tools": list(TOOLS), "max_steps": max_steps}

    history: list[dict] = []
    nudged = False
    last_signature = None
    finished_reason = ""

    for step in range(1, max_steps + 1):
        yield {"type": "thinking", "step": step}
        try:
            raw = ask_llm(build_prompt(question, history))
        except requests.exceptions.ConnectionError:
            yield {"type": "error", "text": "连不上本地 Ollama，请先运行 ollama serve"}
            return

        call = parse_tool_call(raw)

        # ── 对策③：模型没调工具、但显然还在"要干活"的任务里 → 推它一把（只推一次）
        if call is None and not nudged and looks_like_task(question) and step < max_steps:
            nudged = True
            history.append({"role": "assistant", "content": raw.strip()[:200]})
            history.append({
                "role": "tool",
                "content": "你还没有调用任何工具。请立刻输出一行 JSON 调用合适的工具，不要解释。",
            })
            yield {"type": "nudge", "text": "模型没调工具，已提醒它一次"}
            continue

        # ── 没有工具调用 → 模型自己决定收尾，直接用它的回答
        if call is None:
            yield {"type": "answer", "text": raw.strip() or "（模型没有输出内容）"}
            return

        name, args = call
        signature = f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

        # ── 对策⑤：重复调用同一个工具 → 说明它已经拿到结果但不会收尾，强制进入收尾
        if signature == last_signature:
            yield {"type": "repeat",
                   "text": f"模型重复调用 {name}（参数相同），判定任务已完成，转入收尾"}
            finished_reason = "repeat"
            break

        last_signature = signature
        yield {"type": "tool_call", "step": step, "name": name, "args": args,
               "raw": raw.strip()[:300]}

        ok, summary, detail = execute_tool(name, args)
        yield {"type": "tool_result", "step": step, "name": name,
               "ok": ok, "summary": summary, "detail": detail}

        # ── 对策④：结果显式注入下一轮提示词
        history.append({"role": "assistant", "content": raw.strip()[:300]})
        history.append({"role": "tool", "content": summary})

    # ── 收尾阶段：用"没有工具说明"的提示词，逼模型写总结
    has_result = any(item["role"] == "tool" for item in history)
    if not has_result:
        yield {"type": "answer",
               "text": f"（已达到最大步数 {max_steps} 步，没能完成任务。你可以换个说法再试一次。）"}
        return

    if not finished_reason:
        yield {"type": "repeat", "text": f"已达到最大步数 {max_steps}，转入收尾"}
    yield {"type": "summarizing"}
    try:
        answer = ask_llm(build_summary_prompt(question, history))
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "text": "连不上本地 Ollama，请先运行 ollama serve"}
        return
    yield {"type": "answer", "text": answer.strip() or "（收尾时模型没有输出内容）"}


# ═══════════════════════════ 命令行入口 ═══════════════════════════
def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    print(f"\n🙋 用户：{question}\n" + "─" * 60)
    for event in run_agent(question):
        kind = event["type"]
        if kind == "thinking":
            print(f"🤔 第 {event['step']} 步思考中…")
        elif kind == "tool_call":
            print(f"🔧 调用工具 {event['name']}  参数={event['args']}")
        elif kind == "tool_result":
            mark = "✅" if event["ok"] else "❌"
            print(f"{mark} {event['summary']}")
        elif kind == "nudge":
            print(f"👉 {event['text']}")
        elif kind == "repeat":
            print(f"🔁 {event['text']}")
        elif kind == "summarizing":
            print("📝 收尾中（已隐藏工具列表，逼模型写总结）…")
        elif kind == "answer":
            print(f"\n🤖 {event['text']}\n")
        elif kind == "error":
            print(f"⚠️  {event['text']}")


if __name__ == "__main__":
    main()
