"""YOLO 目标检测模块 —— 负责"看"

职责：给一张图片，返回"画面里有什么、在哪、有多确定"。
它不负责描述图片、也不负责聊天，那是大模型（src/ai.py）的事。

用法（单独测试）:
    python src/detector.py                          # 用默认示例图
    python src/detector.py data/samples/zidane.jpg  # 指定图片

用法（在代码里调用）:
    from detector import detect, format_for_llm
    for d in detect("data/samples/bus.jpg"):
        print(d["label"], d["confidence"], d["box"])

关于权重:
    - 默认使用官方预训练权重 yolov8n.pt（能识别 COCO 的 80 类常见物体：
      人、车、猫、狗、杯子、手机、椅子……），第一次运行会自动下载（约 6 MB）
    - 在魔搭上训练出自己的权重后，把它放进 weights/ 目录，
      然后调用时传参即可：detect("图片.jpg", weights="weights/best.pt")
"""
import sys
import time
from pathlib import Path

from ultralytics import YOLO

# 项目根目录（src/ 的上一层）。用绝对路径拼权重，
# 这样不管你在哪个目录执行命令，都能找到模型文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_WEIGHTS = str(WEIGHTS_DIR / "yolov8n.pt")

# 全局缓存：模型只加载一次。反复调用时不必重复读盘（每次省几秒）。
_model_cache: dict[str, YOLO] = {}


def _get_model(weights: str) -> YOLO:
    if weights not in _model_cache:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def detect(image_path: str, weights: str = DEFAULT_WEIGHTS, conf: float = 0.35) -> list[dict]:
    """对一张图片做目标检测。

    参数:
        image_path: 图片路径
        weights:    权重文件路径（默认官方 yolov8n.pt）
        conf:       置信度阈值，低于该值的结果会被丢弃（默认 0.35，过滤掉噪声）

    返回:
        [{"label": "person", "confidence": 0.94, "box": [x1, y1, x2, y2]}, ...]
    """
    model = _get_model(weights)
    results = model.predict(image_path, conf=conf, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "box": [round(float(v), 1) for v in box.xyxy[0]],
            })
    return detections


def format_for_llm(detections: list[dict]) -> str:
    """把检测结果汇总成一句简短清单，塞进给大模型的 prompt 里。

    刻意只给"类别 + 数量"，不给置信度，原因有两个：
      1. 信息更紧凑，能力弱的模型不容易被一串小数带跑偏（实测有效）
      2. "这是什么场景"的理解应该交给大模型，YOLO 只负责"看到什么"
    """
    if not detections:
        return "检测结果：没有识别出任何物体"

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    items = "、".join(f"{label} × {n}" for label, n in counts.items())
    return f"检测到的物体：{items}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(SAMPLES_DIR / "bus.jpg")

    print(f"正在检测: {path}")
    start = time.time()
    detections = detect(path)
    cost = time.time() - start

    print(f"\n推理耗时: {cost:.2f} 秒")
    print(f"检测到 {len(detections)} 个目标：")
    for d in detections:
        print(f"  {d['label']:<12} 置信度 {d['confidence']:<6} 位置 {d['box']}")
