"""相册智能分类工具 —— 用 YOLO 的检测结果把照片分门别类

【思路】
    照片本身没有"标签"，但 YOLO 能告诉我们"画面里有哪些物体"。
    于是把"物体类别"映射成"人类翻相册时会用到的分类"
    （人像 / 宠物动物 / 美食餐具 / 交通工具……）。

【分类标准是怎么定的】
    先列出模型能识别的全部类别（本模型为 COCO 的 80 类），
    再按"翻相册的习惯"分组 —— 而不是凭想象定分类，那样会出现模型根本认不出的分类。

【一张图检测到多种物体时，归到哪一类？】
    两种可选策略（用 --strategy 切换）：
      priority（默认）：按优先级，CATEGORIES 的书写顺序即优先级（人像最优先）
                        理由：人像照最常被翻找
      count           ：哪个分类检测到的物体最多，就归到哪一类
                        理由：更贴近"这张图主要拍了什么"
    建议先用 --dry-run 对比两种策略的实际分布，再决定。

【局限性（诚实说明）】
    本工具按"画面里的物体"分类，**不做场景理解**。
    因此纯风景照、截图、文档照片等没有可识别物体的图片会落入「未识别」。
    要真正做场景分类，需要额外引入场景分类模型（如 Places365）或图像描述模型。

【用法】
    # 对比两种分类策略（不复制任何文件）
    python src/organizer.py -i data/album --dry-run

    # 正式分类（复制到输出目录，不改动原文件）
    python src/organizer.py -i data/album -o output/album_sorted --strategy priority

    # 追求速度时（个人相册上会有更多"未识别"）
    python src/organizer.py -i data/album -o out --imgsz 320 --conf 0.35
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

# 相册分类 -> 对应的模型类别。字典顺序 = 优先级（越靠前越优先）
CATEGORIES: dict[str, list[str]] = {
    "人像": ["person"],
    "宠物动物": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "美食餐具": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                 "donut", "cake", "bowl", "cup", "wine glass", "fork", "knife", "spoon", "bottle"],
    "交通工具": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "电子设备": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                 "toaster", "refrigerator", "clock"],
    "家具室内": ["chair", "couch", "bed", "dining table", "toilet", "sink", "potted plant", "vase", "book"],
    "户外街道": ["traffic light", "fire hydrant", "stop sign", "bench", "parking meter"],
    "运动娱乐": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                 "skateboard", "surfboard", "tennis racket"],
    "随身物品": ["backpack", "umbrella", "handbag", "tie", "suitcase", "scissors"],
    "玩具其他": ["teddy bear", "hair drier", "toothbrush"],
}

CATEGORY_ORDER = list(CATEGORIES)
CLASS_TO_CATEGORY = {cls: cat for cat, classes in CATEGORIES.items() for cls in classes}
UNKNOWN_CATEGORY = "未识别"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_category(label_counter: Counter, strategy: str) -> str:
    """决定一张图片归到哪个相册分类。

    label_counter: 该图检测到的 类别 -> 数量
    strategy:      priority（按优先级） 或 count（按数量最多）
    """
    # 把"检测到的物体"展开成"所属分类"，重复的按数量重复（供 count 策略使用）
    cats: list[str] = []
    for label, n in label_counter.items():
        cat = CLASS_TO_CATEGORY.get(label)
        if cat:
            cats.extend([cat] * n)

    if not cats:
        return UNKNOWN_CATEGORY

    if strategy == "count":
        counts = Counter(cats)
        top = max(counts.values())
        candidates = [c for c, n in counts.items() if n == top]
        return min(candidates, key=CATEGORY_ORDER.index)  # 数量相同时按优先级

    return min(set(cats), key=CATEGORY_ORDER.index)  # 优先级策略


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def detect_all(images: list[Path], weights: str, conf: float, imgsz: int, batch: int) -> list[dict]:
    """分批跑检测，只记录"检测到了哪些物体"（分类不在这里决定）。

    关于 imgsz 的取舍（同一批 52 张个人相册照片的实测对照）：
        imgsz=320, conf=0.35  →  未识别率 40.4%
        imgsz=640, conf=0.25  →  未识别率 26.9%
    小尺寸在"物体大而清晰"的数据集（如 COCO）上够用且快几倍，
    但个人相册里的物体往往更小更杂，所以默认取 640 优先保证效果。
    """
    model = YOLO(weights)
    records: list[dict] = []

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, verbose=False)
        for path, result in zip(chunk, results):
            labels = [model.names[int(box.cls)] for box in result.boxes]
            records.append({"path": path, "labels": Counter(labels)})
        print(f"  已处理 {min(start + batch, len(images))}/{len(images)} 张", flush=True)

    return records


def assign(records: list[dict], strategy: str) -> Counter:
    counter: Counter = Counter()
    for r in records:
        r["category"] = pick_category(r["labels"], strategy)
        counter[r["category"]] += 1
    return counter


def print_summary(counter: Counter, total: int, title: str) -> None:
    print()
    print(title)
    print(f"{'分类':<10}{'数量':>6}{'占比':>10}")
    print("-" * 28)
    for name in CATEGORY_ORDER + [UNKNOWN_CATEGORY]:
        n = counter.get(name, 0)
        if n:
            print(f"{name:<10}{n:>6}{n / total * 100:>9.1f}%")


def write_report(records: list[dict], out_dir: Path, args: argparse.Namespace) -> Path:
    report = out_dir / "report.md"
    counter = Counter(r["category"] for r in records)

    lines = [
        "# 相册分类报告",
        "",
        f"- 输入目录：`{args.input}`",
        f"- 图片总数：{len(records)}",
        f"- 使用权重：`{args.weights}`",
        f"- 检测参数：`imgsz={args.imgsz}, conf={args.conf}`",
        f"- 归类策略：`{args.strategy}`"
        "（priority = 按优先级；count = 按检测到的物体数量最多）",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 | 占比 |",
        "|---|---|---|",
    ]
    for name in CATEGORY_ORDER + [UNKNOWN_CATEGORY]:
        n = counter.get(name, 0)
        if n:
            lines.append(f"| {name} | {n} | {n / len(records) * 100:.1f}% |")

    lines += ["", "## 明细", ""]
    for name in CATEGORY_ORDER + [UNKNOWN_CATEGORY]:
        group = [r for r in records if r["category"] == name]
        if not group:
            continue
        lines.append(f"### {name}（{len(group)} 张）")
        lines.append("")
        for r in group:
            detail = "、".join(f"{k}×{v}" for k, v in r["labels"].most_common()) or "未检出物体"
            lines.append(f"- `{r['path'].name}` — {detail}")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="相册智能分类工具（基于 YOLO 检测结果）")
    parser.add_argument("-i", "--input", required=True, help="输入目录（放照片的文件夹）")
    parser.add_argument("-o", "--output", default="output/album_sorted", help="输出目录")
    parser.add_argument("--weights", default="weights/best.pt", help="YOLO 权重文件")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理尺寸，默认 640。调到 320 会快几倍，但个人相册上漏检更多（见 README 实测对照）")
    parser.add_argument("--batch", type=int, default=8, help="批处理大小")
    parser.add_argument("--strategy", choices=["priority", "count"], default="priority",
                        help="多物体时的归类策略：priority 按优先级 / count 按数量最多")
    parser.add_argument("--dry-run", action="store_true", help="只预览分类结果，不复制文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(f"目录里没有找到图片：{input_dir}")

    print(f"找到 {len(images)} 张图片，开始检测（imgsz={args.imgsz}, conf={args.conf}）...")
    records = detect_all(images, args.weights, args.conf, args.imgsz, args.batch)

    # 先用两种策略各算一遍，便于对比（--dry-run 时打印两张表）
    counter_priority = assign(records, "priority")
    counter_count = assign(records, "count")

    if args.dry_run:
        print_summary(counter_priority, len(records), "【策略 A：priority — 按优先级，人像最优先】")
        print_summary(counter_count, len(records), "【策略 B：count — 按检测到的物体数量最多】")
        print()
        print("（--dry-run 模式：没有复制任何文件）")
        print("确定策略后，去掉 --dry-run 并加上 --strategy priority（或 count）即可真正分类")
        return

    # 按选定策略重新归类
    assign(records, args.strategy)
    print_summary(Counter(r["category"] for r in records), len(records),
                  f"【实际使用策略：{args.strategy}】")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        target_dir = out_dir / r["category"]
        target_dir.mkdir(exist_ok=True)
        shutil.copy2(r["path"], target_dir / r["path"].name)

    report = write_report(records, out_dir, args)
    print()
    print(f"分类完成：{out_dir}")
    print(f"报告文件：{report}")


if __name__ == "__main__":
    main()
