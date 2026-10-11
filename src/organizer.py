"""相册智能分类工具 —— 两级流水线：YOLO 物体检测 + 图像分类兜底

【思路】
    照片本身没有"标签"，但模型能告诉我们两件事：
      1. YOLO 检测：画面里有哪些**物体**（人、车、猫、桌子……）→ 适合物体密集的照片
      2. 图像分类：整张图属于**哪一类**（截图/文档/风景/表情包/中餐）→ 适合"没有明确物体"的图

    所以做成两级流水线：
        照片 → YOLO 检测
                 ├── 检测到物体 → 按物体归类（人像 / 宠物动物 / 美食餐具 ……）
                 └── 没检测到   → 交给分类模型判断整图类型（截图 / 文档 / 风景 / 表情包）

【为什么要做成两级】
    只用检测时，截图、文档、风景这类"没有物体"的照片会全部落入「未识别」——
    在 52 张个人相册上实测高达 40.4%（调参后 26.9%）。
    加分类兜底后，这部分照片能被正确归类。这是本项目最核心的设计决策。

【分类标准是怎么定的】
    检测侧：先列出模型能识别的 80 类物体，再按"翻相册的习惯"分组（见 CATEGORIES）
    分类侧：自己采集 1027 张图片训练的分类模型，输出 5 类（见 CLS_CLASS_TO_CATEGORY）

【一张图检测到多种物体时，归到哪一类】
    priority：按 CATEGORIES 的书写顺序（人像最优先）
    count   ：哪个分类的物体数量最多就归哪类（默认）

【局限性（诚实说明）】
    - 检测侧按"画面里的物体"分类，不做场景理解
    - 分类侧只认识训练过的 5 类；遇到训练集之外的图会"强行"归到某一类
    - 分类模型独立验证集 Top-1 准确率 94.6%，其中 screenshot 与 meme 之间最易混
      （原因：meme 类里包含大量聊天截图，与"截图"类别天然重叠）

【用法】
    # 对比两种归类策略（不复制文件）
    python src/organizer.py -i data/album --dry-run

    # 正式分类（两级流水线）
    python src/organizer.py -i data/album -o output/album_sorted

    # A/B 对照：关掉分类兜底，只看纯检测的效果
    python src/organizer.py -i data/album --dry-run --no-cls
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

# ────────────────── 检测侧：物体类别 → 相册分类 ──────────────────
# 字典顺序 = 优先级（priority 策略下，越靠前越优先）
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

# ────────────────── 分类侧：分类模型类名 → 相册分类 ──────────────────
# 分类模型是在自己采集的 1027 张图上训练的（独立验证集 Top-1 94.6%）
CLS_CLASS_TO_CATEGORY = {
    "screenshot": "截图",
    "document": "文档",
    "scenery": "风景",
    "meme": "表情包",
    "food_cn": "美食餐具",   # 与检测侧的食物分类合并，避免出现两个食物文件夹
}
# 分类侧新增的相册分类（food_cn 已并入「美食餐具」，不重复列出）
CLS_ONLY_CATEGORIES = ["截图", "文档", "风景", "表情包"]

UNKNOWN_CATEGORY = "未识别"
ALL_CATEGORY_ORDER = CATEGORY_ORDER + CLS_ONLY_CATEGORIES + [UNKNOWN_CATEGORY]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ────────────────────────── 检测侧 ──────────────────────────
def pick_category(label_counter: Counter, strategy: str) -> str:
    """根据检测到的物体决定相册分类。

    label_counter: 该图检测到的 类别 -> 数量
    strategy:      priority（按优先级） 或 count（按数量最多）
    """
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

    return min(set(cats), key=CATEGORY_ORDER.index)


# ────────────────────────── 通用 ──────────────────────────
def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def detect_all(images: list[Path], weights: str, conf: float, imgsz: int, batch: int) -> list[dict]:
    """第一级：跑物体检测，只记录"检测到了哪些物体"。

    imgsz 取舍（52 张个人相册实测对照）：
        imgsz=320, conf=0.35  →  未识别率 40.4%
        imgsz=640, conf=0.25  →  未识别率 26.9%
    小尺寸在"物体大而清晰"的数据集上够用且快，但个人相册物体更小更杂，故默认 640。
    """
    model = YOLO(weights)
    records: list[dict] = []

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, verbose=False)
        for path, result in zip(chunk, results):
            labels = [model.names[int(box.cls)] for box in result.boxes]
            records.append({"path": path, "labels": Counter(labels)})
        print(f"  [检测] {min(start + batch, len(images))}/{len(images)} 张", flush=True)

    return records


def classify_fallback(records: list[dict], cls_weights: str, batch: int) -> int:
    """第二级：对"一个物体都没检测到"的图片，用分类模型判断整图类型。

    返回被兜底处理的图片数量。
    """
    pending = [r for r in records if not r["labels"]]
    if not pending:
        return 0

    print(f"  [分类] 有 {len(pending)} 张没检测到物体，交给分类模型兜底判断...")
    model = YOLO(cls_weights)

    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        results = model.predict([str(r["path"]) for r in chunk], imgsz=224, verbose=False)
        for record, result in zip(chunk, results):
            top1 = int(result.probs.top1)
            record["cls_label"] = model.names[top1]
            record["cls_conf"] = float(result.probs.top1conf)
        print(f"  [分类] {min(start + batch, len(pending))}/{len(pending)} 张", flush=True)

    return len(pending)


def category_of(record: dict, strategy: str) -> str:
    """决定一张图最终归到哪个相册分类（两级流水线的核心逻辑）。"""
    if record["labels"]:                      # 第一级有结果 → 按物体分类
        return pick_category(record["labels"], strategy)
    if record.get("cls_label"):               # 第二级兜底 → 按整图类型分类
        return CLS_CLASS_TO_CATEGORY.get(record["cls_label"], UNKNOWN_CATEGORY)
    return UNKNOWN_CATEGORY


def assign(records: list[dict], strategy: str) -> Counter:
    counter: Counter = Counter()
    for r in records:
        r["category"] = category_of(r, strategy)
        counter[r["category"]] += 1
    return counter


def print_summary(records: list[dict], counter: Counter, title: str) -> None:
    total = len(records)
    print()
    print(title)
    print(f"{'分类':<10}{'数量':>6}{'占比':>10}")
    print("-" * 28)
    for name in ALL_CATEGORY_ORDER:
        n = counter.get(name, 0)
        if n:
            print(f"{name:<10}{n:>6}{n / total * 100:>9.1f}%")

    by_detect = sum(1 for r in records if r["labels"])
    by_cls = sum(1 for r in records if not r["labels"] and r.get("cls_label"))
    unknown = counter.get(UNKNOWN_CATEGORY, 0)
    print("-" * 28)
    print(f"判定来源：YOLO 检测 {by_detect} 张 ｜ 分类兜底 {by_cls} 张 ｜ 仍未识别 {unknown} 张")


def write_report(records: list[dict], out_dir: Path, args: argparse.Namespace) -> Path:
    report = out_dir / "report.md"
    counter = Counter(r["category"] for r in records)

    lines = [
        "# 相册分类报告",
        "",
        f"- 输入目录：`{args.input}`",
        f"- 图片总数：{len(records)}",
        f"- 检测权重：`{args.weights}`（imgsz={args.imgsz}, conf={args.conf}）",
        f"- 分类权重：`{args.cls_weights}`" if not args.no_cls else "- 分类兜底：**已关闭**",
        f"- 归类策略：`{args.strategy}`（priority = 按优先级；count = 按数量最多）",
        "",
        "## 判定来源",
        "",
        "| 来源 | 数量 |",
        "|---|---|",
        f"| YOLO 物体检测 | {sum(1 for r in records if r['labels'])} |",
        f"| 分类模型兜底 | {sum(1 for r in records if not r['labels'] and r.get('cls_label'))} |",
        f"| 仍未识别 | {counter.get(UNKNOWN_CATEGORY, 0)} |",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 | 占比 |",
        "|---|---|---|",
    ]
    for name in ALL_CATEGORY_ORDER:
        n = counter.get(name, 0)
        if n:
            lines.append(f"| {name} | {n} | {n / len(records) * 100:.1f}% |")

    lines += ["", "## 明细", ""]
    for name in ALL_CATEGORY_ORDER:
        group = [r for r in records if r["category"] == name]
        if not group:
            continue
        lines.append(f"### {name}（{len(group)} 张）")
        lines.append("")
        for r in group:
            if r["labels"]:
                detail = "、".join(f"{k}×{v}" for k, v in r["labels"].most_common())
                lines.append(f"- `{r['path'].name}` — [检测] {detail}")
            elif r.get("cls_label"):
                lines.append(f"- `{r['path'].name}` — [分类兜底] {r['cls_label']}"
                             f"（置信度 {r['cls_conf']:.3f}）")
            else:
                lines.append(f"- `{r['path'].name}` — 未检出物体，分类模型也未启用")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="相册智能分类工具（YOLO 检测 + 分类兜底）")
    parser.add_argument("-i", "--input", required=True, help="输入目录（放照片的文件夹）")
    parser.add_argument("-o", "--output", default="output/album_sorted", help="输出目录")
    parser.add_argument("--weights", default="weights/best.pt", help="YOLO 检测权重")
    parser.add_argument("--cls-weights", default="weights/cls_best.pt",
                        help="分类模型的权重（用于兜底判定）")
    parser.add_argument("--no-cls", action="store_true",
                        help="关闭分类兜底，只看纯检测的效果（做 A/B 对照用）")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="检测推理尺寸，默认 640。调到 320 会快几倍，但个人相册上漏检更多")
    parser.add_argument("--batch", type=int, default=8, help="批处理大小")
    parser.add_argument("--strategy", choices=["priority", "count"], default="count",
                        help="多物体时的归类策略：priority 按优先级 / count 按数量最多")
    parser.add_argument("--dry-run", action="store_true", help="只预览分类结果，不复制文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(f"目录里没有找到图片：{input_dir}")

    print(f"找到 {len(images)} 张图片")
    print(f"【第一级】YOLO 检测（imgsz={args.imgsz}, conf={args.conf}）...")
    records = detect_all(images, args.weights, args.conf, args.imgsz, args.batch)

    if not args.no_cls:
        print("【第二级】分类模型兜底...")
        classify_fallback(records, args.cls_weights, args.batch)
    else:
        print("【第二级】已按 --no-cls 关闭")

    if args.dry_run:
        # 两种策略都算一遍，便于对比
        print_summary(records, assign(records, "priority"),
                      "【策略 A：priority — 按优先级，人像最优先】")
        print_summary(records, assign(records, "count"),
                      "【策略 B：count — 按检测到的物体数量最多】")
        print()
        print("（--dry-run 模式：没有复制任何文件）")
        return

    counter = assign(records, args.strategy)
    print_summary(records, counter, f"【实际使用策略：{args.strategy}】")

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
