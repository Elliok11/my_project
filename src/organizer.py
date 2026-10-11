"""相册智能分类工具 —— 分类优先的两级流水线

【整体架构】
    照片 → 【第一级】分类模型判断
              ├── 截图 / 文档 / 风景 / 表情包 / 美食  → 直接采纳
              └── 「其他」                          → 交给【第二级】YOLO 检测细分
                                                        （人像 / 宠物动物 / 交通工具 …）

【为什么是"分类优先"而不是"检测优先"】
    实测对照（52 张真实相册照片，结果经人工逐张核对）：

        检测优先：风景只找到 9 张 —— 另外 15 张被 YOLO 的误检"截流"，
                  错分到了「人像」「宠物动物」「交通工具」等分类
        分类优先：风景找到 24 张 —— 人工核对全部正确 ✅

    原因：YOLO 在风景照上会误检（从风景里"看出"人物、长椅、大象、车辆）。
          在"检测优先"的架构下，这些图在第一级就被截流，分类模型没有机会纠正。
          调换顺序后，分类模型先说话，"没有物体"的图不会再被误检带偏。

【⚠️ 当前状态（重要）】
    分类模型目前只训练了 5 类，还没有「其他」类，
    所以现阶段它永远会输出 5 类之一 → 第二级检测实际上不会被调用（≈ 纯分类）。
    等重训加入「其他」类后，本逻辑会自动完整生效，**无需改代码**。

【关于置信度阈值】
    实测发现模型置信度**不可信**：判错的图置信度高达 0.99，判对的图反而只有 0.4。
    所以默认 `--cls-conf 0.0`（不做阈值拦截），
    只依据"分类结果是不是「其他」"来决定要不要交给检测。

【同一张图检测到多种物体时的归类策略】
    priority：按 CATEGORIES 的书写顺序（人像最优先）
    count   ：哪个分类的物体数量最多就归哪类（默认）

【用法】
    # 预览（不复制文件）
    python src/organizer.py -i data/album --dry-run

    # 正式分类
    python src/organizer.py -i data/album -o output/album_sorted
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
CLS_CLASS_TO_CATEGORY = {
    "screenshot": "截图",
    "document": "文档",
    "scenery": "风景",
    "meme": "表情包",
    "food_cn": "美食餐具",     # 与检测侧的食物分类合并
    "other": None,              # 「其他」→ None 表示"交给检测细分"（重训出该类后生效）
}
CLS_ONLY_CATEGORIES = ["截图", "文档", "风景", "表情包"]

UNKNOWN_CATEGORY = "未识别"
ALL_CATEGORY_ORDER = CATEGORY_ORDER + CLS_ONLY_CATEGORIES + [UNKNOWN_CATEGORY]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ────────────────────────── 检测侧逻辑 ──────────────────────────
def pick_category(label_counter: Counter, strategy: str) -> str:
    """根据检测到的物体决定相册分类（仅第二级使用）。"""
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
        return min(candidates, key=CATEGORY_ORDER.index)

    return min(set(cats), key=CATEGORY_ORDER.index)


# ────────────────────────── 通用 ──────────────────────────
def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_classifier(records: list[dict], weights: str, imgsz: int, batch: int,
                   conf_threshold: float) -> None:
    """【第一级】跑分类模型，并判断哪些图可以直接采纳分类结果。

    采纳规则：分类结果能映射到具体分类（不是「其他」），且置信度 ≥ 阈值。
    """
    model = YOLO(weights)
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        results = model.predict([str(r["path"]) for r in chunk], imgsz=imgsz, verbose=False)
        for record, result in zip(chunk, results):
            label = model.names[int(result.probs.top1)]
            conf = float(result.probs.top1conf)
            category = CLS_CLASS_TO_CATEGORY.get(label, None)
            record["cls_label"] = label
            record["cls_conf"] = conf
            record["cls_category"] = category
            record["cls_accept"] = (category is not None and conf >= conf_threshold)
        print(f"  [第一级·分类] {min(start + batch, len(records))}/{len(records)} 张", flush=True)


def run_detector(records: list[dict], weights: str, conf: float, imgsz: int, batch: int) -> None:
    """【第二级】跑物体检测（只对第一级判定为「其他」的图片执行）。"""
    model = YOLO(weights)
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        results = model.predict([str(r["path"]) for r in chunk], conf=conf, imgsz=imgsz, verbose=False)
        for record, result in zip(chunk, results):
            record["labels"] = Counter(model.names[int(box.cls)] for box in result.boxes)
        print(f"  [第二级·检测] {min(start + batch, len(records))}/{len(records)} 张", flush=True)


def category_of(record: dict, strategy: str) -> str:
    """决定一张图最终归到哪个相册分类。"""
    # ① 第一级分类结果被采纳 → 直接用
    if record.get("cls_accept"):
        return record["cls_category"]
    # ② 第二级检测有结果 → 按物体分类
    if record.get("labels"):
        return pick_category(record["labels"], strategy)
    # ③ 两级都没给出具体结果 → 退回分类模型原先的判断（若有）
    if record.get("cls_category"):
        return record["cls_category"]
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

    by_cls = sum(1 for r in records if r.get("cls_accept"))
    by_det = sum(1 for r in records if not r.get("cls_accept") and r.get("labels"))
    unknown = counter.get(UNKNOWN_CATEGORY, 0)
    print("-" * 28)
    print(f"判定来源：分类直接采纳 {by_cls} 张 ｜ 交给检测 {by_det} 张 ｜ 仍未识别 {unknown} 张")


def write_report(records: list[dict], out_dir: Path, args: argparse.Namespace) -> Path:
    report = out_dir / "report.md"
    counter = Counter(r["category"] for r in records)

    lines = [
        "# 相册分类报告",
        "",
        f"- 输入目录：`{args.input}`",
        f"- 图片总数：{len(records)}",
        "- 流水线：**分类优先**（先分类模型判断，遇「其他」才交给 YOLO 检测细分）",
        f"- 分类权重：`{args.cls_weights}`（imgsz={args.cls_imgsz}, 采纳阈值={args.cls_conf}）",
        f"- 检测权重：`{args.weights}`（imgsz={args.imgsz}, conf={args.conf}）",
        f"- 多物体归类策略：`{args.strategy}`",
        "",
        "## 判定来源",
        "",
        "| 来源 | 数量 |",
        "|---|---|",
        f"| 第一级：分类模型直接采纳 | {sum(1 for r in records if r.get('cls_accept'))} |",
        f"| 第二级：交给 YOLO 检测判定 | {sum(1 for r in records if not r.get('cls_accept') and r.get('labels'))} |",
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
            if r.get("cls_accept"):
                detail = f"[分类] {r['cls_label']}（置信度 {r['cls_conf']:.3f}）"
            elif r.get("labels"):
                detail = "[检测] " + "、".join(f"{k}×{v}" for k, v in r["labels"].most_common())
            elif r.get("cls_label"):
                detail = f"[分类·未达阈值] {r['cls_label']}（置信度 {r['cls_conf']:.3f}）"
            else:
                detail = "未判定"
            lines.append(f"- `{r['path'].name}` — {detail}")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="相册智能分类工具（分类优先的两级流水线）")
    parser.add_argument("-i", "--input", required=True, help="输入目录（放照片的文件夹）")
    parser.add_argument("-o", "--output", default="output/album_sorted", help="输出目录")
    parser.add_argument("--cls-weights", default="weights/cls_best.pt", help="分类模型权重")
    parser.add_argument("--cls-conf", type=float, default=0.0,
                        help="分类结果被直接采纳的置信度门槛。默认 0（不拦截）"
                             "—— 实测置信度不可信（判错的 0.99、判对的 0.4）")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="分类输入尺寸，默认 224")
    parser.add_argument("--weights", default="weights/best.pt", help="YOLO 检测权重（第二级用）")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=640, help="检测推理尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=8, help="批处理大小")
    parser.add_argument("--strategy", choices=["priority", "count"], default="count",
                        help="多物体时的归类策略")
    parser.add_argument("--dry-run", action="store_true", help="只预览分类结果，不复制文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(f"目录里没有找到图片：{input_dir}")

    records: list[dict] = [{"path": p} for p in images]
    print(f"找到 {len(images)} 张图片，流水线 = 分类优先")

    # 【第一级】分类模型
    print("【第一级】分类模型判断...")
    run_classifier(records, args.cls_weights, args.cls_imgsz, args.batch, args.cls_conf)

    # 【第二级】只对"分类结果不是具体类别"的图跑检测
    need_detect = [r for r in records if not r.get("cls_accept")]
    if need_detect:
        print(f"【第二级】{len(need_detect)} 张需要交给 YOLO 检测细分...")
        run_detector(need_detect, args.weights, args.conf, args.imgsz, args.batch)
    else:
        print("【第二级】无需检测（分类模型已全部给出具体类别）")

    if args.dry_run:
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
