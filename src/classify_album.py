"""相册分类工具（纯分类 + 置信度兜底到「其他」）

【工作方式】
    每一张照片都交给分类模型判断（截图 / 文档 / 风景 / 表情包 / 美食）
        ├── 置信度 ≥ 阈值  →  归入对应分类
        └── 置信度 <  阈值  →  归入「其他」（模型没把握，不强行归类）

【为什么用置信度兜底，而不是训练一个「其他」类】
    曾考虑过训练第 6 类「其他」，但放弃，原因：
      1. 「其他」是个杂类，内容五花八门，要学好需要很大的样本量和多样性
      2. 采集来的"其他"样本容易模糊它与「风景」等已有类别的边界，
         反而会跟那些类抢样本，把原本能分对的图带偏
    所以改用规则兜底：没把握的图直接进「其他」文件夹，交给人复核。

【⚠️ 已知局限：置信度并不完全可靠（实测数据）】
    在 52 张真实相册上人工核对后发现：
        判错的图：置信度 0.99 / 0.93 / 0.77 / 0.70 / 0.66（都很自信）
        判对的图：最低到 0.54 / 0.44 / 0.40（反而不自信）
    即"自信程度"和"是否正确"没有可靠关系（深度学习里叫校准差）。因此：
      - 阈值设高 → 会误伤答对的图
      - 阈值设低 → 拦不住答错的图
    本工具会打印「不同阈值下会有多少张进其他」的对照表，供按实际需要选。

【用法】
    # 预览 + 看阈值对照表（不复制文件）
    python src/classify_album.py -i data/my_photos --dry-run

    # 正式分类（默认阈值 0.6）
    python src/classify_album.py -i data/my_photos -o output/cls_sorted

    # 调阈值：更严（更多图进「其他」）/ 更松 / 完全不兜底
    python src/classify_album.py -i data/my_photos -o out --min-conf 0.8
    python src/classify_album.py -i data/my_photos -o out --min-conf 0.0
"""
import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

# 分类模型输出的英文类名 → 相册分类（中文）
CLS_NAMES = {
    "screenshot": "截图",
    "document": "文档",
    "scenery": "风景",
    "meme": "表情包",
    "food_cn": "美食",
}
OTHER_DIR = "其他"                      # 置信度低于阈值的图放这里
SWEEP_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def classify_all(images: list[Path], weights: str, imgsz: int, batch: int) -> list[dict]:
    """对全部图片跑分类模型（不做任何检测），记录类别与置信度。"""
    model = YOLO(weights)
    records: list[dict] = []

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], imgsz=imgsz, verbose=False)
        for path, result in zip(chunk, results):
            top1 = int(result.probs.top1)
            records.append({
                "path": path,
                "label": model.names[top1],                     # 英文类名
                "category": CLS_NAMES.get(model.names[top1], model.names[top1]),
                "conf": float(result.probs.top1conf),
            })
        print(f"  已处理 {min(start + batch, len(images))}/{len(images)} 张", flush=True)

    return records


def print_summary(records: list[dict], min_conf: float) -> None:
    counter = Counter(r["category"] for r in records)
    confs: dict[str, list[float]] = defaultdict(list)
    for r in records:
        confs[r["category"]].append(r["conf"])
    low = [r for r in records if r["conf"] < min_conf]

    print()
    print(f"{'分类':<10}{'数量':>6}{'占比':>9}{'平均置信度':>12}")
    print("-" * 40)
    for name in list(CLS_NAMES.values()):
        n = counter.get(name, 0)
        if n:
            avg = sum(confs[name]) / len(confs[name])
            print(f"{name:<10}{n:>6}{n / len(records) * 100:>8.1f}%{avg:>12.3f}")
    if low:
        avg = sum(r["conf"] for r in low) / len(low)
        print(f"{OTHER_DIR:<10}{len(low):>6}{len(low) / len(records) * 100:>8.1f}%{avg:>12.3f}")
    print("-" * 40)
    print(f"合计 {len(records)} 张；其中 {len(low)} 张因置信度 < {min_conf} 被归入「{OTHER_DIR}」")


def print_threshold_sweep(records: list[dict]) -> None:
    """打印"不同阈值下会有多少张进其他"的对照表，供选阈值。"""
    total = len(records)
    print()
    print("不同阈值下会被归入「其他」的数量（供你选阈值参考）：")
    for t in SWEEP_THRESHOLDS:
        n = sum(1 for r in records if r["conf"] < t)
        bar = "█" * min(40, n)
        print(f"  {t:.2f} → {n:>3} 张 ({n / total * 100:>5.1f}%)  {bar}")


def write_report(records: list[dict], out_dir: Path, args, min_conf: float) -> Path:
    report = out_dir / "report.md"
    counter = Counter(r["category"] for r in records)
    low = [r for r in records if r["conf"] < min_conf]

    lines = [
        "# 相册分类报告（纯分类 + 置信度兜底）",
        "",
        f"- 输入目录：`{args.input}`",
        f"- 图片总数：{len(records)}",
        f"- 使用权重：`{args.weights}`（imgsz={args.imgsz}）",
        f"- **置信度阈值**：{min_conf}（低于此值的归入「{OTHER_DIR}」）",
        "",
        "> 本工具不使用物体检测，全部照片由 5 类分类模型判断；",
        f"> 模型没把握（置信度 < {min_conf}）的不强行归类，统一放进「{OTHER_DIR}」供人工复核。",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 | 占比 | 平均置信度 |",
        "|---|---|---|---|",
    ]
    confs: dict[str, list[float]] = defaultdict(list)
    for r in records:
        confs[r["category"]].append(r["conf"])
    for name in list(CLS_NAMES.values()):
        n = counter.get(name, 0)
        if n:
            avg = sum(confs[name]) / len(confs[name])
            lines.append(f"| {name} | {n} | {n / len(records) * 100:.1f}% | {avg:.3f} |")
    if low:
        avg = sum(r["conf"] for r in low) / len(low)
        lines.append(f"| {OTHER_DIR} | {len(low)} | {len(low) / len(records) * 100:.1f}% | {avg:.3f} |")

    lines += ["", "## 明细（含置信度，便于人工复核）", ""]
    for name in list(CLS_NAMES.values()):
        group = [r for r in records if r["category"] == name and r["conf"] >= min_conf]
        if not group:
            continue
        lines.append(f"### {name}（{len(group)} 张）")
        lines.append("")
        for r in sorted(group, key=lambda x: -x["conf"]):
            lines.append(f"- `{r['path'].name}` — {r['conf']:.3f}")
        lines.append("")
    if low:
        lines.append(f"### {OTHER_DIR}（{len(low)} 张，置信度低于 {min_conf}）")
        lines.append("")
        for r in sorted(low, key=lambda x: x["conf"]):
            lines.append(f"- `{r['path'].name}` — {r['conf']:.3f}"
                         f"（模型觉得最像「{r['category']}」但没把握）")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="相册分类工具（纯分类 + 置信度兜底到「其他」）")
    parser.add_argument("-i", "--input", required=True, help="输入目录")
    parser.add_argument("-o", "--output", default="output/cls_sorted", help="输出目录")
    parser.add_argument("--weights", default="weights/cls_best.pt", help="分类模型权重")
    parser.add_argument("--imgsz", type=int, default=224, help="分类输入尺寸，默认 224")
    parser.add_argument("--batch", type=int, default=16, help="批处理大小")
    parser.add_argument("--min-conf", type=float, default=0.6,
                        help="置信度阈值：低于此值的归入「其他」，默认 0.6（填 0 表示不兜底）")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不复制文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(f"目录里没有找到图片：{input_dir}")

    print(f"找到 {len(images)} 张图片，用分类模型判断（imgsz={args.imgsz}）...")
    records = classify_all(images, args.weights, args.imgsz, args.batch)
    print_summary(records, args.min_conf)
    print_threshold_sweep(records)

    if args.dry_run:
        print()
        print("（--dry-run 模式：没有复制任何文件）")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        folder = OTHER_DIR if r["conf"] < args.min_conf else r["category"]
        target = out_dir / folder
        target.mkdir(exist_ok=True)
        shutil.copy2(r["path"], target / r["path"].name)

    report = write_report(records, out_dir, args, args.min_conf)
    print()
    print(f"分类完成：{out_dir}")
    print(f"报告文件：{report}")


if __name__ == "__main__":
    main()
