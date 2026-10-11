"""只用分类模型整理相册 —— 不走物体检测那一级

【它和 organizer.py 的区别】

    organizer.py      两级流水线：YOLO 检测优先 → 认不出的才交给分类模型
                      结果 = 检测的 12 类（人像/宠物动物/美食餐具/…）+ 分类的 4 类

    classify_album.py 单级：所有照片都交给分类模型 → 只有 5 类
                      （截图 / 美食 / 风景 / 文档 / 表情包）

【什么时候用这个】
    - 你的相册内容本来就集中在这 5 类
    - 你想单独看看分类模型工作的效果
    - 你想对比"纯分类"和"两级流水线"哪个更适合你

【⚠️ 必须知道的代价】
    分类模型只认识 5 类。相册里如果有"人像""宠物""汽车""家具"这类照片，
    它没有对应类别，只能**强行归到最像的那一类** → 必然产生误判。
    所以本工具额外做两件事：
      1. 记录每张图的置信度
      2. 把置信度低于阈值（默认 0.6）的单独放进「低置信度」文件夹，供人工复核

【用法】
    # 先预览（不复制文件）
    python src/classify_album.py -i data/my_photos --dry-run

    # 正式分类
    python src/classify_album.py -i data/my_photos -o output/cls_sorted

    # 调整置信度门槛 / 换权重
    python src/classify_album.py -i data/my_photos --min-conf 0.5 --weights weights/cls_best.pt
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
LOW_CONF_DIR = "低置信度"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def classify_all(images: list[Path], weights: str, imgsz: int, batch: int) -> list[dict]:
    """对全部图片跑分类模型（不做任何检测）。"""
    model = YOLO(weights)
    records: list[dict] = []

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], imgsz=imgsz, verbose=False)
        for path, result in zip(chunk, results):
            top1 = int(result.probs.top1)
            records.append({
                "path": path,
                "label": model.names[top1],                    # 英文类名
                "category": CLS_NAMES.get(model.names[top1], model.names[top1]),
                "conf": float(result.probs.top1conf),
            })
        print(f"  已处理 {min(start + batch, len(images))}/{len(images)} 张", flush=True)

    return records


def print_summary(records: list[dict], min_conf: float) -> None:
    print()
    print(f"{'分类':<10}{'数量':>6}{'占比':>9}{'平均置信度':>12}")
    print("-" * 40)

    counter = Counter(r["category"] for r in records)
    confs: dict[str, list[float]] = defaultdict(list)
    for r in records:
        confs[r["category"]].append(r["conf"])

    for name in list(CLS_NAMES.values()):
        n = counter.get(name, 0)
        if n:
            avg = sum(confs[name]) / len(confs[name])
            print(f"{name:<10}{n:>6}{n / len(records) * 100:>8.1f}%{avg:>12.3f}")

    low = [r for r in records if r["conf"] < min_conf]
    print("-" * 40)
    print(f"合计 {len(records)} 张；其中置信度低于 {min_conf} 的：{len(low)} 张"
          f"（{len(low) / len(records) * 100:.1f}%）→ 建议人工复核")


def write_report(records: list[dict], out_dir: Path, args, min_conf: float) -> Path:
    report = out_dir / "report.md"
    counter = Counter(r["category"] for r in records)

    lines = [
        "# 相册分类报告（纯分类模型，无物体检测）",
        "",
        f"- 输入目录：`{args.input}`",
        f"- 图片总数：{len(records)}",
        f"- 使用权重：`{args.weights}`（imgsz={args.imgsz}）",
        f"- 置信度门槛：{min_conf}",
        "",
        "> ⚠️ 本模式**不使用 YOLO 物体检测**，全部照片都交给 5 类分类模型判断。",
        "> 相册中不属于这 5 类的照片（如人像、宠物、汽车）会被强制归类，可能误判。",
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

    lines += ["", "## 明细（含置信度，便于人工复核）", ""]
    for name in list(CLS_NAMES.values()):
        group = [r for r in records if r["category"] == name]
        if not group:
            continue
        lines.append(f"### {name}（{len(group)} 张）")
        lines.append("")
        for r in sorted(group, key=lambda x: -x["conf"]):
            flag = " ⚠️低置信度" if r["conf"] < min_conf else ""
            lines.append(f"- `{r['path'].name}` — {r['conf']:.3f}{flag}")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="只用分类模型整理相册（5 类）")
    parser.add_argument("-i", "--input", required=True, help="输入目录")
    parser.add_argument("-o", "--output", default="output/cls_sorted", help="输出目录")
    parser.add_argument("--weights", default="weights/cls_best.pt", help="分类模型权重")
    parser.add_argument("--imgsz", type=int, default=224, help="分类输入尺寸，默认 224")
    parser.add_argument("--batch", type=int, default=16, help="批处理大小")
    parser.add_argument("--min-conf", type=float, default=0.6,
                        help="置信度门槛，低于此值的会被单独放进「低置信度」文件夹")
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

    if args.dry_run:
        print()
        print("（--dry-run 模式：没有复制任何文件）")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        folder = LOW_CONF_DIR if r["conf"] < args.min_conf else r["category"]
        target = out_dir / folder
        target.mkdir(exist_ok=True)
        shutil.copy2(r["path"], target / r["path"].name)

    report = write_report(records, out_dir, args, args.min_conf)
    print()
    print(f"分类完成：{out_dir}")
    print(f"报告文件：{report}")


if __name__ == "__main__":
    main()
