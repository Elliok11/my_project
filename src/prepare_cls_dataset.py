"""把"随手收集的图片"整理成分类任务所需的 train/val 目录结构

【为什么分类任务比检测任务省事】
    检测（detect）要回答"物体在哪"，所以必须**逐个画框**；
    分类（classify）只要回答"整张图属于哪类"，所以**把图片丢进对应文件夹就行**，零标注成本。

【你要做的】
    1. 在 data/cls_raw/ 下，为每个类别建一个文件夹，把图片丢进去：
           data/cls_raw/
           ├── screenshot/    截图
           ├── document/      文档、试卷、笔记
           ├── scenery/       风景
           ├── meme/          表情包、聊天图
           └── food_cn/       中式餐食
       ⚠️ 类别文件夹名必须是纯英文/数字/下划线（中文名会导致云端训练崩溃，见下）
    2. 运行本脚本，自动按比例划分成 train / val，并把文件名统一成纯 ASCII：
           python src/prepare_cls_dataset.py

【为什么必须重命名文件（真实踩过的坑）】
    Windows 上的中文文件名（如 微信图片_20261010.jpg）经过 "zip → 上传到 Linux 云端 → 解压"
    之后，编码会损坏，在 Python 里变成非法字符（surrogate）。
    训练时计算数据集哈希会直接崩溃：
        UnicodeEncodeError: 'utf-8' codec can't encode characters ... surrogates not allowed
    所以本脚本把所有图片统一重命名为 <类别>_<序号>.jpg（纯 ASCII），
    并生成 manifest.csv 记录「原文件名 → 新文件名」的对照，便于追溯来源。

【设计说明】
    - 用固定随机种子划分，保证结果可复现（每次运行划分一致）
    - 复制时把长边缩到 512（分类训练只需 224，缩小体积可让上传快 10~20 倍）
    - 每类少于 50 张会告警：数据太少，模型学不会
    - 类别不均衡会提示：某类特别多/特别少时，模型会偏向多数类
"""
import argparse
import csv
import random
import re
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES_PER_CLASS = 50
CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def copy_image(src: Path, dst: Path, max_size: int) -> None:
    """复制图片。max_size > 0 时把长边缩到该尺寸。

    为什么可以缩放：分类模型训练时的输入只有 224×224，
    原图 4000×3000 的像素纯属浪费 —— 缩放后数据集体积能小 10~20 倍，
    上传到云端和训练的等待时间都大幅缩短。
    """
    if max_size <= 0:
        shutil.copy2(src, dst)
        return
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            im.save(dst, quality=90)
    except Exception:
        shutil.copy2(src, dst)  # 个别格式处理失败时退回原样复制


def main() -> None:
    parser = argparse.ArgumentParser(description="把图片整理成分类任务的 train/val 结构")
    parser.add_argument("--raw", default="data/cls_raw", help="你放原始图片的目录（每个子目录 = 一个类别）")
    parser.add_argument("--out", default="data/cls_dataset", help="输出目录（train/val 结构）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--max-size", type=int, default=512,
                        help="复制时长边缩放到该尺寸（默认 512，分类只需 224；填 0 表示不缩放）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，保证划分可复现")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.is_dir():
        raise SystemExit(f"找不到原始目录：{raw_dir}\n请先建好目录并把图片按类别放进去。")

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"{raw_dir} 下没有子目录。请为每个类别建一个文件夹，例如 {raw_dir}/screenshot/")

    # 类别名必须是纯 ASCII —— 中文类别名经 zip→Linux 往返后会损坏，训练直接崩溃
    bad_names = [d.name for d in class_dirs if not CLASS_NAME_PATTERN.match(d.name)]
    if bad_names:
        raise SystemExit(
            "❌ 类别文件夹名不合规：\n"
            f"   {', '.join(bad_names)}\n"
            "   必须是纯英文/数字/下划线（例如 screenshot / document / food_cn）。\n"
            "   原因：中文名在 Windows→zip→Linux 的往返中会编码损坏，导致训练崩溃。"
        )

    # 提醒被忽略的非图片文件
    skipped = [p for p in raw_dir.rglob("*")
               if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS]
    if skipped:
        print(f"⚠️ 有 {len(skipped)} 个文件不是支持的图片格式，将被忽略（如 .gif）：")
        for p in skipped[:5]:
            print(f"   {p.relative_to(raw_dir)}")
        print()

    # 检查输出目录是否已在 git 外，避免误提交
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    random.seed(args.seed)
    stats = []
    manifest: list[dict] = []

    for cls_dir in class_dirs:
        images = find_images(cls_dir)
        if not images:
            print(f"⚠️ 跳过空类别：{cls_dir.name}")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split, items in (("train", train_images), ("val", val_images)):
            target = out_dir / split / cls_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for index, img in enumerate(items, start=1):
                # 统一重命名为纯 ASCII：<类别>_<序号>.jpg，彻底绕开编码问题
                name = f"{cls_dir.name}_{index:03d}.jpg"
                copy_image(img, target / name, args.max_size)
                manifest.append({
                    "split": split,
                    "class": cls_dir.name,
                    "new_name": f"{split}/{cls_dir.name}/{name}",
                    "original": str(img.relative_to(raw_dir)),
                })

        stats.append({"类别": cls_dir.name, "总数": len(images),
                      "train": len(train_images), "val": len(val_images)})

    if not stats:
        raise SystemExit("没有找到任何图片。请把图片放进各类别文件夹后重试。")

    print()
    print(f"{'类别':<16}{'总数':>7}{'train':>8}{'val':>7}")
    print("-" * 40)
    for s in sorted(stats, key=lambda x: -x["总数"]):
        print(f"{s['类别']:<16}{s['总数']:>7}{s['train']:>8}{s['val']:>7}")
    total = sum(s["总数"] for s in stats)
    print("-" * 40)
    print(f"{'合计':<16}{total:>7}")

    # 检查提醒
    print()
    too_few = [s["类别"] for s in stats if s["总数"] < MIN_IMAGES_PER_CLASS]
    if too_few:
        print(f"⚠️ 以下类别少于 {MIN_IMAGES_PER_CLASS} 张，模型很可能学不会：{('、'.join(too_few))}")
    counts = [s["总数"] for s in stats]
    if counts and max(counts) > min(counts) * 3:
        print("⚠️ 类别数量差异超过 3 倍，模型会偏向多数类。建议补充少的那几类。")
    if len(stats) < 2:
        print("⚠️ 只有 1 个类别，分类任务至少需要 2 类。")

    # 写重命名对照表（便于追溯每张图的原始来源）
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "new_name", "original"])
        writer.writeheader()
        writer.writerows(manifest)

    print()
    print(f"✅ 数据集已生成：{out_dir}")
    print(f"✅ 重命名对照表：{manifest_path}")
    print("   下一步：把 data/cls_dataset 打包上传到魔搭，运行 train/train_cls.py")


if __name__ == "__main__":
    main()
