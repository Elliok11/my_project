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
    2. 运行本脚本，自动按比例划分成 train / val：
           python src/prepare_cls_dataset.py

【设计说明】
    - 用固定随机种子划分，保证结果可复现（每次运行划分一致）
    - 保留原文件名（便于追溯来源），统一转成 .jpg；遇到重名自动加序号，不会互相覆盖
    - 每类少于 50 张会告警：数据太少，模型学不会
    - 类别不均衡会提示：某类特别多/特别少时，模型会偏向多数类
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES_PER_CLASS = 50


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

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    random.seed(args.seed)
    stats = []

    for cls_dir in class_dirs:
        images = find_images(cls_dir)
        if not images:
            print(f"⚠️  跳过空类别：{cls_dir.name}")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split, items in (("train", train_images), ("val", val_images)):
            target = out_dir / split / cls_dir.name
            target.mkdir(parents=True, exist_ok=True)
            used_names: set[str] = set()
            for img in items:
                # 保留原文件名（便于追溯来源），统一转成 .jpg；
                # 不同子目录里可能有同名文件，重名时自动加序号，避免互相覆盖
                name = f"{img.stem}.jpg"
                if name in used_names or (target / name).exists():
                    i = 2
                    while (f"{img.stem}_{i}.jpg" in used_names
                           or (target / f"{img.stem}_{i}.jpg").exists()):
                        i += 1
                    name = f"{img.stem}_{i}.jpg"
                used_names.add(name)
                copy_image(img, target / name, args.max_size)

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
        print(f"⚠️  以下类别少于 {MIN_IMAGES_PER_CLASS} 张，模型很可能学不会：{('、'.join(too_few))}")
    counts = [s["总数"] for s in stats]
    if counts and max(counts) > min(counts) * 3:
        print("⚠️  类别数量差异超过 3 倍，模型会偏向多数类。建议补充少的那几类。")
    if len(stats) < 2:
        print("⚠️  只有 1 个类别，分类任务至少需要 2 类。")

    print()
    print(f"✅ 数据集已生成：{out_dir}")
    print("   下一步：把 data/cls_dataset 整个目录上传到魔搭，运行 train/train_cls.py")


if __name__ == "__main__":
    main()
