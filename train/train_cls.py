"""YOLO 图像分类模型训练脚本 —— 建议在魔搭 ModelScope 免费 GPU 上运行

【为什么需要单独训一个分类模型】
    检测模型（detect）只能回答"图里有哪些物体"。
    对于截图、文档、风景这类**没有明确物体**的图片，它无能为力（表现为"未识别"）。
    分类模型回答的是"**整张图属于哪一类**"，正好补上这个缺口。

【为什么分类任务更省事】
    分类不需要画框，只要把图片按类别放进文件夹 —— **零标注成本**。

【在魔搭 Notebook 里的用法】
    !pip install ultralytics -q
    !python train/train_cls.py --data cls_dataset --epochs 50 --batch 32 --device 0

【训练完成后】
    权重在 runs/classify/train/weights/best.pt，下载到本机 weights/，
    然后由 organizer.py 在"检测不到物体"时调用它兜底。
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 图像分类训练（建议在免费 GPU 上运行）")
    parser.add_argument("--data", default="cls_dataset",
                        help="数据集目录（里面应有 train/ 和 val/ 两个子目录）")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="基础模型，默认官方预训练的分类模型 yolov8n-cls（迁移学习）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="分类任务的输入尺寸，默认 224（分类不需要高分辨率，224 就够且很快）")
    parser.add_argument("--batch", type=int, default=32, help="批大小")
    parser.add_argument("--device", default="0", help="GPU 编号；用本机 CPU 训练则填 cpu")
    args = parser.parse_args()

    print(f"[训练开始] 数据={args.data} 模型={args.model} 轮数={args.epochs} "
          f"尺寸={args.imgsz} 批大小={args.batch} 设备={args.device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        # 不指定 project/name：使用 ultralytics 的默认输出目录。
        # 实际保存位置以训练日志里 save_dir= 那一行为准（不同版本可能不同）
    )

    print("\n[训练完成] 结果目录见训练日志里 save_dir= 那一行")
    print("  最佳权重   : <save_dir>/weights/best.pt")
    print("  混淆矩阵   : <save_dir>/confusion_matrix.png   ← 一定要下载，用来分析错分")
    print("  训练曲线   : <save_dir>/results.png")
    print("下载到本机后，权重建议命名为 weights/cls_best.pt，避免和检测权重混淆。")


if __name__ == "__main__":
    main()
