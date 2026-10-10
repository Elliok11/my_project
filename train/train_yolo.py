"""YOLO 训练脚本 —— 建议在魔搭 ModelScope 免费 GPU 上运行

为什么不在这台笔记本上训练？
    本机是双核 CPU、没有可用于训练的显卡（详见 README"已知问题"），
    训 50 轮 coco128 在本机需要数小时，在免费 GPU 上约 5 分钟。

在魔搭 Notebook 里的用法:
    !pip install ultralytics -q
    !python train/train_yolo.py --data coco128.yaml --epochs 50 --batch 16 --device 0

换成自己的数据集（YOLO 格式）:
    !python train/train_yolo.py --data datasets/helmet/data.yaml --epochs 100

训练完成后:
    权重在 runs/train/weights/best.pt，下载到本机 weights/ 目录，
    然后就能用：python src/ai.py -i 图片 "问题" --weights weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 训练脚本（建议在免费 GPU 上运行）")
    parser.add_argument("--data", default="coco128.yaml",
                        help="数据集配置（YOLO 格式的 yaml），默认 coco128（128 张，自动下载）")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="基础模型，默认官方预训练 yolov8n（迁移学习，比从零训练又快又准）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--device", default="0", help="GPU 编号；用本机 CPU 训练则填 cpu")
    args = parser.parse_args()

    print(f"[训练开始] 数据集={args.data} 模型={args.model} 轮数={args.epochs} "
          f"尺寸={args.imgsz} 批大小={args.batch} 设备={args.device}")

    model = YOLO(args.model)  # 加载预训练权重（迁移学习）
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="train",
    )

    print("\n[训练完成] 最佳权重：runs/train/weights/best.pt")
    print("请下载到本机 weights/ 目录，然后用 --weights 参数推理。")


if __name__ == "__main__":
    main()
