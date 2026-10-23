"""Bài 2: huấn luyện GoogLeNet từ đầu trên VinaFood21."""

from __future__ import annotations

import argparse
import json

from torch import nn, optim

from data_utils.vinafood21_dataset import build_vinafood21_dataloaders
from models.googlenet import GoogLeNet
from training.engine import (
    configure_utf8_output,
    evaluate,
    fit,
    resolve_device,
    seed_everything,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/VinaFood21")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    loaders = build_vinafood21_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        args.validation_ratio, args.seed,
    )
    num_classes = len(loaders.class_to_idx)
    print("Thống kê dataset:", json.dumps(loaders.statistics, ensure_ascii=False, indent=2))
    print(f"Thiết bị: {device}")
    model = GoogLeNet(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    fit(
        model, loaders.train, loaders.validation, optimizer, loss_fn, device,
        args.epochs, num_classes, "checkpoints/assignment-2", loaders.class_to_idx,
    )
    if loaders.test is not None:
        print("Test:", evaluate(loaders.test, model, loss_fn, device, num_classes))
    else:
        print("Dataset không có test split; bỏ qua đánh giá test.")


if __name__ == "__main__":
    main()
