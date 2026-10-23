"""Bài 4: fine-tune microsoft/resnet-50 trên VinaFood21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim

from data_utils.vinafood21_dataset import build_vinafood21_dataloaders
from training.engine import (
    configure_utf8_output,
    evaluate,
    fit,
    resolve_device,
    seed_everything,
)

MODEL_NAME = "microsoft/resnet-50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/VinaFood21")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch ở giai đoạn unfreeze")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="LR của backbone")
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--inference-image", help="Chỉ inference từ model đã lưu")
    parser.add_argument("--model-dir", default="checkpoints/assignment-4/huggingface_model")
    return parser.parse_args()


def load_for_inference(model_dir: str | Path, device: torch.device):
    """Nạp lại model và processor đã lưu bằng save_pretrained."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    return processor, model


@torch.no_grad()
def predict_image(image_path: str | Path, model_dir: str | Path, device: torch.device) -> dict:
    processor, model = load_for_inference(model_dir, device)
    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    logits = model(pixel_values=inputs["pixel_values"].to(device)).logits
    class_id = int(logits.argmax(dim=1).item())
    return {
        "class_id": class_id,
        "label": model.config.id2label[class_id],
        "confidence": float(logits.softmax(dim=1)[0, class_id].item()),
    }


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.inference_image:
        print(json.dumps(predict_image(args.inference_image, args.model_dir, device), ensure_ascii=False))
        return

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    loaders = build_vinafood21_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        args.validation_ratio, args.seed, processor=processor,
    )
    label2id = loaders.class_to_idx
    num_classes = len(label2id)
    id2label = {index: label for label, index in label2id.items()}
    print("Thống kê dataset:", json.dumps(loaders.statistics, ensure_ascii=False, indent=2))
    print(f"Thiết bị: {device}")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    output_dir = Path("checkpoints/assignment-4")

    # Giai đoạn 1: backbone giữ nguyên pretrained weights, chỉ học head mới.
    for parameter in model.resnet.parameters():
        parameter.requires_grad = False
    head_optimizer = optim.Adam(model.classifier.parameters(), lr=args.head_learning_rate)
    fit(
        model, loaders.train, loaders.validation, head_optimizer, loss_fn, device,
        args.head_epochs, num_classes, output_dir / "phase1", label2id,
        frozen_eval_modules=(model.resnet,),
    )

    # Giai đoạn 2: mở stage cuối; dùng LR backbone nhỏ hơn LR head.
    for parameter in model.resnet.encoder.stages[-1].parameters():
        parameter.requires_grad = True
    frozen_backbone = (
        model.resnet.embedder,
        *model.resnet.encoder.stages[:-1],
    )
    optimizer = optim.Adam([
        {"params": model.resnet.encoder.stages[-1].parameters(), "lr": args.learning_rate},
        {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
    ])
    initial_metrics = evaluate(
        loaders.validation, model, loss_fn, device, num_classes
    )
    fit(
        model, loaders.train, loaders.validation, optimizer, loss_fn, device,
        args.epochs, num_classes, output_dir, label2id,
        initial_validation_metrics=initial_metrics,
        frozen_eval_modules=frozen_backbone,
    )

    # Lưu định dạng Hugging Face để nạp lại đồng thời model và processor.
    hf_dir = output_dir / "huggingface_model"
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    if loaders.test is not None:
        print("Test:", evaluate(loaders.test, model, loss_fn, device, num_classes))
    else:
        print("Dataset không có test split; bỏ qua đánh giá test.")


if __name__ == "__main__":
    main()
