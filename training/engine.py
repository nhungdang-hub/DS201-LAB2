"""Training/evaluation pipeline dùng chung cho model PyTorch và Hugging Face."""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics


def configure_utf8_output() -> None:
    """Giữ log tiếng Việt hoạt động cả khi Windows redirect stdout/stderr."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def seed_everything(seed: int) -> None:
    """Cố định các nguồn ngẫu nhiên phổ biến."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng torch.cuda.is_available() là False.")
    return device


def _logits(model_output: Any) -> torch.Tensor:
    """Chuẩn hóa output Tensor hoặc ImageClassifierOutput về logits."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    raise TypeError(f"Không nhận diện được output model: {type(model_output)!r}")


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    frozen_eval_modules: Sequence[nn.Module] = (),
) -> float:
    model.train()
    # requires_grad=False không ngăn BatchNorm cập nhật running statistics.
    # Các module backbone bị đóng băng phải tiếp tục ở eval mode trong phase 1.
    for module in frozen_eval_modules:
        module.eval()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = _logits(model(images))
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")
    if sample_count == 0:
        raise ValueError("Training DataLoader rỗng.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Đánh giá đúng với mọi batch_size; không dùng test để chọn checkpoint."""
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    all_targets, all_predictions = [], []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = _logits(model(images))
        loss = loss_fn(logits, labels)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        all_targets.append(labels.cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())
    if sample_count == 0:
        raise ValueError("Evaluation DataLoader rỗng.")
    metrics = classification_metrics(
        torch.cat(all_targets), torch.cat(all_predictions), num_classes=num_classes
    )
    return {"loss": loss_sum / sample_count, **metrics}


def _plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_f1_macro"] for row in history], label="F1-macro")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Accuracy")
    axes[1].set(title="Validation metrics", xlabel="Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    num_classes: int,
    checkpoint_dir: str | Path,
    class_to_idx: dict[str, int],
    initial_best_score: float = float("-inf"),
    frozen_eval_modules: Sequence[nn.Module] = (),
) -> tuple[list[dict[str, float]], float]:
    """Train và lưu checkpoint tốt nhất dựa duy nhất trên validation F1-macro."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_score = initial_best_score
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            frozen_eval_modules=frozen_eval_modules,
        )
        validation = evaluate(validation_loader, model, loss_fn, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation["f1_macro"] > best_score:
            best_score = validation["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "class_to_idx": class_to_idx,
                },
                best_path,
            )

    history_path = checkpoint_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    if history:
        _plot_history(history, checkpoint_dir / "learning_curves.png")
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return history, best_score
