"""Các metric phân loại đa lớp, không phụ thuộc scikit-learn."""

from __future__ import annotations

import torch


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Tính accuracy và precision/recall/F1 macro với zero_division=0."""
    targets = targets.detach().to(torch.long).flatten().cpu()
    predictions = predictions.detach().to(torch.long).flatten().cpu()
    if targets.numel() == 0 or targets.shape != predictions.shape:
        raise ValueError("targets và predictions phải khác rỗng và cùng kích thước.")
    if num_classes <= 0:
        raise ValueError("num_classes phải dương.")

    valid = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    if not bool(valid.all()):
        raise ValueError("Nhãn hoặc dự đoán nằm ngoài khoảng num_classes.")
    indices = targets * num_classes + predictions
    confusion = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    true_positive = confusion.diag().to(torch.float64)
    predicted_positive = confusion.sum(dim=0).to(torch.float64)
    actual_positive = confusion.sum(dim=1).to(torch.float64)
    precision = torch.where(predicted_positive > 0, true_positive / predicted_positive, 0.0)
    recall = torch.where(actual_positive > 0, true_positive / actual_positive, 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }
