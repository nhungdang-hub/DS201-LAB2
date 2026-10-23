"""Unit tests kiến trúc, metrics và một training step tổng hợp."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models.googlenet import GoogLeNet, InceptionBlock
from models.resnet18 import BasicBlock, ResNet18
from training.metrics import classification_metrics
from training.engine import evaluate, fit, train_one_epoch


@torch.no_grad()
def test_googlenet_forward_and_inception_channels() -> None:
    model = GoogLeNet(num_classes=21).eval()
    inputs = torch.randn(2, 3, 224, 224)
    features = model.forward_features(inputs)
    assert {name: value.shape[1] for name, value in features.items()} == model.stage_channels
    assert features["3a"].shape[2:] == (28, 28)
    assert features["4e"].shape[2:] == (14, 14)
    assert features["5b"].shape[2:] == (7, 7)
    assert model(inputs).shape == (2, 21)


def test_googlenet_required_convolution_and_pooling() -> None:
    model = GoogLeNet()
    assert model.conv1[0].kernel_size == (7, 7)
    assert model.conv1[0].stride == (2, 2)
    assert model.conv1[0].padding == (3, 3)
    pools = [module for module in model.modules() if isinstance(module, nn.MaxPool2d)]
    assert pools
    assert all(module.ceil_mode for module in pools)
    assert len([module for module in model.modules() if isinstance(module, InceptionBlock)]) == 9


@torch.no_grad()
def test_resnet18_forward_blocks_and_special_pooling() -> None:
    model = ResNet18(num_classes=21).eval()
    inputs = torch.randn(2, 3, 224, 224)
    features = model.forward_features(inputs)
    assert model(inputs).shape == (2, 21)
    assert len([module for module in model.modules() if isinstance(module, BasicBlock)]) == 8
    assert features["conv2_x"].shape == (2, 64, 56, 56)
    # padding=0 theo đề tạo 56 -> 27 -> 13 -> 6, khác bảng ResNet chuẩn.
    assert features["conv3_x"].shape == (2, 128, 27, 27)
    assert features["conv4_x"].shape == (2, 256, 13, 13)
    assert features["conv5_x"].shape == (2, 512, 6, 6)
    for pool in (model.stage_pool3, model.stage_pool4, model.stage_pool5):
        assert pool.kernel_size == 3
        assert pool.stride == 2
        assert pool.padding == 0


@torch.no_grad()
def test_projection_shortcut_has_no_shape_mismatch() -> None:
    block = BasicBlock(64, 128).eval()
    assert block(torch.randn(2, 64, 16, 16)).shape == (2, 128, 16, 16)


def test_metrics_support_batch_larger_than_one() -> None:
    targets = torch.tensor([0, 1, 2, 2, 1, 0])
    predictions = torch.tensor([0, 1, 1, 2, 0, 0])
    metrics = classification_metrics(targets, predictions, num_classes=3)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert set(metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_synthetic_forward_backward_optimizer_step() -> None:
    model = ResNet18(num_classes=21)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([1, 7])
    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_frozen_backbone_keeps_batchnorm_statistics() -> None:
    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"image": torch.randn(3, 8, 8), "label": torch.tensor(index % 2)}

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
            self.classifier = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            features = self.backbone(inputs).mean(dim=(2, 3))
            return self.classifier(features)

    model = TinyClassifier()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    running_mean = model.backbone[1].running_mean.clone()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    train_one_epoch(
        DataLoader(TinyDataset(), batch_size=2),
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        frozen_eval_modules=(model.backbone,),
    )
    assert torch.equal(model.backbone[1].running_mean, running_mean)
    assert model.backbone.training is False


def test_engine_accepts_huggingface_style_logits_object() -> None:
    class ObjectOutputModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=inputs)

    samples = [
        {"image": torch.tensor([4.0, 1.0]), "label": torch.tensor(0)},
        {"image": torch.tensor([1.0, 4.0]), "label": torch.tensor(1)},
    ]
    metrics = evaluate(
        DataLoader(samples, batch_size=2),
        ObjectOutputModel(),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=2,
    )
    assert metrics["accuracy"] == 1.0


def test_fit_restores_initial_best_when_new_epoch_is_worse(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(7.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        [{"image": torch.ones(1), "label": torch.tensor(0)}], batch_size=1
    )

    def fake_train(*args, **kwargs) -> float:
        with torch.no_grad():
            model.weight.fill_(1.0)
        return 0.5

    def fake_evaluate(*args, **kwargs) -> dict[str, float]:
        return {
            "loss": 0.5,
            "accuracy": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
            "f1_macro": 0.5,
        }

    monkeypatch.setattr("training.engine.train_one_epoch", fake_train)
    monkeypatch.setattr("training.engine.evaluate", fake_evaluate)
    monkeypatch.setattr("training.engine._plot_history", lambda *args: None)
    initial_metrics = {
        "loss": 0.2,
        "accuracy": 0.9,
        "precision_macro": 0.9,
        "recall_macro": 0.9,
        "f1_macro": 0.9,
    }
    fit(
        model,
        loader,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        epochs=1,
        num_classes=2,
        checkpoint_dir=tmp_path,
        class_to_idx={"a": 0, "b": 1},
        initial_validation_metrics=initial_metrics,
    )
    assert torch.equal(model.weight, torch.full_like(model.weight, 7.0))
    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    assert checkpoint["epoch"] == 0
    assert checkpoint["validation_metrics"]["f1_macro"] == 0.9


def test_huggingface_resnet50_classifier_when_cached() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=21,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        pytest.skip(f"Checkpoint Hugging Face chưa có trong cache: {exc}")
    assert model.config.num_labels == 21
    assert model.classifier[-1].out_features == 21
