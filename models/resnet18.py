"""ResNet-18 theo biến thể MaxPool giữa các residual stage của notebook."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Hai convolution 3x3 và một identity/projection shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """ResNet-18 với MaxPool 3x3/s2/p0 ở mỗi ranh giới residual stage."""

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(64, 64)
        self.stage_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.stage3 = self._make_stage(64, 128)
        self.stage_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.stage4 = self._make_stage(128, 256)
        self.stage_pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.stage5 = self._make_stage(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.initial_pool(self.relu(self.bn1(self.conv1(x))))
        outputs = {"conv1": x}
        x = self.stage2(x)
        outputs["conv2_x"] = x
        x = self.stage3(self.stage_pool3(x))
        outputs["conv3_x"] = x
        x = self.stage4(self.stage_pool4(x))
        outputs["conv4_x"] = x
        x = self.stage5(self.stage_pool5(x))
        outputs["conv5_x"] = x
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)["conv5_x"]
        return self.classifier(torch.flatten(self.avgpool(x), 1))
