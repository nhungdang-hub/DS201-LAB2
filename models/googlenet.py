"""GoogLeNet (Inception v1) cài đặt trực tiếp bằng PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class ConvReLU(nn.Sequential):
    """Convolution theo sau bởi ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
        )


class InceptionBlock(nn.Module):
    """Khối Inception bốn nhánh của GoogLeNet v1."""

    def __init__(
        self,
        in_channels: int,
        conv1_channels: int,
        conv3_reduce_channels: int,
        conv3_channels: int,
        conv5_reduce_channels: int,
        conv5_channels: int,
        pool_proj_channels: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvReLU(in_channels, conv1_channels, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvReLU(in_channels, conv3_reduce_channels, kernel_size=1),
            ConvReLU(conv3_reduce_channels, conv3_channels, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvReLU(in_channels, conv5_reduce_channels, kernel_size=1),
            ConvReLU(conv5_reduce_channels, conv5_channels, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvReLU(in_channels, pool_proj_channels, kernel_size=1),
        )
        self.out_channels = (
            conv1_channels + conv3_channels + conv5_channels + pool_proj_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x))
        return torch.cat(branches, dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet không auxiliary classifier, đầu ra là logits của 21 lớp."""

    stage_channels = {
        "3a": 256,
        "3b": 480,
        "4a": 512,
        "4b": 512,
        "4c": 512,
        "4d": 528,
        "4e": 832,
        "5a": 832,
        "5b": 1024,
    }

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        # Tên conv1 được giữ rõ ràng để dễ kiểm tra padding theo đề bài.
        self.conv1 = ConvReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2_reduce = ConvReLU(64, 64, kernel_size=1)
        self.conv2 = ConvReLU(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Trả feature map từng Inception stage để kiểm tra kích thước."""
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(self.conv2_reduce(x)))
        features: dict[str, torch.Tensor] = {}
        for name in ("3a", "3b"):
            x = getattr(self, f"inception{name}")(x)
            features[name] = x
        x = self.maxpool3(x)
        for name in ("4a", "4b", "4c", "4d", "4e"):
            x = getattr(self, f"inception{name}")(x)
            features[name] = x
        x = self.maxpool4(x)
        for name in ("5a", "5b"):
            x = getattr(self, f"inception{name}")(x)
            features[name] = x
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)["5b"]
        x = torch.flatten(self.avgpool(x), 1)
        return self.classifier(self.dropout(x))
