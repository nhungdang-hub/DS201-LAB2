"""Các kiến trúc CNN dùng trong bài thực hành 2."""

from .googlenet import GoogLeNet, InceptionBlock
from .resnet18 import BasicBlock, ResNet18

__all__ = ["GoogLeNet", "InceptionBlock", "BasicBlock", "ResNet18"]
