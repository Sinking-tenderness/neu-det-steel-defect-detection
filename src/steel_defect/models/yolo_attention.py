from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        return self.act(self.conv(torch.cat([avg, max_value], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module for YOLO feature maps."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class EMAAttention(nn.Module):
    """Efficient multi-scale attention for dense defect textures.

    This module is lightweight, keeps the feature map shape unchanged, and
    combines horizontal/vertical context with local 3x3 context. It is a better
    fit than heavy attention blocks when NEU-DET images are small and grayscale.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        groups = min(factor, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.groups = max(groups, 1)
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class AttentionWrapper(nn.Module):
    """Wrap a parsed Ultralytics layer while preserving routing metadata."""

    def __init__(self, module: nn.Module, attention: nn.Module, attention_name: str):
        super().__init__()
        self.module = module
        self.attention = attention
        self.attention_name = attention_name
        for attr in ("i", "f", "type", "np"):
            if hasattr(module, attr):
                setattr(self, attr, getattr(module, attr))
        self.type = f"{getattr(module, 'type', module.__class__.__name__)}+{attention_name}"

    def forward(self, x):
        return self.attention(self.module(x))


def _conv_out_channels(obj: object) -> int | None:
    if isinstance(obj, nn.Conv2d):
        return obj.out_channels
    conv = getattr(obj, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return conv.out_channels
    return None


def infer_out_channels(module: nn.Module) -> int | None:
    """Infer common Ultralytics block output channels without a dummy pass."""
    for attr in ("cv3", "cv2", "cv1", "conv"):
        child = getattr(module, attr, None)
        channels = _conv_out_channels(child)
        if channels is not None:
            return channels
    for child in reversed(list(module.modules())):
        channels = _conv_out_channels(child)
        if channels is not None:
            return channels
    return None


def add_attention(model: nn.Module, attention: str, layer_indices: list[int]) -> list[int]:
    """Insert lightweight attention after selected YOLO layers.

    The wrapper keeps Ultralytics routing attributes (`i`, `f`, `type`, `np`), so
    training and validation can still use the standard YOLO forward path.
    """
    if attention == "none":
        return []
    wrapped: list[int] = []
    layers = model.model.model
    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            continue
        module = layers[idx]
        channels = infer_out_channels(module)
        if channels is None:
            continue
        if attention == "cbam":
            attention_module = CBAM(channels)
        elif attention == "ema":
            attention_module = EMAAttention(channels)
        elif attention == "se":
            attention_module = SEAttention(channels)
        else:
            raise ValueError(f"Unsupported attention type: {attention}")
        layers[idx] = AttentionWrapper(module, attention_module, attention.upper())
        wrapped.append(idx)
    return wrapped
