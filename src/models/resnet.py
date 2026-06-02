"""From-scratch ResNet-18 for image classification.

HARD RULE (CLAUDE.md): NO pretrained weights. This is a plain ``nn.Module`` built
from scratch and randomly initialized -- nothing is downloaded.

Output contract: ``forward(x) -> class_logits (B, 300)`` (class_id 0..299).
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 300


class BasicBlock(nn.Module):
    """ResNet basic residual block: two 3x3 convs + identity/projection skip."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # Projection shortcut when shape changes (stride>1 or channel mismatch).
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet with a configurable block layout (default ResNet-18: (2,2,2,2))."""

    def __init__(
        self,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_classes: int = NUM_CLASSES,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        c = base_channels
        self.in_ch = c
        # Stem: 7x7 stride-2 conv + max-pool (standard ImageNet-style stem).
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(c, layers[0], stride=1)
        self.layer2 = self._make_layer(c * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(c * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(c * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_resnet(num_classes: int = NUM_CLASSES, base_channels: int = 64) -> ResNet:
    return ResNet((2, 2, 2, 2), num_classes=num_classes, base_channels=base_channels)


if __name__ == "__main__":
    model = build_resnet()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}   # Expected: (2, 300)")
    print(f"Params: {n_params/1e6:.1f}M")
