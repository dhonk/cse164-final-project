"""From-scratch U-Net for semantic segmentation.

HARD RULE (CLAUDE.md): NO pretrained weights. This is a plain ``nn.Module`` built
from scratch and randomly initialized, so there is nothing to download and the
no-pretrained rule is satisfied by construction.

Output contract (assumed by dataset.py / evaluate.py):
    forward(x) -> seg_logits (B, 301, H, W)
The seg head emits ONE channel per segmentation id, so ``seg_logits.argmax(1)``
yields seg ids directly: 0 = background, 1..300 = foreground (class_id k -> id
k+1). There is no k->k+1 remap anywhere downstream. Logits come back at the input
resolution; predictions are nearest-resized to the original (H, W) for scoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SEG_CLASSES = 301  # 0 = background, 1..300 = foreground seg ids


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2, the standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 (max-pool) then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed-conv upsample, concat the encoder skip, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd spatial sizes so concat always lines up.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Classic 4-stage U-Net.

    ``base_channels`` scales the whole network (encoder dims are
    ``[base, 2*base, 4*base, 8*base]`` with a ``16*base`` bottleneck); drop it to
    e.g. 32 if GPU memory is tight.
    """

    def __init__(self, num_seg_classes: int = NUM_SEG_CLASSES, in_ch: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        # Encoder
        self.inc = DoubleConv(in_ch, c)        # stride 1
        self.down1 = Down(c, c * 2)            # stride 2
        self.down2 = Down(c * 2, c * 4)        # stride 4
        self.down3 = Down(c * 4, c * 8)        # stride 8
        self.down4 = Down(c * 8, c * 16)       # stride 16 (bottleneck)
        # Decoder (in_ch, skip_ch, out_ch)
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, num_seg_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        logits = self.outc(d)
        # Restore exact input resolution if any odd-size rounding crept in.
        if logits.shape[2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits


def build_unet(num_seg_classes: int = NUM_SEG_CLASSES, base_channels: int = 64) -> UNet:
    return UNet(num_seg_classes=num_seg_classes, base_channels=base_channels)


if __name__ == "__main__":
    model = build_unet()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}   # Expected: (2, 301, 256, 256)")
    print(f"Params: {n_params/1e6:.1f}M")
