"""Model definitions.

HARD RULE (CLAUDE.md): NO pretrained weights. Every layer must be randomly
initialized and trained on the provided competition data only. Whenever a
torchvision/timm constructor is used, pass weights=None / pretrained=False.

Multi-task design: a shared timm encoder (ConvNeXt-V2 Atto, from scratch) feeds
both a classification head (300 classes) and a U-Net segmentation decoder
(301 logits: background id 0 + foreground ids 1..300). The seg head emits one
channel per segmentation id, so ``seg_logits.argmax(1)`` yields seg ids directly
(0 = background, k = foreground class k-1), matching the meters in evaluate.py
and the mask spec in CLAUDE.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


NUM_CLASSES = 300       # classification: 0..299
NUM_SEG_CLASSES = 301   # segmentation:  0 = background, 1..300 = foreground


class UNetDecoderBlock(nn.Module):
    """
    Standard U-Net block: upsample the lower-resolution feature map, concatenate
    with the skip connection, then pass through two conv-BN-ReLU layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample the input by 2x (e.g. stride 32 -> stride 16).
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Convolutions after concatenating the upsampled input with the skip.
        conv_in = (in_channels // 2) + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle any spatial mismatch from odd encoder strides/padding.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        # Concatenate along the channel dimension.
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiTaskConvNeXt(nn.Module):
    """Shared ConvNeXt-V2 encoder with a classification head and U-Net seg decoder.

    The seg decoder outputs ``NUM_SEG_CLASSES`` (301) channels at full input
    resolution; the classification head outputs ``num_classes`` (300) logits.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, num_seg_classes: int = NUM_SEG_CLASSES):
        super().__init__()

        # 1. Encoder: timm ConvNeXt-V2 Atto, RANDOMLY INITIALIZED (pretrained=False
        #    -> no weight download, satisfies the no-pretrained hard rule).
        #    features_only=True makes the model return a list of stage feature maps
        #    at strides [4, 8, 16, 32] when called as self.encoder(x).
        self.encoder = timm.create_model(
            "convnextv2_atto.fcmae",
            pretrained=False,
            features_only=True,
        )

        # Read the per-stage channel dims straight from the encoder so the decoder
        # always matches the actual backbone (no hardcoded, encoder-specific dims).
        enc_channels = self.encoder.feature_info.channels()  # e.g. [40, 80, 160, 320]
        assert len(enc_channels) == 4, f"expected 4 feature stages, got {len(enc_channels)}"

        # 2. Classification head: global-average-pool the deepest feature, map to
        #    num_classes.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(enc_channels[-1], num_classes)

        # 3. U-Net decoder: progressively upsample and fuse with encoder features.
        self.up1 = UNetDecoderBlock(in_channels=enc_channels[3], skip_channels=enc_channels[2], out_channels=256)
        self.up2 = UNetDecoderBlock(in_channels=256,             skip_channels=enc_channels[1], out_channels=128)
        self.up3 = UNetDecoderBlock(in_channels=128,             skip_channels=enc_channels[0], out_channels=64)

        # 4. Final upsampling: stride 4 (deepest decoder feature) back to native
        #    resolution, then a 1x1 conv to the 301 segmentation-id logits.
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4),
            nn.Conv2d(64, num_seg_classes, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder ---
        # timm features_only model returns [f1, f2, f3, f4] at strides 4/8/16/32.
        f1, f2, f3, f4 = self.encoder(x)

        # --- Classification branch ---
        pooled = self.global_pool(f4).flatten(1)   # (B, C4)
        class_logits = self.classifier(pooled)     # (B, num_classes)

        # --- Segmentation branch ---
        d1 = self.up1(f4, f3)   # stride 16
        d2 = self.up2(d1, f2)   # stride 8
        d3 = self.up3(d2, f1)   # stride 4

        seg_logits = self.final_up(d3)             # (B, NUM_SEG_CLASSES, H, W)

        # Guard against any residual size mismatch from odd input dimensions.
        if seg_logits.shape[2:] != x.shape[2:]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        return seg_logits, class_logits


def build_model(num_classes: int = NUM_CLASSES):
    return MultiTaskConvNeXt(num_classes)


# --- Dry run / sanity check ---
if __name__ == "__main__":
    # Simulate a batch of 2 RGB images at 256x256 resolution.
    dummy_input = torch.randn(2, 3, 256, 256)

    model = MultiTaskConvNeXt()

    seg_out, cls_out = model(dummy_input)

    print(f"Input shape:           {tuple(dummy_input.shape)}")
    print(f"Segmentation output:   {tuple(seg_out.shape)}   # Expected: (2, 301, 256, 256)")
    print(f"Classification output: {tuple(cls_out.shape)}        # Expected: (2, 300)")
