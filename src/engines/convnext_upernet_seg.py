"""Single-epoch training loop for ConvNeXt V2 + UPerNet semantic segmentation.

Seg path (see CLAUDE.md Phase 2.3 — we drive the decoder directly, NOT through
the port's `SegmentationModule`):

    feats  = backbone.forward_features_seg(img)   # 4-tuple, strides 4/8/16/32
    logits = decoder(feats)                        # log_softmax at stride-4
    logits = upsample(logits, crop_size)           # bilinear, back to input res
    loss   = NLLLoss(ignore_index=1000)(logits, seg_id)

The decoder emits `log_softmax` (built with `use_softmax=False`), so the matching
criterion is `nn.NLLLoss`. Ignore id is **1000** (NOT 255) — GT `1000` pixels are
unscored, so they must be ignored in the loss too.

Single-GPU / no DDP. AMP via `core.utils.NativeScaler`. Per-iteration LR schedule
via `core.utils.adjust_learning_rate` (fractional epoch). Mirrors the reference
`ConvNeXt-V2/engine_finetune.py` structure, trimmed to our seg task.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils import IGNORE_ID, MetricLogger, SmoothedValue, adjust_learning_rate


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of correctly-classified pixels, excluding ignore (1000) pixels."""
    preds = logits.argmax(dim=1)
    valid = target != IGNORE_ID
    total = int(valid.sum())
    if total == 0:
        return 0.0
    correct = int((preds[valid] == target[valid]).sum())
    return correct / total


def seg_forward(backbone, decoder, samples: torch.Tensor) -> torch.Tensor:
    """Backbone seg features -> UPerNet decoder -> upsample to input resolution.

    Returns `(N, num_class, H, W)` log-probabilities at the input's spatial size.
    """
    feats = backbone.forward_features_seg(samples)
    logits = decoder(feats)  # log_softmax at stride-4
    return F.interpolate(logits, size=samples.shape[-2:], mode="bilinear", align_corners=False)


def train_one_epoch(
    backbone: nn.Module,
    decoder: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    max_norm: float | None = None,
    use_amp: bool = True,
    print_freq: int = 20,
) -> dict[str, float]:
    """Train backbone+decoder for one epoch; returns averaged-stat dict.

    `data_loader` yields `(image, label, seg_id, name)` (from `TrainSegDataset`);
    only `image` and `seg_id` are used here. `args` must expose `lr`, `min_lr`,
    `warmup_epochs`, `epochs` for the per-iteration cosine schedule.
    """
    backbone.train(True)
    decoder.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    params = list(backbone.parameters()) + list(decoder.parameters())

    for step, (samples, _label, seg_id, _name) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per-iteration (not per-epoch) LR schedule; fractional epoch
        adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        target = seg_id.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = seg_forward(backbone, decoder, samples)
                loss = criterion(logits, target)
        else:
            logits = seg_forward(backbone, decoder, samples)
            loss = criterion(logits, target)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad()
        if use_amp:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=params)
        else:
            loss.backward()
            if max_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
            else:
                grad_norm = None
            optimizer.step()

        acc = pixel_accuracy(logits, target)
        metric_logger.update(loss=loss_value, pixel_acc=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
