"""Single-epoch FCMAE self-supervised pre-training loop (Docker-only).

FCMAE = Fully Convolutional Masked Autoencoder. We mask ~60% of patches and
reconstruct pixels (MSE on masked patches only) — self-supervised on the
competition's *own* images, which is allowed (it is "training on provided data,"
NOT a pretrained weight). The produced encoder is the artifact the cls/seg
finetuners consume via `remap_checkpoint_keys` + `load_state_dict(strict=False)`.

The FCMAE model wraps the **sparse** ConvNeXt V2 encoder, which needs
**MinkowskiEngine** — installed only in the Docker image. This engine FILE imports
nothing from Minkowski (the model is passed in), so it stays import-clean on
Windows; but actually *running* it requires the Docker env. `torch.cuda.empty_cache()`
is called each optimizer step, as the reference does, to keep the ME network's
fragmented allocator from OOMing over a long run.

Loader contract: `build_pretrain_loader` (via `ImageOnlyDataset`) yields a plain
image tensor batch `(N, 3, C, C)` — NO labels (unlike the reference's
`(samples, labels)`). `model(imgs, mask_ratio=...)` returns `(loss, pred, mask)`.

Single-GPU / no DDP / no tensorboard. Mirrors `ConvNeXt-V2/engine_pretrain.py`,
trimmed. `args` must expose `lr`, `min_lr`, `warmup_epochs`, `epochs`, and may
optionally expose `mask_ratio` (default 0.6) and `update_freq` (grad accumulation,
default 1).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from ..core.utils import MetricLogger, SmoothedValue, adjust_learning_rate


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    print_freq: int = 20,
) -> dict[str, float]:
    """Pre-train the FCMAE model for one epoch; returns averaged-stat dict.

    `data_loader` yields image-only batches `(N, 3, C, C)` (no labels). The masked
    reconstruction loss is computed inside the model. Per-iteration cosine LR via
    `adjust_learning_rate` (fractional epoch). Optional gradient accumulation over
    `args.update_freq` steps; the GPU cache is cleared on each true optimizer step
    (MinkowskiEngine allocator hygiene).
    """
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    mask_ratio = getattr(args, "mask_ratio", 0.6)
    update_freq = getattr(args, "update_freq", 1)

    optimizer.zero_grad()
    for step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per-iteration (not per-epoch) cosine LR; only on accumulation boundaries
        if step % update_freq == 0:
            adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)

        loss, _, _ = model(samples, mask_ratio=mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        loss = loss / update_freq
        update_grad = (step + 1) % update_freq == 0
        grad_norm = loss_scaler(
            loss, optimizer, parameters=model.parameters(), update_grad=update_grad
        )
        if update_grad:
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # ME network allocator hygiene over long runs

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
