"""Single-epoch train loop + evaluation for ConvNeXt V2 image classification.

Classification path (the backbone's own `forward`, no extra head wiring):

    logits = backbone(img)                 # forward_features -> GAP+LayerNorm -> head
    loss   = CrossEntropyLoss()(logits, class_id)

`backbone(x)` already does `forward_features` (global-average-pool + final
`nn.LayerNorm`) then the linear `head` (build the backbone with
`num_classes=300`), so logits are `(N, 300)` and `class_id` is `0..299` — no
off-by-one here (that only bites segmentation, where seg_id = class_id + 1).

Single-GPU / no DDP. AMP via `core.utils.NativeScaler`. Per-iteration cosine LR
via `core.utils.adjust_learning_rate` (fractional epoch). Mirrors the seg engine
(`convnext_upernet_seg.py`) and the reference `ConvNeXt-V2/engine_finetune.py`.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from ..core.utils import MetricLogger, SmoothedValue, NUM_CLASSES, adjust_learning_rate


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of correctly-classified images (plain top-1)."""
    preds = logits.argmax(dim=1)
    total = target.numel()
    if total == 0:
        return 0.0
    return int((preds == target).sum()) / total


def train_one_epoch(
    model: nn.Module,
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
    """Train the classifier backbone for one epoch; returns averaged-stat dict.

    `data_loader` yields `(image, label, name)` (from `TrainLabeledDataset`);
    only `image` and `label` are used. `args` must expose `lr`, `min_lr`,
    `warmup_epochs`, `epochs` for the per-iteration cosine schedule.
    """
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    params = list(model.parameters())

    for step, (samples, label, _name) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per-iteration (not per-epoch) LR schedule; fractional epoch
        adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        target = label.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(samples)
                loss = criterion(logits, target)
        else:
            logits = model(samples)
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

        acc = accuracy_top1(logits, target)
        metric_logger.update(loss=loss_value, acc1=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    use_amp: bool = True,
    num_classes: int = NUM_CLASSES,
    print_freq: int = 50,
) -> dict[str, float]:
    """Evaluate top-1 and macro (class-balanced) accuracy on a labeled split.

    `data_loader` may yield either `(image, label, name)` (TrainLabeledDataset)
    or `(image, label, seg_id, name, orig_size)` (ValDataset) — we only read the
    first two elements, so both contracts work.

    Macro accuracy = mean per-class recall (each class weighted equally), which
    is what the leaderboard's classification metric rewards (class-balanced), so
    don't rely on plain top-1 for model selection.
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # per-class tallies for macro accuracy
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples, label = batch[0], batch[1]
        samples = samples.to(device, non_blocking=True)
        target = label.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(samples)
        else:
            logits = model(samples)

        preds = logits.argmax(dim=1)
        acc = accuracy_top1(logits, target)
        metric_logger.update(acc1=acc)

        t = target.cpu()
        hit = (preds.cpu() == t)
        total_per_class.index_add_(0, t, torch.ones_like(t))
        correct_per_class.index_add_(0, t, hit.long())

    seen = total_per_class > 0
    if int(seen.sum()) == 0:
        macro_acc = 0.0
    else:
        per_class_acc = correct_per_class[seen].float() / total_per_class[seen].float()
        macro_acc = float(per_class_acc.mean())

    print(f"* Acc@1 {metric_logger.acc1.global_avg:.4f}  MacroAcc {macro_acc:.4f}")
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["macro_acc"] = macro_acc
    return stats
