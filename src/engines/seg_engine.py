"""
seg_engine.py: single epoch train loop + evaluation for convnext-v2 + upernet segmenter

Mirrors engines/cls_engine.py and engines/pretrain_engine.py: the engine owns all
single-epoch logic. The seg head emits RAW logits at stride-4 feature resolution
(see cooking/SEG_PIPELINE_PATTERNS.md section 4) -- this engine handles bilinear
upsampling to the target resolution, CrossEntropyLoss (ignore_index = IGNORE_IDX),
and argmax to seg ids. No -1 offset: the 301 channels map to seg ids 0..300 directly
(background = channel 0).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.utils.model_ema import ModelEma

from .utils import adjust_learning_rate
from ..core.utils import (
    SegConfigs,
    mask_confusion_matrix,
    iou_from_confusion,
    boundary_f_score,
)


logger = logging.getLogger(__name__)


def train_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        crit: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.Device,
        epoch: int,
        lr: float,
        configs: SegConfigs,
        model_ema: ModelEma | None = None,
    ) -> dict:
    """
    for segmentation fine tuning

    args:
        data_loader (DataLoader): yields (samples, masks, labels) from TrainMaskedDataset
        model (nn.Module): ConvNeXt_UPerNet emitting raw logits [B, 301, h, w]
        crit (nn.Module): nn.CrossEntropyLoss(ignore_index = IGNORE_IDX)
        optimizer (torch.optim.Optimizer)
        device (torch.Device)
        epoch (int): current epoch
        lr (float): base/peak lr that adjust_learning_rate anneals from
        configs (SegConfigs)
        model_ema (ModelEma | None): if given, its shadow weights are updated on
            every optimizer-step boundary (mirrors cls_engine.train_one_epoch)
    """
    model.train()
    optimizer.zero_grad()

    n_steps = len(data_loader)
    update_freq = configs.update_freq
    # accumulate loss on-GPU; sync once at epoch end (avoids a per-step .item() stall)
    total_loss = torch.zeros((), device=device)

    # `labels` is the whole-image seg id; unused here (kept for signature parity /
    # a future deep-supervision aux head).
    for step, (samples, masks, labels) in enumerate(data_loader):
        if step % update_freq == 0:
            lr = adjust_learning_rate(
                optimizer = optimizer,
                epoch = step / n_steps + epoch,
                lr = lr,
                min_lr = configs.min_lr,
                warmup_epochs = configs.warmup_epochs,
                epochs = configs.epochs)

        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(samples)
            # upsample stride-4 logits to the mask resolution before the loss
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = crit(logits, masks)

        # accumulate on-GPU (detached) -- no sync here; finite check deferred to epoch end
        total_loss += loss.detach()

        loss /= update_freq
        loss.backward()
        # flush on accumulation boundaries AND on the final step, so a partial
        # tail window (n_steps not a multiple of update_freq) still steps the
        # optimizer instead of silently dropping its gradients next epoch.
        if (step + 1) % update_freq == 0 or (step + 1) == n_steps:
            optimizer.step()
            optimizer.zero_grad()

            # EMA shadow updates only on optimizer-step boundaries (mirrors the
            # ConvNeXt-V2 reference engine_finetune.py / cls_engine.train_one_epoch)
            if model_ema is not None:
                model_ema.update(model)

    # single sync for the whole epoch (replaces the per-step .item())
    avg_loss = (total_loss / max(n_steps, 1)).item()
    if not math.isfinite(avg_loss):
        logger.warning("Loss is infinite. Issue with pipeline/parameters, stopping training.")
        raise ValueError("Loss is infinite")
    logging.info(f"Seg Finetune epoch {epoch}/{configs.epochs} Avg Loss: {avg_loss:.6f}")
    return {"avg_loss": avg_loss}


@torch.no_grad()
def predict_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        crit: nn.Module,
        device: torch.Device,
        configs: SegConfigs,
        tag: str = "Validation",
    ) -> dict:
    """
    validation over ValDataset (batched with core.dataset.cls_val_collate ->
    (images, labels, seg_masks, orig_sizes)): images are stacked at the transform
    resolution; seg_masks is a list of original-size LongTensors (variable H x W);
    orig_sizes is unused here -- each mask already carries its true H x W.

    Reports foreground-only mIoU (class ids 1..300, background id 0 excluded per
    CLAUDE.md), mean boundary-F-score, and average CE loss.
    """
    model.eval()

    num_classes = configs.num_classes
    hist = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    boundary_scores: list[float] = []
    total_loss = 0.0
    n = 0

    for images, _, seg_masks, _ in data_loader:
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            # hflip TTA: average logits of the image and its un-flipped mirror.
            logits = (logits + torch.flip(model(torch.flip(images, dims=[3])), dims=[3])) / 2

        # per-sample: val masks are at their own original (variable) size, so we
        # interpolate each image's logits to that mask's H x W independently.
        for b in range(len(seg_masks)):
            H, W = seg_masks[b].shape
            lg = F.interpolate(logits[b:b + 1], size=(H, W),
                               mode="bilinear", align_corners=False)
            gt = seg_masks[b].to(device).long()

            total_loss += crit(lg, gt.unsqueeze(0)).item()
            n += 1

            pred = lg.argmax(1)[0].cpu().numpy()   # seg-id map 0..300
            gt_np = seg_masks[b].numpy()

            hist += mask_confusion_matrix(pred, gt_np, num_classes)
            boundary_scores.append(boundary_f_score(pred, gt_np))

    # foreground-only mIoU: class ids 1..300 (background id 0 excluded)
    miou, _ = iou_from_confusion(hist, list(range(1, num_classes + 1)))
    boundary_f = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    avg_loss = total_loss / max(n, 1)

    logging.info(f"{tag}: Avg Loss: {avg_loss:.6f} | mIoU: {miou:.4f} | Boundary-F: {boundary_f:.4f}")
    # TODO: rare-class mIoU once a rare-class set is defined.
    return {"loss": avg_loss, "mIoU": miou, "boundary_f": boundary_f}


@torch.no_grad()
def predict_test(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.Device,
    ) -> list[np.ndarray]:
    """
    predict_test: per-image seg-id maps for submission.csv.

    TestDataset yields (image, orig_size); default collate gives (images, orig_sizes)
    where orig_size is PIL (W, H). Returns a list (not a stacked tensor) because the
    id-maps are at variable original resolutions. finalres.run_seg feeds each map to
    core.utils.encode_mask_ids (which emits "" for all-background).
    """
    model.eval()

    results: list[np.ndarray] = []
    length = len(data_loader)

    for step, (images, orig_sizes) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # hflip TTA: average softmax probs of the image and its mirror (the
            # flipped logits are un-flipped along W before accumulating).
            probs = model(images).float().softmax(1)
            probs += torch.flip(model(torch.flip(images, dims=[3])), dims=[3]).float().softmax(1)

        for b in range(images.shape[0]):
            W, H = int(orig_sizes[b][0]), int(orig_sizes[b][1])
            # bilinear-upsample probs to original H x W, THEN argmax -> the id-map
            # itself lands at the exact original size (no further resize needed).
            lg = F.interpolate(probs[b:b + 1], size=(H, W),
                               mode="bilinear", align_corners=False)
            pred = lg.argmax(1)[0].to(torch.int32).cpu().numpy().astype(np.uint16)
            results.append(pred)

        logging.info(f"Seg prediction progress: {step}/{length}")

    # TODO: multi-scale TTA (accumulate probs across short-side scales) -- SEG_PIPELINE_PATTERNS.md section 5
    return results
