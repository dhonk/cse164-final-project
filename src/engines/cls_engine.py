"""
convnext_cls.py: single epoch train loop + evaluation for convnextv2 classifier
"""

# mirrors implementation of ->  https://github.com/facebookresearch/ConvNeXt-V2/engine-finetune.py
 

from __future__ import annotations
from typing import Any

import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from timm.utils.model_ema import ModelEma

from .utils import adjust_learning_rate
from ..core.utils import cls_metrics, ClsConfigs, PRINT_FREQ


logger = logging.getLogger(__name__)


def train_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        crit: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.Device,
        epoch: int,
        lr: float,
        cutmix: v2.CutMix | None,
        mixup: v2.MixUp | None,
        configs: ClsConfigs,
        model_ema: ModelEma | None = None,
    ) -> dict:
    """
    for classification fine tuning
    
    args:
        data_loader (DataLoader)
        model (nn.Module)
        device (torch.Device)
    """
    model.train()
    optimizer.zero_grad()

    n_steps = len(data_loader)
    preds = []
    gts = []
    total_loss = 0

    update_freq = configs.update_freq

    if cutmix is not None and mixup is not None:
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for step, (samples, labels) in enumerate(data_loader):
        if cutmix is not None and mixup is not None:
            # cool down: no cutmix/mixup after 95% done
            if epoch / configs.epochs < 0.95:
                samples, labels = cutmix_or_mixup(samples, labels)

        if step % update_freq == 0:
            lr = adjust_learning_rate(
                optimizer = optimizer,
                epoch = step / len(data_loader) + epoch,
                lr = lr, 
                min_lr = configs.min_lr, 
                warmup_epochs = configs.warmup_epochs,
                epochs = configs.epochs)
        samples = samples.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            Y = model(samples)
            loss = crit(Y, labels)
            loss_value = loss.item()

        if not math.isfinite(loss.detach()):
            logger.warning("Loss is infinite. Issue with pipeline/parameters, stopping training.")
            raise ValueError("Loss is infinite")

        preds.append(Y.argmax(dim=1).detach())
        gts.append(labels.detach())
        total_loss += loss_value

        loss /= update_freq
        loss.backward()
        # flush on accumulation boundaries AND on the final step, so a partial
        # tail window (n_steps not a multiple of update_freq) still steps the
        # optimizer instead of silently dropping its gradients next epoch.
        if (step + 1) % update_freq == 0 or (step + 1) == n_steps:
            optimizer.step()
            optimizer.zero_grad()

            # EMA shadow updates only on optimizer-step boundaries (mirrors the
            # ConvNeXt-V2 reference engine_finetune.py)
            if model_ema is not None:
                model_ema.update(model)
        
        torch.cuda.synchronize()
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

    if cutmix is None and mixup is None:
        avg_loss = total_loss / len(data_loader)
        preds = torch.cat(preds) if preds else torch.empty(0)
        gts = torch.cat(gts) if gts else torch.empty(0)
        avg_acc, macro_acc = cls_metrics(preds, gts)
        logging.info(f"""Classification Finetune epoch {epoch}/{configs.epochs} Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2f} | Macro Acc: {macro_acc:.2f}""")
        return {"avg_loss" : avg_loss, "avg_acc" : avg_acc, "macro_acc" : macro_acc}
    else:
        avg_loss = total_loss / len(data_loader)
        logging.info(f"""Classification Finetune epoch {epoch}/{configs.epochs} Avg Loss: {avg_loss:.6f}""")
        return {"avg_loss" : avg_loss}


@torch.no_grad()
def predict_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        crit: nn.Module,
        device: torch.Device,
        tag: str = "Validation",
    ) -> dict:
    model.eval()

    total_loss = 0.0
    preds = []
    gts = []

    for _, (image, label, _, _) in enumerate(data_loader, start=1):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image)
            loss = crit(output, label)

        preds.append(output.argmax(dim=1).detach())
        gts.append(label.detach())
        total_loss += loss.item()

    avg_loss = total_loss / max(len(data_loader), 1)
    preds = torch.cat(preds) if preds else torch.empty(0)
    gts = torch.cat(gts) if gts else torch.empty(0)
    accuracy, macro_acc = cls_metrics(preds, gts)
    logging.info(f"""{tag}: Avg Loss: {avg_loss:.6f} | Avg Acc: {accuracy:.2f} | Macro Acc: {macro_acc:.2f} | """)
    return {"loss" : avg_loss, "accuracy" : accuracy, "macro_acc" : macro_acc,}


@torch.no_grad()
def predict_one_epoch_tta(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.Device,
        tag: str = "TTA Validation",
    ) -> dict:
    """
    predict_one_epoch_tta: validate with 10-crop TTA. Mirrors predict_one_epoch but
    consumes the [B, ncrops, C, H, W] crop tensor produced by build_transform_cls_tta
    and averages softmax over the crops (same as predict_test). No loss is reported --
    averaging probabilities makes a CE loss ill-defined; non-TTA loss is tracked
    per epoch by predict_one_epoch.
    """
    model.eval()

    preds = []
    gts = []

    for _, (image, label, _, _) in enumerate(data_loader, start=1):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        B, ncrops, C, H, W = image.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(image.reshape(-1, C, H, W))

        # average softmax over the ncrops views of each image -> [B, num_classes]
        probs = logits.float().softmax(dim=1).reshape(B, ncrops, -1).mean(dim=1)
        preds.append(probs.argmax(dim=1).detach())
        gts.append(label.detach())

    preds = torch.cat(preds) if preds else torch.empty(0)
    gts = torch.cat(gts) if gts else torch.empty(0)
    accuracy, macro_acc = cls_metrics(preds, gts)
    logging.info(f"""{tag}: Avg Acc: {accuracy:.2f} | Macro Acc: {macro_acc:.2f} | """)
    return {"accuracy" : accuracy, "macro_acc" : macro_acc,}


@torch.no_grad()
def predict_test(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.Device,
) -> torch.Tensor:
    """
    predict_test: top-1 predictions with 10-crop test-time augmentation
    """
    model.eval()

    outputs = []
    length = len(data_loader)

    for step, (images, _) in enumerate(data_loader, start = 1):
        images = images.to(device, non_blocking = True)
        B, ncrops, C, H, W = images.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images.reshape(-1, C, H, W))

        # average softmax over the ncrops views of each image -> [B, num_classes]
        probs = logits.float().softmax(dim=1)
        probs = probs.reshape(B, ncrops, -1).mean(dim=1)

        outputs.append(probs.cpu())

        logging.info(f"Prediction progress: {step}/{length}")

    return torch.cat(outputs, dim = 0)