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

from .utils import adjust_learning_rate
from ..core.utils import ClsFinetuneConfigs, PRINT_FREQ

logger = logging.getLogger(__name__)


def train_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.Device,
        epoch: int,
        lr: float,
        configs: ClsFinetuneConfigs,
        update_freq: int = 1,
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
    
    total_loss = 0
    class_correct = [0] * configs.num_classes
    class_freqs = [0] * configs.num_classes
    total_correct = 0
    total_samples = 0

    for step, (samples, labels) in enumerate(data_loader):
        if step % update_freq == 0:
            adjust_learning_rate(
                optimizer = optimizer,
                epoch = step / len(data_loader) + epoch,
                lr = lr, 
                min_lr = configs.min_lr, 
                warmup_epochs = configs.warmup_epochs,
                epochs = configs.epochs)
        samples = samples.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        # TODO: mixup function!

        # TODO: amp!

        Y = model(samples)
        loss = criterion(Y, labels)
        loss_value = loss.item()

        if not math.isfinite(loss):
            logger.warning("Loss is infinite. Issue with pipeline/parameters, stopping training.")
            raise ValueError("Loss is infinite")

        preds = Y.argmax(dim=1)
        total_loss += loss_value
        total_samples += labels.numel()
        total_correct += (preds == labels).sum().item()

        for i, _ in enumerate(class_correct):
            class_freqs[i] += (labels == i).sum().item()
            class_correct[i] += ((labels == i) & (preds == labels)).sum().item()

        # TODO: amp

        loss /= update_freq
        loss.backward()
        if (step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # TODO: ema
        
        torch.cuda.synchronize()

        # TODO: mixup
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / max(total_samples, 1)
    logging.info(f"""===== Classification Finetune epoch {epoch}/{configs.epochs}=====
                    Average Loss: {avg_loss} | Average Accuracy: {avg_acc}""")
    return {"avg_loss" : avg_loss, "avg_acc" : avg_acc}

@torch.no_grad()
def predict_one_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.Device,
    ) -> dict:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (image, label) in enumerate(data_loader, start=1):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output = model(image)
        loss = criterion(output, label)

        pred = output.argmax(dim=1)
        total_correct += (pred == label).sum().item()
        total_samples += label.numel()
        total_loss += loss.item()

        # TODO: better logging of loss
        if step % PRINT_FREQ == 0:
            print(f"Current iteration: {step}, Average Loss: {total_loss / step}")

    logging.info(f"""===== Classification Validation =====
                        Accuracy: {total_correct / len(data_loader)}, Average Loss: {total_loss / len(data_loader)}""")
    return {"accuracy" : total_correct / max(total_samples, 1),
            "loss" : total_loss / max(len(data_loader), 1)}

if __name__ == "__main__":
    from ..models.convnext.convnextv2 import convnextv2_atto
    from ..core.utils import get_device, project_root
    from ..core.dataset import TrainLabeledDataset
    from pathlib import Path

    import torchvision.transforms.v2 as v2

    dev = get_device()
    model = convnextv2_atto().to(dev)
    trans = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float32 [0,1]
    ])
    ds = TrainLabeledDataset((project_root() / "data" ).resolve(), trans)
    dloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1
    )

    predict_one_epoch(dloader, model, dev)