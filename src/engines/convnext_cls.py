"""
convnext_cls.py: single epoch train loop + evaluation for convnextv2 classifier
"""

# mirrors implementation of ->  https://github.com/facebookresearch/ConvNeXt-V2/engine-finetune.py
 

from __future__ import annotations
from typing import Any

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.utils import PRINT_FREQ


def train_one(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.Device,

    ):
    """
    for classification fine tuning
    
    args:
        data_loader (DataLoader)
        model (nn.Module)
        device (torch.Device)
    """
    ...

@torch.no_grad()
def predict_one(
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.Device,
    ) -> dict:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
    iter = 0
    correct_ct = torch.zeros(1)

    for image, label in data_loader:
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        correct_ct = correct_ct.to(device, non_blocking=True)
    
        output = model(image)
        loss = criterion(output, label)
        
        pred = torch.argmax(output)
        correct_ct += (pred == label).sum()

        torch.cuda.synchronize()
        
        total_loss += loss
        iter += 1

        # TODO: better logging of loss
        if iter % PRINT_FREQ == 0:
            print(f"Current iteration: {iter}, Average Loss: {total_loss / iter}")

    correct_ct = correct_ct[0]
    return {"accuracy" : correct_ct / len(data_loader), "loss" : total_loss / len(data_loader)}

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

    predict_one(dloader, model, dev)