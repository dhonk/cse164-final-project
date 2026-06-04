
from __future__ import annotations
from typing import Any
import logging

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.utils import PRINT_FREQ, PretrainConfigs
from .utils import adjust_learning_rate

def train_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.Device,
        epoch: int,
        loss_scaler,
        lr: float,
        configs: PretrainConfigs,
        update_freq = 1,
    ) -> dict:
    loss_sum, loss_count = 0., 0

    model.train(True)
    
    optimizer.zero_grad()
    for step, samples in enumerate(data_loader):
        if step % update_freq == 0:
            adjust_learning_rate(optimizer, step / len(data_loader) + epoch, lr, configs.min_lr, configs.warmup_epochs, configs.epochs)

        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)

        loss, _, _ = model(samples, mask_ratio=configs.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logging.error("Infinite loss, stopping training")
            raise RuntimeError("Reached infinite loss during FCMAE pretraining")
        
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % update_freq == 0)
        
        if (step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        loss_sum += loss_value

        # NOTE: do NOT reassign `lr` from optimizer.param_groups here -- `lr` is
        # the constant base/peak lr that adjust_learning_rate anneals from each
        # step. Overwriting it with the decayed group lr collapses the schedule.

        # loss_value_reduce = utils.all_redce_mean

        # tensorboard stuff

    logging.info(f"PreTrain epoch {epoch}/{configs.epochs}: Average Loss: {loss_sum / len(data_loader)}")
    return {}