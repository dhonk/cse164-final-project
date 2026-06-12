
from __future__ import annotations
from typing import Any
import logging

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.utils import PretrainConfigs
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
    ) -> dict:

    loss_sum = 0.

    model.train()
    optimizer.zero_grad()

    n_steps = len(data_loader)
    update_freq = configs.update_freq

    for step, samples in enumerate(data_loader):
        if step % update_freq == 0:
            adjust_learning_rate(optimizer, step / len(data_loader) + epoch, lr, configs.min_lr, configs.warmup_epochs, configs.epochs)

        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = model(samples, mask_ratio=configs.mask_ratio)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            logging.error("Infinite loss, stopping training")
            raise RuntimeError("Reached infinite loss during FCMAE pretraining")
        
        # flush on accumulation boundaries AND on the final step, so a partial
        # tail window (n_steps not a multiple of update_freq) still steps the
        # optimizer instead of silently dropping its gradients next epoch.
        update_grad = (step + 1) % update_freq == 0 or (step + 1) == n_steps

        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=update_grad)

        if update_grad:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        loss_sum += loss_value

        # NOTE: do NOT reassign `lr` from optimizer.param_groups here -- `lr` is
        # the constant base/peak lr that adjust_learning_rate anneals from each
        # step. Overwriting it with the decayed group lr collapses the schedule.

        # loss_value_reduce = utils.all_redce_mean

        # TODO: add the other things
        # WDYM add the other things??? What are these other things?????

    logging.info(f"PreTrain epoch {epoch + 1}/{configs.epochs}: Average Loss: {loss_sum / len(data_loader) : .6f}")
    return {"avg_loss": loss_sum / len(data_loader)}