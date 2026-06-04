"""
fcmae_pretrain.py: engine for fcmae single epoch

"""


from __future__ import annotations

import argparse

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core import utils
from ..core import log

from ..models.convnext.utils import MetricLogger, SmoothedValue, adjust_learning_rate


def train_one_epoch(model: nn.Module, data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.Device,
                    epoch: int, loss_scaler,
                    warmup_epochs: int = 40, epochs: int = 800, lr: float = 0, min_lr: float = 0.,
                    update_freq: int = 1, log_writer=None, mask_ratio: float = 0.6):
    model.train(True)
    metric_logger = MetricLogger(delimiter = "  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1,  fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = update_freq

    optimizer.zero_grad()
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (not per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer=optimizer, 
                                 epoch=data_iter_step / len(data_loader) + epoch, 
                                 warmup_epochs=warmup_epochs, 
                                 epochs=epochs, 
                                 lr=lr, 
                                 min_lr=min_lr)

        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)

        loss, _, _ = model(samples, mask_ratio=mask_ratio)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError("Loss is {}, stopping training".format(loss_value))
        
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()    # clear GPU cachhe at a regular interval
        
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = all_reduce_mean(loss_value) - training single GPU, no need all reduce mean
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value, head="loss", step=epoch_1000x)
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}