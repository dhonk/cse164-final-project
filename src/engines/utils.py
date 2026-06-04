
import math

import torch


def adjust_learning_rate(
        optimizer: torch.optim.Optimizer, 
        epoch: int | float, 
        lr: float,
        min_lr: float,
        warmup_epochs: int,
        epochs: int
        ) -> float:
    """
    adjusts learning rate

    args:
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): current epoch
        lr (float): learning rate
        min_lr (float): minimum learning rate
        epochs (int): epochs to train for
        warmup_epochs (int): epochs to warm up
    returns:
        lr (float): the updated lr
    """
    if epoch < warmup_epochs:
        new_lr = lr * epoch / warmup_epochs
    else:
        new_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = new_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = new_lr
    return lr
