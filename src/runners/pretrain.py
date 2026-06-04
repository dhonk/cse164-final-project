'''
pretrain.py: harness for FCMAE to pretrain
'''

from __future__ import annotations

import math
import numpy as np

import os
import json
from pathlib import Path

import time
import datetime

import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2

from ..engines.fcmae_pretrain import train_one_epoch
from ..models.convnext import fcmae as FCMAE
from ..models.convnext.utils import NativeScalerWithGradNormCount as NativeScaler

from ..core import dataset
from ..core import utils
from ..core.log import setup_logging, get_logger
from ..core.utils import LOG_DIR, SAVE_DIR, CHECKPOINT_DIR, CHKPT_FREQ, DATA_DIR, RAND_SEED, BATCH_SIZE
from ..core.optim import get_parameter_groups

def runner(epochs: int = 800, warmup_epochs: int = 40, 
           input_size: int = 224, mask_ratio: float = 0.6,
           decoder_depth: int = 1, weight_decay: float = 0.05,
           resume: str = '', start_epoch: int = 0, 
           num_workers: int = 2, pin_mem: bool = True, lr: float = 0):
           
    run_device = utils.get_device()
    
    setup_logging()
    log = get_logger('logs.txt')

    # fix seed for reproduce
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = v2.Compose([
        v2.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_50k = dataset.TrainUnlabeledSet(Path(DATA_DIR), transform=transform_train)
    dataset_cls = dataset.TrainLabeledDataset(Path(DATA_DIR), transform=transform_train)
    dataset_seg = dataset.TrainMaskedDataset(Path(DATA_DIR), transform=transform_train)

    dataset_train = dataset.PretrainDataset(Path(DATA_DIR), transform=transform_train, datasets=[dataset_50k, dataset_cls, dataset_seg])
    print(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = FCMAE.convnextv2_atto(
        mask_ratio=mask_ratio,
        decoder_depth=decoder_depth,
    )
    model.to(device=run_device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    eff_batch_size = BATCH_SIZE * 1 * 1
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    blr = 1.5e-4
    lr = blr * eff_batch_size / 256

    print("base lr: %.2e" % (0 * 256 / eff_batch_size))
    print("actual lr: %.2e" % 0)

    print("accumulate grad iterations: %d" % 1)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = get_parameter_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, run_device, epoch, loss_scaler,)
        if CHECKPOINT_DIR:
            if (epoch + 1) % CHKPT_FREQ == 0 or epoch + 1 == epochs:
                utils.save_checkpoint(path=CHECKPOINT_DIR, model=model, optimizer=optimizer, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        

        with open(os.path.join(LOG_DIR, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))