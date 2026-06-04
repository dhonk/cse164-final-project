"""
pretrain.py: harness for FCMAE self-supervised pre-training.

Pools the three TRAINING splits (unlabeled 50k + labeled 7.5k + seg 3k, all
labels/masks dropped) into one image-only set and trains a fully convolutional
masked autoencoder on it. The saved checkpoint's `model` state_dict is what the
downstream cls/seg finetune consumes (encoder keys remapped, strict=False).

Full runs are Docker-only: FCMAE -> SparseConvNeXtV2 imports spconv, which is not
pip-installable on Windows.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2

from ..core import dataset, utils
from ..core.utils import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHKPT_FREQ,
    DATA_DIR,
    RAND_SEED,
    PretrainConfigs,
)
from ..core.dataset import NORM_MEAN, NORM_STD
from ..engines.fcmae_pretrain import train_one_epoch
from ..models.convnext import fcmae as FCMAE
from .utils import LossScaler

log = logging.getLogger(__name__)


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """
    Split parameters into weight-decay / no-decay groups.

    Biases and 1-D params (norm weights, GRN gamma/beta) are excluded from weight
    decay -- the standard MAE recipe. Replaces the old core.optim.get_parameter_groups.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: LossScaler,
    epoch: int,
    configs: PretrainConfigs,
    tag: str = "fcmae_atto",
) -> None:
    """
    Save the FCMAE training state. The `model` state_dict is the artifact finetune
    loads (encoder keys remapped with strict=False). Writes a per-epoch checkpoint
    plus a rolling `<tag>_last.pth`.
    """
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "configs": dataclasses.asdict(configs),
    }
    torch.save(payload, ckpt_dir / f"{tag}_ep{epoch}.pth")
    torch.save(payload, ckpt_dir / f"{tag}_last.pth")
    log.info("Saved checkpoint %s at epoch %d", tag, epoch)


def runner(
    epochs: int = 800,
    warmup_epochs: int = 40,
    input_size: int = 224,
    mask_ratio: float = 0.6,
    decoder_depth: int = 1,
    weight_decay: float = 0.05,
    blr: float = 1.5e-4,
    min_lr: float = 1e-6,
    resume: str = "",
    start_epoch: int = 0,
    num_workers: int = 2,
    limit: int = 0,
) -> None:
    """
    Drive FCMAE pre-training for `epochs` epochs over the pooled training images.

    args:
        epochs:        total epochs
        warmup_epochs: linear-warmup epochs before cosine decay
        input_size:    crop size fed to the encoder
        mask_ratio:    fraction of patches masked each step
        decoder_depth: FCMAE decoder block count
        weight_decay:  AdamW weight decay (norm/bias params excluded)
        blr:           base lr; actual lr = blr * eff_batch_size / 256
        min_lr:        cosine floor
        resume:        checkpoint path to resume from (TODO: not yet wired)
        start_epoch:   epoch to start counting from
        num_workers:   DataLoader workers
        limit:         if > 0, use only the first `limit` pooled images (smoke runs)
    """
    device = utils.get_device()
    utils.seed_everything(RAND_SEED)
    cudnn.benchmark = True

    # --- data ---------------------------------------------------------------
    # Plain RandomResizedCrop + hflip; project's own (0.5,...) stats, no external data.
    transform_train = v2.Compose([
        v2.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD)),
    ])

    data_root = Path(DATA_DIR)
    dataset_train = dataset.PretrainDataset(
        data_root,
        transform=transform_train,
        datasets=[
            dataset.TrainLabeledDataset(data_root, transform=transform_train),
            dataset.TrainMaskedDataset(data_root, transform=transform_train),
        ],
    )
    if limit > 0:
        dataset_train.items = dataset_train.items[:limit]
    log.info("Pretrain pool: %d images", len(dataset_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- model --------------------------------------------------------------
    model = FCMAE.convnextv2_atto(mask_ratio=mask_ratio, decoder_depth=decoder_depth)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("FCMAE convnextv2_atto: %d trainable params", n_params)

    # --- optimizer / scaler / configs --------------------------------------
    eff_batch_size = BATCH_SIZE
    lr = blr * eff_batch_size / 256
    log.info("base lr %.2e -> actual lr %.2e (eff batch %d)", blr, lr, eff_batch_size)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, weight_decay), lr=lr, betas=(0.9, 0.95)
    )
    loss_scaler = LossScaler(use_amp=True, device=str(device))
    configs = PretrainConfigs(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        mask_ratio=mask_ratio,
    )

    # TODO: wire up `resume` (load model/optimizer/scaler/start_epoch from ckpt).
    if resume:
        log.warning("resume='%s' requested but resume is not implemented yet", resume)

    # --- train loop ---------------------------------------------------------
    log.info("Start FCMAE pre-training for %d epochs", epochs)
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, lr, configs
        )
        if (epoch + 1) % CHKPT_FREQ == 0 or epoch + 1 == epochs:
            save_checkpoint(model, optimizer, loss_scaler, epoch, configs)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    log.info("Pre-training done in %s", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="FCMAE self-supervised pre-training")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--decoder-depth", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--blr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N pooled images (smoke runs)")
    args = parser.parse_args()

    runner(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        input_size=args.input_size,
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        weight_decay=args.weight_decay,
        blr=args.blr,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
