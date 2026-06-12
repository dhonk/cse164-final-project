"""
segtrain.py: harness for segmentation fine tune & inference

Mirrors src/clstrain.py 1:1 in structure (the established cls finetune driver).
Currently bare stubs -- skeleton + signatures only, to be filled in later.

TODO (later, non-stub pass):
  - joint image+mask transforms (v2 transforms both together)
  - FCMAE warm-start (load_checkpoint, strict=False) from checkpoint-pretrain-*.pth
  - mIoU / boundary-F curves in plot_stats
  - extend SegConfigs with run() fields (batch_size, limit, num_workers, etc.)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import logging
import os
import time

from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from timm.data.transforms_factory import create_transform
from timm.utils.model_ema import ModelEma

from .core.utils import *
from .core import dataset as ds

from .models import upernet

from .engines.seg_engine import train_one_epoch, predict_one_epoch

from .utils import build_param_groups_lrd, save_checkpoint, load_checkpoint, LossScaler, DiceCELoss


logger = logging.getLogger(__name__)


def build_transform_train(config: SegConfigs) -> v2.Transform:
    """build transform for seg fine tuning (must transform image AND mask jointly)."""
    return ds.build_transform_seg_train(config)


def build_transform_eval(config: SegConfigs) -> v2.Transform:
    return ds.build_transform_seg_eval(config)


def build_model(config: SegConfigs) -> upernet.ConvNeXt_UPerNet:
    match config.model_size:
        case "atto":
            builder = upernet.atto
        case "femto":
            builder = upernet.femto
        case "pico":
            builder = upernet.pico
        case "nano":
            builder = upernet.nano
        case "tiny":
            builder = upernet.tiny
        case "base":
            builder = upernet.base
        case "large":
            builder = upernet.large
        case "huge":
            builder = upernet.huge
    # NOTE: seg head needs 301 channels (ids 0..300); the upernet factories default
    # num_classes=301. SegConfigs.num_classes is still 300 -- pass +1 here until
    # SegConfigs is fixed in the later pass.
    return builder(
        in_channels=config.channels,
        num_classes=config.num_classes + 1,
    )


def plot_stats(train_stats: list[dict], val_stats: list[dict], save_path: Path,
               val_epochs: list[int] | None = None):
    """Render loss + (mIoU, boundary-F) panels and save the figure."""

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Segementation Fine-Tuning")

    train_x = list(range(len(train_stats)))
    # validation is sparse (every val_every epochs); plot it at its true epochs
    val_x = val_epochs if val_epochs is not None else list(range(len(val_stats)))

    # EMA shadow curves are only present when EMA was enabled (0 < config.ema < 1)
    has_ema = bool(val_stats) and "ema_mIoU" in val_stats[0]

    # panel 1: losses
    ax_loss.plot(train_x, [s["avg_loss"] for s in train_stats], "-o", label="train", markersize=3)
    ax_loss.plot(val_x, [s["loss"] for s in val_stats], "-o", label="val", markersize=3)
    if has_ema:
        ax_loss.plot(val_x, [s["ema_loss"] for s in val_stats], "-o", label="EMA val", markersize=3)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # panel 2: accuracies (val accuracy + val macro accuracy; train if available)
    ax_acc.plot(val_x, [s["mIoU"] for s in val_stats], "-o", label="val mIoU", markersize=3)
    ax_acc.plot(val_x, [s["boundary_f"] for s in val_stats], "-o", label="val boundary F score", markersize=3)
    if has_ema:
        ax_acc.plot(val_x, [s["ema_mIoU"] for s in val_stats], "-o", label="EMA val mIoU", markersize=3)
        ax_acc.plot(val_x, [s["ema_boundary_f"] for s in val_stats], "-o", label="EMA val boundary F score", markersize=3)
    ax_acc.set_title("mIoU, Boundary F Score")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("mIoU/F Score")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)


def save_stats_csv(stats: list[dict], path: Path, epochs: list[int] | None = None):
    """Write a list-of-dicts stats log to CSV, one row per epoch.

    `epochs` overrides the epoch column (val stats are sparse, every val_every epochs);
    defaults to a dense 0..N-1 range when omitted.
    """
    if not stats:
        return
    if epochs is None:
        epochs = list(range(len(stats)))
    fieldnames = ["epoch", *stats[0].keys()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, row in zip(epochs, stats):
            writer.writerow({"epoch": epoch, **row})
    logger.info("Saved stats to %s", path)


def run(config: SegConfigs):
    # env setup
    seed_everything(RAND_SEED)
    device = get_device()
    cudnn.benchmark = True

    # logs
    logger.info("Segmentation Fine Tuning on %s | model size - %s", device, config.model_size)
    logger.info("Hyperparams: epochs - %d | batch size - %d", config.epochs, config.batch_size)

    # build datasets & dataloader
    train_transform = build_transform_train(config)
    val_transform = build_transform_eval(config)
    train_dataset = ds.TrainMaskedDataset(DATA_DIR, train_transform)
    val_dataset = ds.ValDataset(DATA_DIR, val_transform)

    if config.limit > 0:
        train_dataset.items = train_dataset.items[: config.limit]
    logger.info("Cls Finetune pool: %d images", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size = max(config.batch_size, config.limit) if config.limit > 0 else config.batch_size,
        shuffle = True,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True,
    )

    # Full-res variable-size val masks crash Windows workers (WinError 1455 / cudaErrorAlreadyMapped):
    # run val in-process, no pinning. (persistent_workers requires num_workers>0.)
    val_loader = DataLoader(
        val_dataset,
        batch_size = max(config.batch_size, config.limit) if config.limit > 0 else config.batch_size,
        shuffle = False,
        num_workers = 0,
        pin_memory = False,
        drop_last = False,
        persistent_workers = False,
        collate_fn = ds.cls_val_collate,
    )

    # model
    model = build_model(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_params / 1e6)

    # load pretrained params
    ckpts = list(Path(CHECKPOINT_DIR).glob("checkpoint-pretrain-*.pth"))
    latest_ckpt = max(ckpts, key = os.path.getmtime)
    
    load_results = load_checkpoint(model, latest_ckpt)
    
    logger.info("Loaded checkpoint: %s (epoch %d)", latest_ckpt, load_results.get("epoch", -1))
    logger.info("Loaded: %d | Unexpected: %d | Randomized (not found in ckpt): %d",
                load_results.get("loaded", -1),
                load_results.get("unexpected", -1),
                load_results.get("random", -1))

    # EMA shadow weights — build AFTER warm-start so the shadow starts from the
    # warm-started weights (not random). Enabled when 0 < config.ema < 1.
    model_ema = None
    if 0 < config.ema < 1:
        model_ema = ModelEma(model, decay=config.ema, device="", resume="")
        logger.info("Using EMA with decay %.5f", config.ema)

    # lr
    eff_batch_size = config.batch_size * config.update_freq
    lr = config.blr * eff_batch_size / 256
    logger.info("base lr - %.2e | eff batch size - %d | real lr - %.2e",
                config.blr, eff_batch_size, lr)
    
    # loss
    crit = DiceCELoss(num_classes=config.num_classes + 1, ignore_index=IGNORE_IDX,
                      label_smoothing=config.label_smoothing, dice_weight=config.dice_weight)

    # optim
    param_groups = build_param_groups_lrd(model, config.weight_decay, config.layer_wise_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=config.optim_momentum)
    loss_scaler = LossScaler(config.use_amp, "cuda" if check_cuda() else "")

    # cutmix/mixup
    '''
    if config.cutmix > 0:
        cutmix = v2.CutMix(alpha=config.cutmix, num_classes=NUM_CLASSES)
    else:
        cutmix = None
    if config.mixup > 0:
        mixup = v2.MixUp(alpha=config.mixup, num_classes=NUM_CLASSES)
    else:
        mixup = None
    '''

    # train loop
    ckpt_dir = Path(CHECKPOINT_DIR)
    start_time = time.time()

    # accumulate stats
    train_stats = []
    val_stats = []
    val_epochs = []  # which epochs val_stats correspond to (validation runs every val_every)

    val_every = 5  # run validation every N epochs (always also validates the final epoch)

    for epoch in range(config.epochs):
        # train an epoch
        train_stats.append(train_one_epoch(
            data_loader=train_loader,
            model=model,
            crit=crit,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            lr=lr,
            configs=config,
            model_ema=model_ema,
        ))

        # validate every val_every epochs (and always on the final epoch)
        is_last = epoch + 1 == config.epochs
        if (epoch + 1) % val_every == 0 or is_last:
            # validate an epoch - the engine should handle everything...?
            val_stats.append(predict_one_epoch(
                data_loader=val_loader,
                model=model,
                crit=crit,
                device=device,
                configs=config,
            ))
            val_epochs.append(epoch)

            # also validate the EMA shadow so we can compare it against the raw model
            # (merged into the same epoch's val dict under ema_* keys)
            if model_ema is not None:
                ema_stats = predict_one_epoch(
                    data_loader=val_loader,
                    model=model_ema.ema,
                    crit=crit,
                    device=device,
                    configs=config,
                    tag="EMA Validation",
                )
                val_stats[-1].update({f"ema_{k}": v for k, v in ema_stats.items()})

        # checkpoint saving
        if (epoch + 1) % config.save_every == 0 or is_last:
            ckpt_path = ckpt_dir / f"checkpoint-segfinetune-{epoch + 1}.pth"
            save_checkpoint(
                model, ckpt_path,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch = epoch, config = config,
                model_ema = model_ema,
            )
            logger.info("Saved checkpoint to %s", ckpt_path)
        
    # write accumulated stats + final training curves to outputs/ (timestamped
    # so separate runs don't clobber each other)
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_stats_csv(train_stats, out_dir / f"segtrain-train-{stamp}.csv")
    save_stats_csv(val_stats, out_dir / f"segtrain-val-{stamp}.csv", epochs=val_epochs)
    plot_stats(train_stats, val_stats, out_dir / f"segtrain-curves-{stamp}.png", val_epochs=val_epochs)

    total_time = datetime.timedelta(seconds = int(time.time()) - start_time)
    logger.info("SEG fine-tuning done in %s", total_time)


def get_args_parser() -> argparse.ArgumentParser:
    """CLI for python -m src.segtrain. Unset flags fall back to SegConfigs defaults."""
    parser = argparse.ArgumentParser("ConvNeXt-V2 + UPerNet seg fine-tuning", add_help=True)
    return parser


def main():
    setup_logging()

    # too lazy to set up proper configs, change here
    config = SegConfigs()

    run(config)


if __name__ == "__main__":
    main()
