"""
classifier..py: harness for classification fine tune & inference

TODO: try just see how it does for now, if results not strong implement
mixup, TTA
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
matplotlib.use("Agg")  # headless: just save the figure to disk, no display
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

from .models import convnextv2

from .engines.cls_engine import train_one_epoch, predict_one_epoch, predict_one_epoch_tta

from .utils import build_param_groups_lrd, save_checkpoint, load_checkpoint, LossScaler


logger = logging.getLogger(__name__)

# TODO: make for training data
def build_transform_train(config: ClsConfigs) -> v2.Transform:
    """
    build_transform: build transform for cls fine tuning 
    
    from paper:
        RandAug(9, 0.5)

    """
    m, mstd = config.randaug_params
    aa = f"rand-m{int(m)}-mstd{mstd}-inc1"
    
    transform = create_transform(
        input_size = config.size,
        is_training = True,
        color_jitter = config.color_jitter,
        auto_augment = aa,
        interpolation = "bicubic",
        re_prob = config.re_prob,
        re_mode = config.re_mode,
        re_count = config.re_count,
        mean = ds.NORM_MEAN, 
        std = ds.NORM_STD,
    )

    return cast(v2.Transform, transform)


def build_transform_eval(config: ClsConfigs) -> v2.Transform:
    return v2.Compose([
        v2.Resize(
            size = (256), 
            interpolation = v2.InterpolationMode.BICUBIC
        ),
        v2.CenterCrop(size=config.size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=ds.NORM_MEAN, std=ds.NORM_STD),
    ])

def build_model(config: ClsConfigs) -> convnextv2.ConvNeXtV2:
    match config.model_size:
        case "atto":
            builder = convnextv2.atto
        case "femto":
            builder = convnextv2.femto
        case "pico":
            builder = convnextv2.pico
        case "nano":
            builder = convnextv2.nano
        case "tiny":
            builder = convnextv2.tiny
        case "base":
            builder = convnextv2.base
        case "large":
            builder = convnextv2.large
        case "huge":
            builder = convnextv2.huge
    return builder(
        in_channels=config.channels,
        num_classes=config.num_classes,
        drop_path_rate=config.drop_path,
        head_init_scale=config.head_init,
    )
            

def plot_stats(train_stats: list[dict], val_stats: list[dict], save_path: Path):
    """
    Render two side-by-side panels and save the figure:
      1. Loss     — train loss vs val loss
      2. Accuracy — val accuracy vs val macro accuracy

    Stat keys (see cls_engine): train_one_epoch -> avg_loss (+ avg_acc/macro_acc
    ONLY when cutmix/mixup are disabled); predict_one_epoch -> loss/accuracy/macro_acc.
    Train accuracy lines are added to panel 2 only if those keys are present.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Classification Fine-Tuning")

    train_x = list(range(len(train_stats)))
    val_x = list(range(len(val_stats)))

    # panel 1: losses
    ax_loss.plot(train_x, [s["avg_loss"] for s in train_stats], "-o", label="train", markersize=3)
    ax_loss.plot(val_x, [s["loss"] for s in val_stats], "-o", label="val", markersize=3)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # panel 2: accuracies (val accuracy + val macro accuracy; train if available)
    ax_acc.plot(val_x, [s["accuracy"] for s in val_stats], "-o", label="val accuracy", markersize=3)
    ax_acc.plot(val_x, [s["macro_acc"] for s in val_stats], "-o", label="val macro accuracy", markersize=3)
    if val_stats and "ema_accuracy" in val_stats[0]:
        ax_acc.plot(val_x, [s["ema_accuracy"] for s in val_stats], "-o", label="EMA val accuracy", markersize=3)
        ax_acc.plot(val_x, [s["ema_macro_acc"] for s in val_stats], "-o", label="EMA val macro accuracy", markersize=3)
    if train_stats and "avg_acc" in train_stats[0]:
        ax_acc.plot(train_x, [s["avg_acc"] for s in train_stats], "-o", label="train accuracy", markersize=3)
        ax_acc.plot(train_x, [s["macro_acc"] for s in train_stats], "-o", label="train macro accuracy", markersize=3)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)


def save_stats_csv(stats: list[dict], path: Path):
    """Write a list-of-dicts stats log to CSV, one row per epoch."""
    if not stats:
        return
    fieldnames = ["epoch", *stats[0].keys()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, row in enumerate(stats):
            writer.writerow({"epoch": epoch, **row})
    logger.info("Saved stats to %s", path)


def run(config: ClsConfigs):
    # env setup
    seed_everything(RAND_SEED)
    device = get_device()
    cudnn.benchmark = True

    # logs
    logger.info("Classification Fine Tuning on %s | model size - %s", device, config.model_size)
    logger.info("Hyperparams: epochs - %d | batch size - %d", config.epochs, config.batch_size)

    # build datasets & dataloader
    train_transform = build_transform_train(config)
    val_transform = build_transform_eval(config)
    train_dataset = ds.TrainLabeledDataset(DATA_DIR, train_transform)
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

    val_loader = DataLoader(
        val_dataset,
        batch_size = max(config.batch_size, config.limit) if config.limit > 0 else config.batch_size,
        shuffle = True,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
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
    crit = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # optim
    param_groups = build_param_groups_lrd(model, config.weight_decay, config.layer_wise_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=config.optim_momentum)
    loss_scaler = LossScaler(config.use_amp, "cuda" if check_cuda() else "")

    # cutmix/mixup
    if config.cutmix > 0:
        cutmix = v2.CutMix(alpha=config.cutmix, num_classes=NUM_CLASSES)
    else:
        cutmix = None
    if config.mixup > 0:
        mixup = v2.MixUp(alpha=config.mixup, num_classes=NUM_CLASSES)
    else:
        mixup = None

    # train loop
    ckpt_dir = Path(CHECKPOINT_DIR)
    start_time = time.time()

    # accumulate stats
    train_stats = []
    val_stats = []

    for epoch in range(config.epochs):
        # train an epoch
        train_stats.append(train_one_epoch(
            data_loader = train_loader,
            model = model,
            crit = crit,
            optimizer = optimizer,
            device = device,
            epoch = epoch,
            lr = lr,
            cutmix = cutmix,
            mixup = mixup,
            configs = config,
            model_ema = model_ema,
        ))

        # validate an epoch - the engine should handle everything...?
        val_stats.append(predict_one_epoch(
            data_loader=val_loader,
            model=model,
            crit=crit,
            device=device
        ))

        # also validate the EMA shadow so we can compare it against the raw model
        # (merged into the same epoch's val dict under ema_* keys)
        if model_ema is not None:
            ema_stats = predict_one_epoch(
                data_loader=val_loader,
                model=model_ema.ema,
                crit=crit,
                device=device,
                tag="EMA Validation"
            )
            val_stats[-1].update({f"ema_{k}": v for k, v in ema_stats.items()})

        # checkpoint saving
        is_last = epoch + 1 == config.epochs
        if (epoch + 1) % config.save_every == 0 or is_last:
            ckpt_path = ckpt_dir / f"checkpoint-clsfinetune-{epoch + 1}.pth"
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
    save_stats_csv(train_stats, out_dir / f"clstrain-train-{stamp}.csv")
    save_stats_csv(val_stats, out_dir / f"clstrain-val-{stamp}.csv")
    plot_stats(train_stats, val_stats, out_dir / f"clstrain-curves-{stamp}.png")

    # end-of-run 10-crop TTA val: measure how much TTA lifts the final model over
    # the single center-crop val (val_stats[-1]) before relying on it in finalres.
    # NOTE: each sample is 10 crops, so effective batch = batch_size * 10 -- lower
    # batch_size if this OOMs.
    tta_val_dataset = ds.ValDataset(DATA_DIR, ds.build_transform_cls_tta(config))
    tta_val_loader = DataLoader(
        tta_val_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = False,
        collate_fn = ds.cls_val_collate,
    )
    final_val = val_stats[-1] if val_stats else {}
    tta_stats = predict_one_epoch_tta(tta_val_loader, model, device)
    logger.info("Final val (center-crop vs 10-crop TTA): acc %.2f -> %.2f | macro %.2f -> %.2f",
                final_val.get("accuracy", float("nan")), tta_stats["accuracy"],
                final_val.get("macro_acc", float("nan")), tta_stats["macro_acc"])
    if model_ema is not None:
        ema_tta_stats = predict_one_epoch_tta(tta_val_loader, model_ema.ema, device,
                                              tag="EMA TTA Validation")
        logger.info("Final EMA val (center-crop vs 10-crop TTA): acc %.2f -> %.2f | macro %.2f -> %.2f",
                    final_val.get("ema_accuracy", float("nan")), ema_tta_stats["accuracy"],
                    final_val.get("ema_macro_acc", float("nan")), ema_tta_stats["macro_acc"])

    total_time = datetime.timedelta(seconds = int(time.time()) - start_time)
    logger.info("CLS fine-tuning done in %s", total_time)

    

# TODO: finish sometime
def get_args_parser() -> argparse.ArgumentParser:
    """
    CLI for python -m src.pretrain
     
    Unset flags fall back to PretrainConfigs defaults
    """
    parser = argparse.ArgumentParser("FCMAE pre-training", add_help=True)
    # any arg left as None is dropped before dataclasses.replace, so it keeps the
    # PretrainConfigs default. arg dest names match PretrainConfigs field names.

    return parser

def main():
    setup_logging()
    
    # too lazy to set up proper configs, change here
    config = ClsConfigs()

    run(config)

if __name__ == "__main__":
    main()