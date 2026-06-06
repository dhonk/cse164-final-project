"""
tmp_overfit_fcmae.py  ---  TEMPORARY sanity-check, DELETE after use.

Goal: prove the FCMAE forward/backward/optimization loop actually *learns* by
making it OVERFIT a tiny, fixed image pool as hard as possible. If the loss does
not collapse toward ~0 on, say, 8 images with no weight decay and no augmentation,
something in the model/engine is broken (dead grads, detached graph, masking bug).

Aggressive-overfit knobs vs. the real pretrain run:
  * atto backbone (smallest)                      -> fast, still capacity to memorize
  * tiny fixed pool (default 8 imgs)              -> trivially memorizable
  * weight_decay = 0                              -> no pull back toward init
  * DETERMINISTIC transform (resize only)         -> identical pixels every epoch
      (the real run uses RandomResizedCrop+flip, which FIGHTS overfitting)
  * FIXED mask (same patches removed every epoch) -> identical recon target/task
      (the real run re-samples gen_random_mask each step, a moving target)
  * (near-)constant lr (min_lr == lr, no warmup)  -> no schedule masking progress
  * one full batch / epoch, shuffle off           -> same batch repeatedly

Run from project root (CUDA required -- spconv sparse encoder is CUDA-only):
    venv\\Scripts\\python.exe tmp_overfit_fcmae.py
    venv\\Scripts\\python.exe tmp_overfit_fcmae.py --limit 4 --epochs 600 --lr 5e-4
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode

from src.core.utils import (PretrainConfigs, seed_everything, get_device,
                            check_cuda, RAND_SEED, DATA_DIR, SAVE_DIR)
from src.core.dataset import (TrainLabeledDataset, TrainMaskedDataset, TrainUnlabeledDataset,
                             PretrainDataset, NORM_MEAN, NORM_STD)
from src.runners.pretrain import build_model, show_modeled_image
from src.runners.utils import build_param_groups, LossScaler
from src.engines.fcmae_pretrain import train_one_epoch

logger = logging.getLogger(__name__)


def freeze_mask(model: torch.nn.Module) -> None:
    """Replace the model's gen_random_mask with a cache: the FIRST call uses the real
    algorithm, every later call returns that SAME mask (keyed by batch N + img size).
    With a fixed image pool this makes the masked-patch reconstruction target identical
    every epoch -- removes mask sampling as a source of loss variance during overfit."""
    orig = model.gen_random_mask  # bound method
    cache: dict = {}

    def fixed(x: torch.Tensor, mask_ratio: float = 0.6) -> torch.Tensor:
        key = (x.shape[0], x.shape[2])
        if key not in cache:
            cache[key] = orig(x, mask_ratio)
        return cache[key]

    model.gen_random_mask = fixed  # instance attr shadows the class method (no self)


def deterministic_transform(size: int) -> v2.Transform:
    """Resize-only pipeline: NO random crop/flip so every epoch sees identical
    pixels -- essential for a clean overfit signal."""
    return v2.Compose([
        v2.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_args():
    p = argparse.ArgumentParser("FCMAE overfit sanity check")
    p.add_argument("--limit", type=int, default=1, help="number of fixed images to memorize")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2.5e-3, help="constant lr (bypasses blr/256 scaling)")
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--model-size", type=str, default="tiny")
    p.add_argument("--viz-every", type=int, default=50, help="save recon grid every N epochs (0=off)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                   help="fp32 by default for a clean overfit signal")
    return p.parse_args()


def main():
    args = get_args()

    config = dataclasses.replace(
        PretrainConfigs(),
        model_size=args.model_size,
        epochs=args.epochs,
        warmup_epochs=0,            # no warmup
        min_lr=1e-6,             # min_lr == lr -> cosine term is constant -> flat lr
        weight_decay=0.05,          
        mask_ratio=args.mask_ratio,
        batch_size=1024,
        limit=args.limit,
        num_workers=6,
        update_freq=1,
        use_amp=args.amp,
        viz_every=args.viz_every,
        norm_pix_loss=False,
    )

    seed_everything(RAND_SEED)
    device = get_device()
    if not check_cuda():
        logger.warning("CUDA not available -- spconv sparse encoder is CUDA-only and will assert.")
    logger.info("OVERFIT FCMAE | size=%s | %d imgs | lr=%.2e | wd=%g | mask=%.2f | epochs=%d",
                config.model_size, config.limit, args.lr, config.weight_decay,
                config.mask_ratio, config.epochs)

    # --- tiny fixed pool, deterministic pixels --------------------------------
    transform = deterministic_transform(config.size)
    unlabeled = TrainUnlabeledDataset(DATA_DIR, transform)
    masked = TrainMaskedDataset(DATA_DIR, transform)
    dataset = PretrainDataset(DATA_DIR, transform, [unlabeled])
    dataset.items = dataset.items[: config.limit]
    logger.info("Overfit pool: %d images", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,             # same batch every epoch
        num_workers=config.num_workers,
        pin_memory=check_cuda(),
        drop_last=False,
    )

    # --- model / optimizer / amp ----------------------------------------------
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_params / 1e6)

    freeze_mask(model)  # same mask every epoch -> fixed recon target for clean overfit
    logger.info("Mask frozen: gen_random_mask now returns one fixed mask per batch shape.")

    param_groups = build_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=config.optim_momentum)
    loss_scaler = LossScaler(config.use_amp, "cuda" if check_cuda() else "")

    viz_batch = next(iter(loader)).to(device, non_blocking=True) if config.viz_every > 0 else None

    # --- overfit loop ---------------------------------------------------------
    for epoch in range(config.epochs):
        train_one_epoch(
            model, loader, optimizer, device, epoch,
            loss_scaler, args.lr, config, config.update_freq,
        )
        if viz_batch is not None and ((epoch + 1) % config.viz_every == 0 or epoch + 1 == config.epochs):
            show_modeled_image(model, viz_batch, config, epoch + 1, SAVE_DIR)

    logger.info("Overfit run done. If loss collapsed toward ~0, the FCMAE loop learns. "
                "Check %s for reconstruction grids.", SAVE_DIR)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
