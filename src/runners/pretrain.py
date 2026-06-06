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
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode

from ..core.utils import (PretrainConfigs, seed_everything, get_device, check_cuda,
                          RAND_SEED, DATA_DIR, CHECKPOINT_DIR, SAVE_DIR)
from ..core.dataset import (TrainUnlabeledDataset, TrainLabeledDataset,
                           TrainMaskedDataset, PretrainDataset, NORM_MEAN, NORM_STD)
from ..models.convnext.fcmae import (FCMAE, convnextv2_atto, convnextv2_femto,
                                     convnextv2_pico, convnextv2_nano,
                                     convnextv2_tiny, convnextv2_base,
                                     convnextv2_large, convnextv2_huge)
from ..engines.fcmae_pretrain import train_one_epoch
from .utils import build_param_groups, save_checkpoint, load_checkpoint, LossScaler

logger = logging.getLogger(__name__)

def _denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo NORM_MEAN/NORM_STD normalization and clamp to [0, 1] for display."""
    mean = torch.tensor(NORM_MEAN, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def show_modeled_image(
    model: FCMAE,
    samples: torch.Tensor,
    config: PretrainConfigs,
    epoch: int,
    out_dir: str | Path = SAVE_DIR,
    max_images: int = 4,
) -> torch.Tensor:
    """
    Diagnostic: run the FCMAE on a batch and save a (original | masked | recon)
    grid so the reconstruction quality can be eyeballed during pre-training.

    Composites the visible patches with the model's predictions on the masked
    patches, de-normalizes, and writes `out_dir/recon_epoch{epoch}.png`. Returns
    the de-normalized reconstruction tensor (N, 3, H, W).
    """
    from torchvision.utils import save_image

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, pred, mask = model(samples, mask_ratio=config.mask_ratio)

        # pred comes back as (N, C, h, w) conv output; match forward_loss and
        # reshape to (N, L, p*p*c) before unpatchify.
        if pred.dim() == 4:
            n, c, _, _ = pred.shape
            pred = torch.einsum("ncl->nlc", pred.reshape(n, c, -1))
        recon_patches = model.unpatchify(pred)

        # mask: (N, L), 1 = removed/masked -> upsample to pixel space (N, 1, H, W)
        mask_px = model.upsample_mask(mask, model.patch_size).unsqueeze(1).type_as(samples)

        masked_input = samples * (1.0 - mask_px)
        reconstruction = samples * (1.0 - mask_px) + recon_patches * mask_px

        orig = _denormalize(samples)
        masked = _denormalize(masked_input)
        recon = _denormalize(reconstruction)

        n_show = min(samples.shape[0], max_images)
        rows = []
        for i in range(n_show):
            rows.extend([orig[i], masked[i], recon[i]])
        grid = torch.stack(rows, dim=0)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"recon_epoch{epoch}.png"
        save_image(grid, out_path, nrow=3)
        logger.info("Saved reconstruction preview to %s", out_path)

    if was_training:
        model.train()
    return recon

def build_transform(config: PretrainConfigs) -> v2.Transform:
    """
    Create and return the FCMAE pre-training augmentation pipeline.

    Mirrors the ConvNeXt-V2 reference (RandomResizedCrop scale (0.2, 1.0) +
    horizontal flip) but normalizes with the repo's plain NORM_MEAN/NORM_STD
    (0.5/0.5 -- arithmetic, NOT external ImageNet stats, per the no-external-data
    rule). The optimizer / lr / schedule referenced in the old docstring live in
    `run`, not here.
    """
    return v2.Compose([
        v2.RandomResizedCrop(
            config.size,
            scale=(0.2, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

def build_model(config: PretrainConfigs) -> FCMAE:
    """
    Construct an FCMAE from `config.model_size` via the existing fcmae factory
    functions.

    NOTE: the factories currently accept only (in_channels, img_size), so the
    decoder/patch/mask config fields (decoder_depth, decoder_embed_dim,
    patch_size, mask_ratio, norm_pix_loss) are left unused here -- wiring them
    through the factory signatures is a deferred follow-up in fcmae.py.
    """
    match config.model_size:
        case "atto":
            factory = convnextv2_atto
        case "femto":
            factory = convnextv2_femto
        case "pico":
            factory = convnextv2_pico
        case "nano":
            factory = convnextv2_nano
        case "tiny":
            factory = convnextv2_tiny
        case "base":
            factory = convnextv2_base
        case "large":
            factory = convnextv2_large
        case "huge":
            factory = convnextv2_huge
        case _:
            logger.warning("Unknown model_size %r, falling back to atto", config.model_size)
            factory = convnextv2_atto
    return factory(in_channels=config.channels, img_size=config.size, norm_pix_loss=config.norm_pix_loss)

def run(config: PretrainConfigs) -> None:
    """
    Run `config.epochs` epochs of FCMAE pre-training (single GPU, save-only).

    Pools the three TRAINING splits (unlabeled + labeled + seg, labels/masks
    dropped) into one image-only set, builds an FCMAE, and trains it with AMP and
    a per-iteration cosine lr schedule (annealed inside `train_one_epoch` from the
    constant base `lr` computed here). Checkpoints land in CHECKPOINT_DIR every
    `config.save_every` epochs and after the final epoch; optional reconstruction
    previews are written every `config.viz_every` epochs.
    """
    seed_everything(RAND_SEED)
    device = get_device()
    cudnn.benchmark = True
    logger.info("FCMAE pre-training on %s | model_size=%s", device, config.model_size)

    # --- data: pool the three training splits into one image-only set ---------
    transform = build_transform(config)
    unlabeled = TrainLabeledDataset(DATA_DIR, transform)
    labeled = TrainLabeledDataset(DATA_DIR, transform)
    masked = TrainMaskedDataset(DATA_DIR, transform)
    dataset = PretrainDataset(DATA_DIR, transform, [unlabeled, labeled, masked])
    if config.limit > 0:
        dataset.items = dataset.items[: config.limit]
    logger.info("Pre-training pool: %d images", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(check_cuda()),
        drop_last=True,
    )

    # --- model / optimizer / amp ---------------------------------------------
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_params / 1e6)

    eff_batch_size = config.batch_size * config.update_freq
    lr = config.blr * eff_batch_size / 256
    logger.info("base lr=%.2e | eff batch size=%d | actual lr=%.2e", config.blr, eff_batch_size, lr)

    param_groups = build_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=config.optim_momentum)
    loss_scaler = LossScaler(config.use_amp, "cuda" if check_cuda() else "")

    # cache one batch for reconstruction previews (only if enabled)
    viz_batch = None
    if config.viz_every > 0:
        viz_batch = next(iter(loader)).to(device, non_blocking=True)

    # --- training loop --------------------------------------------------------
    ckpt_dir = Path(CHECKPOINT_DIR)
    start_time = time.time()
    for epoch in range(config.epochs):
        train_one_epoch(
            model, loader, optimizer, device, epoch,
            loss_scaler, lr, config, config.update_freq,
        )

        is_last = epoch + 1 == config.epochs
        if (epoch + 1) % config.save_every == 0 or is_last:
            ckpt_path = ckpt_dir / f"checkpoint-{epoch}.pth"
            save_checkpoint(
                model, ckpt_path,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch=epoch, config=config,
            )
            logger.info("Saved checkpoint to %s", ckpt_path)

        if viz_batch is not None and ((epoch + 1) % config.viz_every == 0 or is_last):
            show_modeled_image(model, viz_batch, config, epoch + 1, SAVE_DIR)

    total_time = datetime.timedelta(seconds=int(time.time() - start_time))
    logger.info("FCMAE pre-training done in %s", total_time)


def get_args_parser() -> argparse.ArgumentParser:
    """CLI for `python -m src.runners.pretrain`. Unset flags fall back to PretrainConfigs defaults."""
    parser = argparse.ArgumentParser("FCMAE pre-training", add_help=True)
    # any arg left as None is dropped before dataclasses.replace, so it keeps the
    # PretrainConfigs default. arg dest names match PretrainConfigs field names.
    parser.add_argument("--model-size", type=str, default=None,
                        help="atto/femto/pico/nano/tiny/base/large/huge")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--update-freq", type=int, default=None, help="gradient accumulation steps")
    parser.add_argument("--blr", type=float, default=None, help="base learning rate")
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-embed-dim", type=int, default=None)
    parser.add_argument("--norm-pix-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="use only first N images (0 = all)")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--viz-every", type=int, default=None, help="recon preview every N epochs (0 = off)")
    parser.add_argument("--amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=None,
                        help="toggle AMP loss-scaling")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = get_args_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = dataclasses.replace(PretrainConfigs(), **overrides)
    run(config)


if __name__ == "__main__":
    main()