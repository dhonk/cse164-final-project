"""
tmp_pretrain_finetune.py: scratch script extracted from notebook.ipynb.

Runs the full notebook flow without the notebook overhead:
  1. FCMAE self-supervised pretraining (saves checkpoints + recon previews).
  2. Warm-start a dense ConvNeXtV2 classifier from the latest FCMAE checkpoint
     and fine-tune it on the labeled split.
  3. Report val classification accuracy.

Docker/CUDA only (FCMAE -> SparseConvNeXtV2 imports spconv, which won't
pip-install on Windows). Run from project root:
    python tmp_pretrain_finetune.py
"""

from __future__ import annotations

import os
import sys
import time
import datetime
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import src.models.convnext.convnextv2 as cnv2
from src.core.utils import (
    PretrainConfigs, ClsFinetuneConfigs, seed_everything, get_device,
    RAND_SEED, DATA_DIR, CHECKPOINT_DIR, SAVE_DIR,
)
from src.core.dataset import (
    NORM_MEAN, NORM_STD,
    TrainLabeledDataset, TrainMaskedDataset, TrainUnlabeledDataset,
    ValDataset, PretrainDataset,
)
from src.engines import convnext_cls as cls
from src.engines.fcmae_pretrain import train_one_epoch
from src.runners.pretrain import build_transform, build_model, show_modeled_image
from src.runners.utils import build_param_groups, save_checkpoint, LossScaler

# core.utils logs to logs/<date>.log on import; also stream to the console.
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

DATA_ROOT = Path("./data")


def image_transform(crop: int = 224, train: bool = True) -> v2.Transform:
    """PIL image -> normalized float tensor (image-only splits)."""
    ops = (
        [v2.RandomResizedCrop(crop, scale=(0.4, 1.0), antialias=True),
         v2.RandomHorizontalFlip(0.5),
         v2.ColorJitter(0.2, 0.2, 0.2, 0.05)]
        if train else
        [v2.Resize((crop, crop), antialias=True)]
    )
    return v2.Compose(ops + [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(list(NORM_MEAN), list(NORM_STD)),
    ])


# ============================================================================
# 1. FCMAE pretraining
# ============================================================================
def pretrain() -> None:
    # Smoke-run config. For a full run, drop `limit` (use all ~60.5k imgs),
    # raise `epochs`, and bump `batch_size`/`num_workers`, e.g.:
    #   PretrainConfigs(model_size="huge", epochs=400, warmup_epochs=20,
    #                   num_workers=4, batch_size=1024, save_every=50)
    config = PretrainConfigs(
        model_size="nano",
        epochs=400,
        warmup_epochs=40,
        batch_size=128,
        num_workers=6,
        save_every=50,
        viz_every=50,
    )

    seed_everything(RAND_SEED)
    device = get_device()
    cudnn.benchmark = True
    print(f"FCMAE pretraining on {device} | model_size={config.model_size}")

    # pool the three training splits into one image-only set
    transform = build_transform(config)
    unlabeled = TrainUnlabeledDataset(DATA_DIR, transform)
    labeled = TrainLabeledDataset(DATA_DIR, transform)
    masked = TrainMaskedDataset(DATA_DIR, transform)
    dataset = PretrainDataset(DATA_DIR, transform, [unlabeled, labeled, masked])
    if config.limit > 0:
        dataset.items = dataset.items[: config.limit]
    print(f"Pre-training pool: {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- model / optimizer / amp ---------------------------------------------
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.2f}M")

    eff_batch_size = config.batch_size * config.update_freq
    lr = config.blr * eff_batch_size / 256
    print(f"base lr={config.blr:.2e} | eff batch size={eff_batch_size} | actual lr={lr:.2e}")

    param_groups = build_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=config.optim_momentum)
    loss_scaler = LossScaler(config.use_amp, "cuda" if device.type == "cuda" else "")

    # cache 5 random samples for reconstruction previews (only if enabled)
    N_VIZ = 5
    viz_batch = None
    if config.viz_every > 0:
        viz_idx = random.sample(range(len(dataset)), k=min(N_VIZ, len(dataset)))
        viz_batch = torch.stack([dataset[i] for i in viz_idx]).to(device, non_blocking=True)

    # --- training loop -------------------------------------------------------
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
            print(f"Saved checkpoint to {ckpt_path}")

        if viz_batch is not None and ((epoch + 1) % config.viz_every == 0 or is_last):
            show_modeled_image(model, viz_batch, config, epoch + 1, SAVE_DIR, max_images=N_VIZ)

    total_time = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"FCMAE pre-training done in {total_time}")


# ============================================================================
# 2. Classifier fine-tuning (warm-start from the latest FCMAE checkpoint)
# ============================================================================
DENSE_FACTORIES = {
    "atto": cnv2.convnextv2_atto,   "femto": cnv2.convnextv2_femto,
    "pico": cnv2.convnext_pico,     "nano":  cnv2.convnextv2_nano,
    "tiny": cnv2.convnextv2_tiny,   "base":  cnv2.convnextv2_base,
    "large": cnv2.convnextv2_large, "huge":  cnv2.convnextv2_huge,
}


def finetune() -> tuple[nn.Module, torch.device]:
    seed_everything(RAND_SEED)
    device = get_device()

    # --- locate the most recent FCMAE checkpoint -----------------------------
    ckpts = list(Path(CHECKPOINT_DIR).glob("checkpoint-*.pth"))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint-*.pth in {CHECKPOINT_DIR}; run FCMAE pretraining first.")
    ckpt_path = max(ckpts, key=os.path.getmtime)          # most recent by mtime
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loading checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # --- build a dense classifier matching the pretrained backbone size ------
    # the FCMAE checkpoint stores its PretrainConfigs, so the encoder dims line up.
    pre_cfg = ckpt.get("config")
    model_size = getattr(pre_cfg, "model_size", "atto")

    config = ClsFinetuneConfigs(
        model_size=model_size,
        epochs=40,
        warmup_epochs=10,
        batch_size=128,
        num_workers=4,
        limit=0,          # 0 = all ~7.5k labeled imgs; set small for a smoke run
        save_every=10,
    )

    model = DENSE_FACTORIES[config.model_size](
        in_channels=config.channels,
        num_classes=config.num_classes,
        drop_path_rate=config.drop_path,
        head_init_scale=config.head_init,
    ).to(device)

    # --- warm-start: strip the FCMAE "encoder." prefix, drop decoder keys ----
    # FCMAE stores the backbone under encoder.* (+ proj/decoder/pred/mask_token
    # used only for reconstruction). The dense classifier's downsample_layers.*/
    # stages.* match the encoder 1:1; norm/head/seg_norms stay random (strict=False).
    state = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v
                     for k, v in state.items() if k.startswith("encoder.")}
    result = model.load_state_dict(encoder_state, strict=False)
    loaded = len(encoder_state) - len(result.unexpected_keys)
    print(f"Warm-started {loaded}/{len(encoder_state)} encoder tensors "
          f"({len(result.unexpected_keys)} unexpected, "
          f"{len(result.missing_keys)} left random)")

    # --- data / optimizer / loss ---------------------------------------------
    train_ds = TrainLabeledDataset(DATA_ROOT, image_transform(train=True))
    if config.limit > 0:
        train_ds.items = train_ds.items[: config.limit]
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Fine-tuning on {len(train_ds)} labeled images | model_size={config.model_size}")

    eff_batch_size = config.batch_size * config.update_freq
    lr = config.blr * eff_batch_size / 256
    param_groups = build_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=config.optim_momentum)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # --- fine-tune -----------------------------------------------------------
    ckpt_dir = Path(CHECKPOINT_DIR)
    for epoch in range(config.epochs):
        cls.train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch,
            lr, config, config.update_freq,
        )
        is_last = epoch + 1 == config.epochs
        if (epoch + 1) % config.save_every == 0 or is_last:
            out = ckpt_dir / f"cls_finetune-{epoch}.pth"
            save_checkpoint(model, out, optimizer=optimizer, epoch=epoch, config=config)
            print(f"Saved classifier checkpoint to {out}")

    print("Classifier fine-tuning done.")
    return model, device


# ============================================================================
# 3. Val classification accuracy
# ============================================================================
def _cls_collate(batch):
    # ValDataset yields (img, seg_mask, cls_label, orig_size); the GT masks keep
    # their ORIGINAL (variable) sizes, so default collation can't stack a batch.
    # Keep just the image + cls_label -> the (image, label) 2-tuple eval expects.
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.as_tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, labels


def validate(model: nn.Module, device: torch.device) -> dict:
    val_ds = ValDataset(DATA_ROOT, image_transform(train=False))
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
        collate_fn=_cls_collate,
    )
    metrics = cls.predict_one_epoch(val_loader, model, device)
    print(f"Val classification accuracy: {metrics['accuracy']:.4f}  "
          f"(over {len(val_ds)} images) | val loss: {metrics['loss']:.4f}")
    return metrics


if __name__ == "__main__":
    pretrain()
    model, device = finetune()
    validate(model, device)
