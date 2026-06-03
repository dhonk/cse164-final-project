"""Driver: train ConvNeXt V2 for image classification.

End-to-end stage harness (Phase 4.2): build a dense backbone (its own
`forward_features` + linear `head`, `num_classes=300`), optionally warm-start the
encoder from an FCMAE checkpoint (remap + strict=False; the `head` stays randomly
initialized), train N epochs on the 7.5k `train_labeled` split via the cls engine,
score the 750-image `val` split each `--eval-every` epochs (top-1 AND macro /
class-balanced accuracy), and checkpoint best (by macro_acc -- the leaderboard cls
metric is class-balanced) + last.

Runs locally on the 3090 from RANDOM init (no FCMAE weights needed); pass
`--finetune path/to/fcmae.pth` to warm-start once a pretrained encoder exists.

    venv\\Scripts\\python.exe -m src.trainers.train_cls --epochs 50 --batch-size 64
    # quick smoke (few train imgs, 1 epoch, still scores all val):
    venv\\Scripts\\python.exe -m src.trainers.train_cls --epochs 1 --limit-train 64 --batch-size 16

DEVIATION (documented, same as train_seg): the optimizer is a plain `AdamW` over
the backbone params, NOT `core.optim.create_optimizer` with layer-wise LR decay.
Layer decay matters when finetuning a *pretrained* backbone; from random init it
buys little. Wiring `create_optimizer` + layer decay is a follow-up for when FCMAE
weights land.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..core.utils import (
    NativeScaler,
    remap_checkpoint_keys,
    save_checkpoint,
    seed_everything,
)
from ..core.dataset import TrainLabeledDataset, ValDataset, build_transforms
from ..core.log import setup_logging, get_logger
from ..models.convnext import convnextv2 as cv
from ..engines.convnext_cls import train_one_epoch, evaluate

# name -> (size factory, per-stage dims). dims kept for ckpt provenance / parity
# with train_seg (the cls path itself only needs the factory).
MODELS = {
    "atto": (cv.convnextv2_atto, [40, 80, 160, 320]),
    "femto": (cv.convnextv2_femto, [48, 96, 192, 384]),
    "nano": (cv.convnextv2_nano, [80, 160, 320, 640]),
    "tiny": (cv.convnextv2_tiny, [96, 192, 384, 768]),
}

log = get_logger(__name__)


def get_args():
    p = argparse.ArgumentParser(description="Train ConvNeXt V2 image classifier")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./checkpoints/cls")
    p.add_argument("--model", default="atto", choices=list(MODELS))
    p.add_argument("--finetune", default="", help="FCMAE encoder ckpt to warm-start (optional)")
    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    # data / runtime
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--limit-train", type=int, default=0, help="subset N train imgs (smoke)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(args, device):
    factory, dims = MODELS[args.model]
    # backbone(x) = forward_features (GAP + LayerNorm) -> linear head; (N, 300) logits.
    model = factory(num_classes=300, drop_path_rate=args.drop_path).to(device)

    if args.finetune:
        # Warm-start the ENCODER only from an FCMAE checkpoint. remap converts the
        # sparse layout to the dense backbone; strict=False lets the `head`
        # (absent from the FCMAE ckpt) stay randomly initialized.
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = remap_checkpoint_keys(state)
        msg = model.load_state_dict(state, strict=False)
        log.info("FCMAE warm-start: missing=%d unexpected=%d",
                 len(msg.missing_keys), len(msg.unexpected_keys))
    return model, dims


def _val_collate(batch):
    """Stack only image + label from ValDataset's 5-tuple.

    `ValDataset` yields `(image, label, seg_id, name, orig_size)`; the seg_id GT
    masks are at native (variable) resolution, which the default collate can't
    stack. The cls `evaluate` reads only batch[0:2], so we drop the rest here.
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return images, labels


def main():
    args = get_args()
    setup_logging()
    seed_everything(args.seed)
    use_amp = (not args.no_amp) and args.device == "cuda" and torch.cuda.is_available()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    log.info("device=%s use_amp=%s model=%s", device, use_amp, args.model)

    # --- data ----------------------------------------------------------------
    train_ds = TrainLabeledDataset(args.data_root, build_transforms("cls", args.crop_size))
    if args.limit_train:
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_ds = ValDataset(args.data_root, build_transforms("eval", args.crop_size))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=_val_collate,
    )
    log.info("train_labeled: %d imgs, %d batches/epoch; val: %d imgs",
             len(train_ds), len(train_loader), len(val_ds))

    # --- model / optim -------------------------------------------------------
    model, dims = build_model(args, device)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = NativeScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- train loop ----------------------------------------------------------
    best_macro = -1.0
    for epoch in range(args.epochs):
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            scaler, args, max_norm=args.max_norm, use_amp=use_amp,
        )
        is_last = epoch == args.epochs - 1
        if args.eval_every and (is_last or (epoch + 1) % args.eval_every == 0):
            scores = evaluate(model, val_loader, device, use_amp=use_amp)
            macro = float(scores["macro_acc"])
            log.info("[epoch %d] acc1=%.4f macro_acc=%.4f",
                     epoch, scores.get("acc1", 0.0), macro)
            extra = {"args": vars(args), "scores": scores, "dims": dims}
            save_checkpoint(out_dir / "last.pth", model, optimizer, scaler, epoch, extra)
            if macro > best_macro:
                best_macro = macro
                save_checkpoint(out_dir / "best.pth", model, optimizer, scaler, epoch, extra)
                log.info("  new best macro_acc=%.4f -> saved best.pth", best_macro)

    log.info("done. best macro_acc=%.4f", best_macro)


if __name__ == "__main__":
    main()
