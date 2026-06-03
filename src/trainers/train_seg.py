"""Driver: train ConvNeXt V2 + UPerNet for semantic segmentation.

End-to-end stage harness (Phase 4.1): build a dense backbone + UPerNet decoder,
optionally warm-start the backbone from an FCMAE checkpoint (remap + strict=False;
the seg_norms/head stay randomly initialized), train N epochs on the 3k
`train_seg` split via the seg engine, score the 750-image `val` split each
`--eval-every` epochs with the reference kaggle metric, and checkpoint best
(by segmentation_score) + last.

Runs locally on the 3090 from RANDOM init (no FCMAE weights needed); pass
`--finetune path/to/fcmae.pth` to warm-start once a pretrained encoder exists.

    venv\\Scripts\\python.exe -m src.trainers.train_seg --epochs 50 --batch-size 16
    # quick smoke (few train batches, 1 epoch, still scores all val):
    venv\\Scripts\\python.exe -m src.trainers.train_seg --epochs 1 --limit-train 32 --batch-size 8

DEVIATION (documented): the optimizer is a plain `AdamW` over backbone+decoder
params, NOT `core.optim.create_optimizer` with layer-wise LR decay. Layer decay
matters when finetuning a *pretrained* backbone; from random init it buys little
and complicates the two-module (backbone+decoder) param grouping. Wiring
`create_optimizer` + layer decay is a follow-up for when FCMAE weights land.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..core.utils import (
    IGNORE_ID,
    NativeScaler,
    load_checkpoint,
    remap_checkpoint_keys,
    save_checkpoint,
    seed_everything,
)
from ..core.dataset import TrainSegDataset, ValDataset, build_transforms
from ..core.evaluate import resize_mask_nearest, score_val
from ..core.log import setup_logging, get_logger
from ..models.convnext import convnextv2 as cv
from ..models.upernet.upernet import build_upernet
from ..engines.convnext_upernet_seg import seg_forward, train_one_epoch

# name -> (size factory, per-stage dims). dims feed UPerNet's fpn_inplanes/fc_dim.
MODELS = {
    "atto": (cv.convnextv2_atto, [40, 80, 160, 320]),
    "femto": (cv.convnextv2_femto, [48, 96, 192, 384]),
    "nano": (cv.convnextv2_nano, [80, 160, 320, 640]),
    "tiny": (cv.convnextv2_tiny, [96, 192, 384, 768]),
}

log = get_logger(__name__)


def get_args():
    p = argparse.ArgumentParser(description="Train ConvNeXt V2 + UPerNet segmentation")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./checkpoints/seg")
    p.add_argument("--model", default="atto", choices=list(MODELS))
    p.add_argument("--finetune", default="", help="FCMAE encoder ckpt to warm-start (optional)")
    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
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


def build_models(args, device):
    factory, dims = MODELS[args.model]
    # num_classes=300 builds the (unused-here) cls head; the seg path uses
    # forward_features_seg + seg_norms. drop_path_rate regularizes the backbone.
    backbone = factory(num_classes=300, drop_path_rate=args.drop_path).to(device)
    decoder = build_upernet(dims, num_class=301).to(device)

    if args.finetune:
        # Warm-start the ENCODER only from an FCMAE checkpoint. remap converts the
        # sparse layout to the dense backbone; strict=False lets seg_norms + head
        # (absent from the FCMAE ckpt) stay randomly initialized.
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = remap_checkpoint_keys(state)
        msg = backbone.load_state_dict(state, strict=False)
        log.info("FCMAE warm-start: missing=%d unexpected=%d",
                 len(msg.missing_keys), len(msg.unexpected_keys))
    return backbone, decoder, dims


@torch.no_grad()
def evaluate_val(backbone, decoder, args, device, use_amp):
    """Predict masks for every val image and score via the reference kaggle metric.

    Iterates `ValDataset` directly (masks are at native res / variable size, which
    the default DataLoader collate can't stack). For each image: forward at crop
    res -> argmax seg ids (0..300) -> resize NEAREST back to original (W,H). The
    `class_id` is the majority foreground seg id minus 1 (placeholder -- seg
    selection reads segmentation_score, not classification accuracy).
    """
    backbone.eval()
    decoder.eval()
    ds = ValDataset(args.data_root, build_transforms("eval", args.crop_size))
    rows = []
    for i in range(len(ds)):
        image, _label, _seg_gt, name, orig_size = ds[i]
        x = image.unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = seg_forward(backbone, decoder, x)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)  # (C,C), ids 0..300
        pred = resize_mask_nearest(pred, orig_size)                    # back to (H,W)
        fg = pred[pred > 0]
        class_id = int(np.bincount(fg).argmax()) - 1 if fg.size else 0
        class_id = max(0, min(299, class_id))
        rows.append({"image": name, "class_id": class_id, "seg_mask": pred})
    return score_val(rows, args.data_root)


def main():
    args = get_args()
    setup_logging()
    seed_everything(args.seed)
    use_amp = (not args.no_amp) and args.device == "cuda" and torch.cuda.is_available()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    log.info("device=%s use_amp=%s model=%s", device, use_amp, args.model)

    # --- data ----------------------------------------------------------------
    train_ds = TrainSegDataset(args.data_root, build_transforms("seg", args.crop_size))
    if args.limit_train:
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    log.info("train_seg: %d imgs, %d batches/epoch", len(train_ds), len(train_loader))

    # --- model / optim -------------------------------------------------------
    backbone, decoder, _dims = build_models(args, device)
    criterion = nn.NLLLoss(ignore_index=IGNORE_ID).to(device)
    scaler = NativeScaler()
    params = list(backbone.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # --- train loop ----------------------------------------------------------
    best_seg = -1.0
    for epoch in range(args.epochs):
        train_one_epoch(
            backbone, decoder, criterion, train_loader, optimizer, device, epoch,
            scaler, args, max_norm=args.max_norm, use_amp=use_amp,
        )
        is_last = epoch == args.epochs - 1
        if args.eval_every and (is_last or (epoch + 1) % args.eval_every == 0):
            scores = evaluate_val(backbone, decoder, args, device, use_amp)
            seg = float(scores["segmentation_score"])
            log.info("[epoch %d] seg=%.4f mIoU=%.4f boundaryF=%.4f rareMIoU=%.4f auto=%.4f",
                     epoch, seg, scores["mean_iou"], scores["boundary_f_score"],
                     scores["rare_class_miou"], scores["automated_score"])
            extra = {"args": vars(args), "scores": scores, "dims": _dims,
                     "decoder": decoder.state_dict()}
            save_checkpoint(out_dir / "last.pth", backbone, optimizer, scaler, epoch, extra)
            if seg > best_seg:
                best_seg = seg
                save_checkpoint(out_dir / "best.pth", backbone, optimizer, scaler, epoch, extra)
                log.info("  new best segmentation_score=%.4f -> saved best.pth", best_seg)

    log.info("done. best segmentation_score=%.4f", best_seg)


if __name__ == "__main__":
    main()
