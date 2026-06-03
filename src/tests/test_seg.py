"""Predict segmentation masks for every test image + build the final submission.

Phase 5.1 (mask prediction) + 5.3 (merge cls class_id with seg masks, write one
`submission.csv`, validate). Loads a segmentation checkpoint from
`trainers/train_seg.py` (backbone + UPerNet decoder), predicts a seg-id mask per
image, resizes it back to the image's ORIGINAL (W,H) with **nearest-neighbor**
(never bilinear — it would invent fractional ids), and RLE-encodes it.

The image-level `class_id` comes from a **classification** checkpoint when
`--cls-ckpt` is given (the proper merge: the cls head is the better classifier);
otherwise it falls back to the majority foreground seg id (minus 1) from the mask.

    # final submission (both heads):
    venv\\Scripts\\python.exe -m src.tests.test_seg --seg-ckpt checkpoints/seg/best.pth \\
        --cls-ckpt checkpoints/cls/best.pth --output submission.csv
    # smoke on labeled val (writes + scores via the reference validator):
    venv\\Scripts\\python.exe -m src.tests.test_seg --seg-ckpt ... --split val --output val_sub.csv

Predicted ids are always in `0..300` (argmax over 301 channels) — **never 1000**
(ignore is ground-truth-only). Always runs the starter validator before you trust
the file (`--no-validate` to skip).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core.dataset import build_transforms
from ..core.evaluate import resize_mask_nearest, write_submission, score_val
from ..core.log import setup_logging, get_logger
from ..trainers.train_seg import MODELS
from ..models.upernet.upernet import build_upernet
from ..engines.convnext_upernet_seg import seg_forward
from .test_cls import InferDataset, infer_collate, load_cls_model, predict_cls

log = get_logger(__name__)


def load_seg_model(ckpt_path: str | Path, device):
    """Rebuild backbone + UPerNet decoder from a `train_seg` checkpoint (strict).

    `args["model"]` names the size factory (and thus the decoder's `fpn_inplanes`
    dims); both the backbone (`model`) and the `decoder` state load strict.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    margs = ck.get("args", {})
    model_name = margs.get("model", "atto")
    factory, dims = MODELS[model_name]
    backbone = factory(num_classes=300).to(device)
    backbone.load_state_dict(ck["model"], strict=True)
    decoder = build_upernet(dims, num_class=301).to(device)
    decoder.load_state_dict(ck["decoder"], strict=True)
    backbone.eval()
    decoder.eval()
    log.info("loaded seg model=%s from %s (epoch %s)",
             model_name, ckpt_path, ck.get("epoch"))
    return backbone, decoder, margs


@torch.no_grad()
def predict_masks(backbone, decoder, loader, device, use_amp: bool = True) -> dict[str, np.ndarray]:
    """Predict a native-resolution seg-id mask per image; returns `{name: mask}`.

    Forward at crop res -> `argmax` over 301 channels = seg ids `0..300` ->
    `resize_mask_nearest` back to the image's original `(W, H)`.
    """
    backbone.eval()
    decoder.eval()
    out: dict[str, np.ndarray] = {}
    for images, names, orig_sizes in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = seg_forward(backbone, decoder, images)  # (N, 301, H, W)
        ids = logits.argmax(dim=1).cpu().numpy().astype(np.int64)  # (N, H, W), 0..300
        for pred, name, orig in zip(ids, names, orig_sizes):
            out[name] = resize_mask_nearest(pred, orig)  # back to native (H, W)
    return out


def _fallback_class_id(mask: np.ndarray) -> int:
    """Image-level class from the mask: majority foreground seg id minus 1.

    Used only when no cls checkpoint is supplied. seg_id k (1..300) -> class k-1
    (0..299); all-background -> class 0. Clamped into range defensively.
    """
    fg = mask[mask > 0]
    if fg.size == 0:
        return 0
    return max(0, min(299, int(np.bincount(fg).argmax()) - 1))


def run_validator(submission: Path, data_root: Path, split: str) -> int:
    """Run the starter validator (format check; also scores on `val`).

    Invoked as a subprocess with cwd = `starter/` so its `from kaggle_metric
    import ...` resolves. Returns the process exit code (0 = passed).
    """
    starter = Path(__file__).resolve().parents[2] / "starter"
    cmd = [sys.executable, "validate_submission_csv.py",
           "--submission", str(submission.resolve()),
           "--data-root", str(data_root.resolve()),
           "--split", split]
    log.info("validating: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(starter), capture_output=True, text=True)
    if proc.stdout:
        log.info("validator stdout:\n%s", proc.stdout.strip())
    if proc.returncode != 0:
        log.error("validator FAILED (exit %d):\n%s", proc.returncode, proc.stderr.strip())
    else:
        log.info("validator PASSED")
    return proc.returncode


def get_args():
    p = argparse.ArgumentParser(description="Predict masks + build submission.csv")
    p.add_argument("--seg-ckpt", required=True, help="checkpoint from train_seg")
    p.add_argument("--cls-ckpt", default="", help="checkpoint from train_cls (for class_id)")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    setup_logging()
    use_amp = (not args.no_amp) and args.device == "cuda" and torch.cuda.is_available()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = InferDataset(args.data_root, args.split, build_transforms("eval", args.crop_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=infer_collate)
    log.info("inference: split=%s %d images use_amp=%s", args.split, len(ds), use_amp)

    # --- segmentation masks --------------------------------------------------
    backbone, decoder, _margs = load_seg_model(args.seg_ckpt, device)
    masks = predict_masks(backbone, decoder, loader, device, use_amp)

    # --- class ids (cls head if available, else derived from the mask) -------
    if args.cls_ckpt:
        cls_model, _ = load_cls_model(args.cls_ckpt, device)
        cls_preds = predict_cls(cls_model, loader, device, use_amp)
        log.info("class_id source: cls checkpoint %s", args.cls_ckpt)
    else:
        cls_preds = {name: _fallback_class_id(m) for name, m in masks.items()}
        log.info("class_id source: majority-foreground fallback (no --cls-ckpt)")

    # --- merge -> submission -------------------------------------------------
    rows = [{"image": name, "class_id": int(cls_preds.get(name, 0)), "seg_mask": mask}
            for name, mask in masks.items()]
    out_path = Path(args.output)
    write_submission(rows, out_path)
    log.info("wrote %d rows -> %s", len(rows), out_path)

    # --- offline score (val only) + format validation ------------------------
    if args.split == "val":
        scores = score_val(rows, args.data_root, "val")
        log.info("[val score] auto=%.4f seg=%.4f mIoU=%.4f boundaryF=%.4f rareMIoU=%.4f clsMacroAcc=%.4f",
                 scores["automated_score"], scores["segmentation_score"], scores["mean_iou"],
                 scores["boundary_f_score"], scores["rare_class_miou"],
                 scores["classification_macro_accuracy"])
    if not args.no_validate:
        run_validator(out_path, Path(args.data_root), args.split)


if __name__ == "__main__":
    main()
