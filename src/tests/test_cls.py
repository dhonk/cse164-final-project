"""Predict image-level class_id for every test image (Phase 5.2).

Loads a classification checkpoint produced by `trainers/train_cls.py`, runs the
dense backbone over the inference split, and emits a per-image `class_id` in
`0..299`. Standalone, this writes a `submission.csv` whose segmentation column is
all-background (`"0"`); the real segmentation masks are merged in by
`tests/test_seg.py` (which imports `predict_cls` from here).

    venv\\Scripts\\python.exe -m src.tests.test_cls --cls-ckpt checkpoints/cls/best.pth
    # smoke on the labeled val split (also runnable for a quick sanity check):
    venv\\Scripts\\python.exe -m src.tests.test_cls --cls-ckpt ... --split val --output val_cls.csv

The class head outputs `(N, 300)` logits and `argmax` gives `class_id` directly
(0..299) — no off-by-one (that only bites segmentation, where seg_id = class+1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from ..core.dataset import TestDataset, ValDataset, build_transforms
from ..core.evaluate import write_submission
from ..core.log import setup_logging, get_logger
from ..trainers.train_cls import MODELS

log = get_logger(__name__)


# --- shared inference dataset (reused by test_seg) ----------------------------
class InferDataset(Dataset):
    """Normalizes an inference split to uniform `(image, name, orig_size)` items.

    `--split test` -> `TestDataset` (the real hidden submission target, image-only).
    `--split val`  -> `ValDataset` (has GT; lets us smoke-test + score offline).
    `orig_size` is PIL `(W, H)` so predicted masks can be resized back exactly.
    """

    def __init__(self, data_root: str | Path, split: str, transform) -> None:
        if split == "test":
            self.base = TestDataset(data_root, transform)
            self._kind = "test"
        elif split == "val":
            self.base = ValDataset(data_root, transform)
            self._kind = "val"
        else:
            raise ValueError(f"unknown inference split: {split!r}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if self._kind == "test":
            image, name, orig_size = item
        else:  # val 5-tuple: (image, label, seg_id, name, orig_size)
            image, _label, _seg, name, orig_size = item
        return image, name, orig_size


def infer_collate(batch):
    """Stack images; keep names + orig sizes as plain lists (variable per image)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    names = [b[1] for b in batch]
    orig_sizes = [b[2] for b in batch]
    return images, names, orig_sizes


# --- model loading ------------------------------------------------------------
def load_cls_model(ckpt_path: str | Path, device):
    """Rebuild the cls backbone from a `train_cls` checkpoint and load it strict.

    The checkpoint's `args["model"]` names the size factory; weights load strict
    (the `head` is included — this is a fully-trained cls model, not an FCMAE
    encoder warm-start).
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    margs = ck.get("args", {})
    model_name = margs.get("model", "atto")
    factory, _dims = MODELS[model_name]
    model = factory(num_classes=300).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    log.info("loaded cls model=%s from %s (epoch %s)",
             model_name, ckpt_path, ck.get("epoch"))
    return model, margs


@torch.no_grad()
def predict_cls(model, loader, device, use_amp: bool = True) -> dict[str, int]:
    """Predict `class_id` (0..299) for every image; returns `{name: class_id}`."""
    model.eval()
    preds: dict[str, int] = {}
    for images, names, _orig in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        cls = logits.argmax(dim=1).cpu().tolist()
        for name, c in zip(names, cls):
            preds[name] = int(c)
    return preds


def get_args():
    p = argparse.ArgumentParser(description="Predict class_id for test images")
    p.add_argument("--cls-ckpt", required=True, help="checkpoint from train_cls")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--output", default="submission_cls.csv")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
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

    model, _margs = load_cls_model(args.cls_ckpt, device)
    preds = predict_cls(model, loader, device, use_amp)

    # Standalone: segmentation column is all-background ("0"); test_seg merges masks.
    rows = [{"image": name, "class_id": cid, "segmentation_rle": "0"}
            for name, cid in preds.items()]
    write_submission(rows, args.output)
    log.info("wrote %d cls rows -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
