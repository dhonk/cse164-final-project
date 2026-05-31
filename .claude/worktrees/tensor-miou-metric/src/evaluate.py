"""Inference + submission generation.

For each test image, predict an image-level class_id (0..299) and a pixel mask,
resize the mask back to the ORIGINAL image size with nearest-neighbor, convert
to segmentation ids (foreground class k -> k+1), and RLE-encode it.

Constraints (CLAUDE.md):
    - Predicted ids in 0..300 only; NEVER emit 1000.
    - Mask must match the input image's exact W/H (nearest-neighbor resize).
    - RLE is row-major, 1-indexed, non-overlapping triples.
    - Always run starter/validate_submission_csv.py before submitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from dataset import TestDataset, ValDataset, build_transforms
from model import build_model
from utils import encode_rle, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


@torch.no_grad()
def predict_one(model, image, orig_size, device):
    """Return (class_id, seg_id_mask resized to orig_size) for a single image."""
    # TODO: forward pass; argmax cls; argmax seg; nearest-neighbor resize to
    #       orig_size; map class indices -> segmentation ids (k -> k+1).
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    device = get_device()

    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device)  # our own training output only
    model.load_state_dict(state)
    model.eval()

    # Submission path treats val like test (predictions only); val's labels are
    # used separately for offline scoring, not here.
    eval_tf = build_transforms("eval")
    if args.split == "val":
        dataset = ValDataset(args.data_root, eval_tf)
    else:
        dataset = TestDataset(args.data_root, eval_tf)

    rows: list[dict] = []
    for sample in dataset:  # TODO: batch with a DataLoader
        if args.split == "val":
            image, _label, _gt_mask, name, orig_size = sample
        else:
            image, name, orig_size = sample
        class_id, seg_mask = predict_one(model, image, orig_size, device)
        rows.append(
            {
                "image": name,
                "class_id": int(class_id),
                "segmentation_rle": encode_rle(seg_mask),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
