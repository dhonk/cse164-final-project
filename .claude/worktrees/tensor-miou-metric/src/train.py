"""Training entrypoint.

Multi-task training over the labeled + segmentation splits, optionally with
pseudo-labeled unlabeled data. Use val/ ONLY for model selection.

Loss notes (CLAUDE.md):
    - Segmentation CrossEntropy with ignore_index for 1000 (ignore) pixels.
    - Segmentation dominates the score; boundary-aware loss is a bonus direction.
    - Class-balanced sampling helps rare-class mIoU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import TrainLabeledDataset, TrainSegDataset, ValDataset
from model import build_model
from utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def train_one_epoch(model, loaders, optimizer, device) -> dict[str, float]:
    model.train()
    # TODO: iterate batches, compute cls + seg losses, backprop, return metrics.
    raise NotImplementedError


@torch.no_grad()
def validate(model, val_loader, device) -> dict[str, float]:
    model.eval()
    # TODO: compute mIoU / boundary F / macro-acc on val for model selection.
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # TODO: build datasets/loaders (TrainLabeledDataset, TrainSegDataset, ValDataset), scheduler.
    _ = (TrainLabeledDataset, TrainSegDataset, ValDataset)

    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        # train_one_epoch(...); metrics = validate(...)
        # if metrics["score"] > best: torch.save(model.state_dict(), args.out / "best.pt")
        raise NotImplementedError


if __name__ == "__main__":
    main()
