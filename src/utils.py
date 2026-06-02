"""Shared helpers: reproducibility, mask <-> id encoding, RLE.

Mask spec (see CLAUDE.md):
    segmentation_id = R + G * 256
    0          -> background / non-target
    1..300     -> foreground, segmentation_id = class_id + 1
    1000       -> ignore region (ground-truth only, never predicted)
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

IGNORE_ID = 1000
NUM_CLASSES = 300

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure root logging with a stdout handler and an optional file handler.

    Idempotent: existing handlers are cleared first so repeated calls (e.g. in a
    notebook or across script re-runs) don't duplicate every line. Pass
    ``log_file`` to also tee output to a per-run logfile. Returns the project
    logger; modules should use ``get_logger(__name__)``.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger("cse164")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger under the ``cse164`` namespace."""
    return logging.getLogger(f"cse164.{name}" if name else "cse164")


def seed_everything(seed: int = 0) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rgb_to_seg_id(mask_rgb: np.ndarray) -> np.ndarray:
    """Decode an (H, W, 3) RGB PNG mask into an (H, W) segmentation-id array."""
    r = mask_rgb[..., 0].astype(np.int64)
    g = mask_rgb[..., 1].astype(np.int64)
    return r + g * 256


def seg_id_to_class_id(seg_id: np.ndarray) -> np.ndarray:
    """Map a segmentation-id map to an image-level class-id label map.

    Foreground seg id k (1..300) -> class id k-1 (0..299); background (0) and
    ignore (1000) -> IGNORE_ID so a CrossEntropy loss skips them. NOT used by the
    seg branch -- the seg head trains directly on seg ids -- but provided for
    completeness / deriving class labels from masks.
    """
    seg_id = np.asarray(seg_id, dtype=np.int64)
    out = np.full(seg_id.shape, IGNORE_ID, dtype=np.int64)
    fg = (seg_id >= 1) & (seg_id <= NUM_CLASSES)
    out[fg] = seg_id[fg] - 1
    return out


def class_id_to_seg_id(class_id: np.ndarray) -> np.ndarray:
    """Inverse of `seg_id_to_class_id`: foreground class k -> seg id k+1.

    NOTE: inference does NOT need this -- the model's seg head emits seg ids
    directly (301 channels whose index == seg id), so ``argmax`` already yields
    0..300. Kept for round-tripping / building masks from class-indexed maps.
    """
    return np.asarray(class_id, dtype=np.int64) + 1


def encode_rle(seg_mask: np.ndarray) -> str:
    """Encode an (H, W) segmentation-id mask as row-major 1-indexed RLE triples.

    Only non-zero (foreground) pixels are stored as ``start length value``
    triples: ``start`` is 1-indexed into the row-major flattened mask, ``length``
    is the run length, and ``value`` is the seg id (1..300). Background (0) is left
    as gaps; an all-background mask encodes to ``"0"`` (NOT an empty string) so the
    submission CSV never contains null/NaN fields. The official decoder
    (kaggle_metric.decode_rle_to_mask) treats "0" and "" identically.
    """
    flat = np.asarray(seg_mask, dtype=np.int64).reshape(-1)
    nonzero = flat != 0
    if not np.any(nonzero):
        return "0"
    idx = np.flatnonzero(nonzero)
    values = flat[idx]

    run_break = np.ones(len(idx), dtype=bool)
    run_break[1:] = (idx[1:] != idx[:-1] + 1) | (values[1:] != values[:-1])
    starts = np.flatnonzero(run_break)
    ends = np.r_[starts[1:], len(idx)]

    parts: list[str] = []
    for start_pos, end_pos in zip(starts, ends):
        start = int(idx[start_pos]) + 1
        length = int(end_pos - start_pos)
        value = int(values[start_pos])
        parts.extend([str(start), str(length), str(value)])
    return " ".join(parts)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
