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
    """Map segmentation ids to a training label map.

    Foreground class k lives at seg_id k+1; background (0) and ignore (1000)
    map to `ignore_index` so they are excluded from the loss.
    """
    # TODO: implement (e.g. background -> 0 or ignore, foreground -> seg_id - 1)
    raise NotImplementedError

def class_id_to_seg_id(class_id: np.ndarray) -> np.ndarray:
    """Inverse of `seg_id_to_class_id` for building submission masks."""
    # TODO: foreground class k -> seg_id k+1, background -> 0
    raise NotImplementedError


def encode_rle(seg_mask: np.ndarray) -> str:
    """Encode an (H, W) segmentation-id mask as row-major 1-indexed RLE triples.

    See starter/kaggle_metric.py::encode_mask_ids for the exact format.
    """
    # TODO: implement (or reuse starter.kaggle_metric.encode_mask_ids)
    raise NotImplementedError


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
