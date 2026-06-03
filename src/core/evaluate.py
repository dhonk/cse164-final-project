"""
Evaluation & submission writing.

We DON'T reinvent the scorer -- ``starter/kaggle_metric.py`` is the reference
implementation the leaderboard uses, so we import and call it directly. This
module is the thin glue around it:

    write_submission(rows, path)   -> CSV ``image,class_id,segmentation_rle``
    score_val(preds, data_root)    -> dict of mIoU / boundary-F / rare-mIoU /
                                      macro-acc (for OFFLINE model selection)
    resize_mask_nearest(mask, sz)  -> resize a predicted seg-id mask back to the
                                      image's ORIGINAL (W,H) with nearest-neighbor

Prediction-row contract (shared by both entry points). ``rows``/``preds`` is
either a list of row dicts or a dict keyed by image filename; each row carries:

    image            test/val filename (e.g. "val_00000.JPEG")
    class_id         predicted image-level class in 0..299
    segmentation_rle OR seg_mask  -- supply ONE:
        segmentation_rle : an already-encoded RLE string, OR
        seg_mask         : a 2D seg-id array (0/1..300) at the image's ORIGINAL
                           resolution, which we encode via core.utils.encode_rle.

Predictions must already be at original resolution (use ``resize_mask_nearest``)
and must NEVER contain id 1000 (ignore is ground-truth-only). The val GT masks
are read at their native resolution and encoded with ignore (1000) preserved, so
the scorer can skip those pixels.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import encode_rle, rgb_to_seg_id

SUBMISSION_COLUMNS = ["image", "class_id", "segmentation_rle"]


# --- reference scorer (starter/kaggle_metric.py, imported, never edited) ------
@lru_cache(maxsize=1)
def _kaggle_metric():
    """Load starter/kaggle_metric.py by file path (it's not on the src package).

    Loaded lazily + cached so importing this module never hard-depends on the
    starter file location at import time. The starter module guards its own
    ``kaggle_metric_utilities`` import, so it runs standalone here.
    """
    path = Path(__file__).resolve().parents[2] / "starter" / "kaggle_metric.py"
    spec = importlib.util.spec_from_file_location("kaggle_metric", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --- prediction helpers -------------------------------------------------------
def resize_mask_nearest(mask: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
    """Resize a (H,W) seg-id mask to original (W,H) with nearest-neighbor.

    ``orig_size`` is ``(W, H)`` -- the PIL ``.size`` convention the datasets
    return. Nearest is mandatory: bilinear would invent fractional/blended class
    ids. No-op if the mask is already the target size.
    """
    w, h = orig_size
    mask = np.asarray(mask)
    if mask.shape == (h, w):
        return mask
    # mode "I" = 32-bit signed int; NEAREST preserves the exact ids.
    img = Image.fromarray(mask.astype(np.int32), mode="I").resize((w, h), Image.NEAREST)
    return np.asarray(img, dtype=np.int64)


def _normalize_rows(rows) -> list[dict]:
    """Coerce rows/preds into a list of ``{image, class_id, segmentation_rle}``.

    Accepts a dict keyed by image filename or any iterable of row dicts; encodes
    ``seg_mask`` via ``encode_rle`` when no ``segmentation_rle`` is supplied.
    """
    items = [{"image": k, **v} for k, v in rows.items()] if isinstance(rows, dict) else list(rows)
    out: list[dict] = []
    for r in items:
        rle = r.get("segmentation_rle")
        if rle is None:
            rle = encode_rle(r["seg_mask"])
        out.append({"image": str(r["image"]), "class_id": int(r["class_id"]), "segmentation_rle": rle})
    return out


# --- submission writing -------------------------------------------------------
def write_submission(rows, path: str | Path) -> Path:
    """Write a ``submission.csv`` (``image,class_id,segmentation_rle``).

    See the module docstring for the row contract. All-background masks encode to
    ``"0"`` (never an empty/NaN field). Returns the written path.
    """
    df = pd.DataFrame(_normalize_rows(rows), columns=SUBMISSION_COLUMNS)
    df.to_csv(path, index=False)
    return Path(path)


# --- val scoring (model selection) --------------------------------------------
@lru_cache(maxsize=2)
def _solution_frame(data_root: str, split: str) -> pd.DataFrame:
    """Build (and cache) the GT solution frame for a labeled split.

    Columns: ``image, class_id, height, width, segmentation_rle``. GT masks are
    read at NATIVE resolution; ignore (1000) is preserved in the RLE so the
    scorer skips those pixels. Cached because re-reading 750 masks every epoch of
    model selection is wasteful.
    """
    root = Path(data_root)
    with open(root / split / "classification.json") as f:
        items = json.load(f)
    rows = []
    for it in items:
        name = it["image"]
        mask_rgb = np.array(Image.open(root / split / "masks" / f"{Path(name).stem}.png").convert("RGB"))
        seg = rgb_to_seg_id(mask_rgb)
        h, w = seg.shape
        rows.append(
            {
                "image": name,
                "class_id": int(it["class_id"]),
                "height": int(h),
                "width": int(w),
                "segmentation_rle": encode_rle(seg),
            }
        )
    return pd.DataFrame(rows)


def score_val(preds, data_root: str | Path = "./data", split: str = "val") -> dict[str, float]:
    """Score predictions on a labeled split via the reference kaggle metric.

    ``preds`` follows the prediction-row contract (module docstring) and MUST
    cover every image in the split (the scorer rejects missing/extra rows).
    Predictions must already be at original resolution. Returns the
    ``detailed_score`` dict: ``automated_score, segmentation_score,
    classification_macro_accuracy, mean_iou, boundary_f_score, rare_class_miou``.
    """
    km = _kaggle_metric()
    solution = _solution_frame(str(Path(data_root)), split)
    submission = pd.DataFrame(_normalize_rows(preds), columns=SUBMISSION_COLUMNS)
    return km.detailed_score(solution, submission)
