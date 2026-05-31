"""
Datasets and Transforms

This module owns all data loading for the five splits and the PNG<->tensor side of
the segmentation masks. It deliberately does **not** touch RLE encoding/decoding —
that contract lives in ``starter/kaggle_metric.py`` and is imported by the
inference/submission code instead.

Segmentation target encoding (confirmed design):

    target pixel = segmentation_id directly
        0          -> background
        1..300     -> foreground, where segmentation_id = class_id + 1
        1000       -> ignore region (ground-truth only) -> IGNORE_INDEX

The model therefore has ``NUM_SEG_CLASSES`` (301) output channels and ``argmax``
maps straight back to submission segmentation ids. No manual +-1 happens here:
keeping the raw seg-ids as targets *is* the ``class_id k <-> seg_id k+1`` mapping.

No pretrained weights or external data are involved anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NUM_CLASSES = 300       # image-level classes, 0..299
NUM_SEG_CLASSES = 301   # 0=background, 1..300 foreground (= class_id + 1)
RAW_IGNORE_ID = 1000    # ground-truth-only ignore id found in mask PNGs
IGNORE_INDEX = 255      # value handed to CrossEntropyLoss(ignore_index=...)

# Trained from scratch, so there is no ImageNet stat to honour; use simple
# symmetric normalization. Swap for dataset-computed stats later if desired.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

IMAGE_EXTS = {".jpg", ".jpeg"}


# --------------------------------------------------------------------------- #
# Low-level helpers (reused by inference code)
# --------------------------------------------------------------------------- #

def read_json(path: str | Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB (handles grayscale / CMYK / palette JPEGs)."""
    with Image.open(path) as img:
        return img.convert("RGB")


def decode_mask_to_seg_ids(path: str | Path) -> np.ndarray:
    """Decode an RGB-encoded mask PNG into a 2D array of segmentation ids.

    ``segmentation_id = R + G * 256``. Returns an ``int32`` ``[H, W]`` array whose
    values are in ``{0..300, 1000}``. No remapping is applied here.
    """
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    return arr[..., 0] + arr[..., 1] * 256


def seg_ids_to_train_target(seg_ids: np.ndarray, validate: bool = True) -> np.ndarray:
    """Remap raw seg-ids to a training target: ``1000 -> IGNORE_INDEX``, rest kept.

    Values ``0..300`` are already the training class indices, so they pass through
    untouched. ``IGNORE_INDEX`` (255) sits above the valid foreground range and is
    excluded from the loss via ``ignore_index``.
    """
    target = seg_ids.astype(np.int64, copy=True)
    target[target == RAW_IGNORE_ID] = IGNORE_INDEX
    if validate:
        bad = (target > NUM_CLASSES) & (target != IGNORE_INDEX)
        if bad.any():
            offending = np.unique(target[bad])[:8]
            raise ValueError(
                f"mask contains unexpected segmentation ids {offending.tolist()} "
                f"(expected 0..{NUM_CLASSES} or {RAW_IGNORE_ID})"
            )
    return target


def list_images(images_dir: str | Path) -> list[str]:
    """Sorted list of image filenames in a directory (deterministic ordering)."""
    images_dir = Path(images_dir)
    return sorted(
        p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
# Geometric ops are applied jointly to image + mask via torchvision v2, which
# dispatches by tv_tensor type: masks use nearest interpolation automatically and
# are skipped by photometric/normalize ops. The 1000->IGNORE_INDEX remap is done
# *after* geometry so nearest resampling never invents fractional ids.


class TrainSegTransform:
    """Joint image+mask augmentation for supervised segmentation training."""

    def __init__(
        self,
        crop_size: int = 256,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        scale: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        self.geom = v2.Compose(
            [
                v2.RandomResizedCrop(crop_size, scale=scale, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.photo = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Normalize(list(mean), list(std))]
        )

    def __call__(self, image: Image.Image, seg_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        img = tv_tensors.Image(v2.functional.pil_to_tensor(image))
        mask = tv_tensors.Mask(torch.from_numpy(seg_ids.astype(np.int64)))
        img, mask = self.geom(img, mask)
        img = self.photo(img)
        img = self.normalize(img)
        target = torch.as_tensor(mask, dtype=torch.long).clone()
        target[target == RAW_IGNORE_ID] = IGNORE_INDEX
        return img, target


class TrainImageTransform:
    """Image-only augmentation for classification / unlabeled training."""

    def __init__(
        self,
        crop_size: int = 256,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        scale: tuple[float, float] = (0.4, 1.0),
    ) -> None:
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(crop_size, scale=scale, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(list(mean), list(std)),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(tv_tensors.Image(v2.functional.pil_to_tensor(image)))


class EvalImageTransform:
    """Deterministic image preprocessing for val / test (resize to model input).

    Masks are intentionally handled outside this transform so they can be kept at
    full resolution for metric-correct scoring; only the image is resized here.
    """

    def __init__(
        self,
        eval_size: int = 256,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
    ) -> None:
        self.transform = v2.Compose(
            [
                v2.Resize((eval_size, eval_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(list(mean), list(std)),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(tv_tensors.Image(v2.functional.pil_to_tensor(image)))


def make_transforms(
    split: str,
    crop_size: int = 256,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
):
    """Factory returning the appropriate transform for a split.

    ``split`` in ``{"seg", "cls", "unlabeled", "eval"}``.
    """
    if split == "seg":
        return TrainSegTransform(crop_size, mean, std)
    if split in {"cls", "unlabeled"}:
        return TrainImageTransform(crop_size, mean, std)
    if split == "eval":
        return EvalImageTransform(crop_size, mean, std)
    raise ValueError(f"unknown split {split!r}")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class LabeledClassificationDataset(Dataset):
    """Image-level labels from ``metadata/train_labeled.json``.

    Each record is ``{class_id, image}`` with ``image`` relative to ``data_root``.
    """

    def __init__(self, data_root: str | Path, meta_json: str | Path, transform: Callable):
        self.data_root = Path(data_root)
        self.records = read_json(meta_json)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        image = load_image(self.data_root / rec["image"])
        return self.transform(image), int(rec["class_id"])


class SegmentationDataset(Dataset):
    """Pixel labels from ``metadata/train_seg.json``.

    Each record is ``{class_id, image, mask, segmentation_id}`` with paths relative
    to ``data_root``. Returns ``(image, mask_long)`` or, with ``return_class``,
    ``(image, mask_long, class_id)``.
    """

    def __init__(
        self,
        data_root: str | Path,
        meta_json: str | Path,
        transform: TrainSegTransform,
        return_class: bool = False,
    ):
        self.data_root = Path(data_root)
        self.records = read_json(meta_json)
        self.transform = transform
        self.return_class = return_class

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = load_image(self.data_root / rec["image"])
        seg_ids = decode_mask_to_seg_ids(self.data_root / rec["mask"])
        image_t, mask_t = self.transform(image, seg_ids)
        if self.return_class:
            return image_t, mask_t, int(rec["class_id"])
        return image_t, mask_t


class UnlabeledDataset(Dataset):
    """Unlabeled images from ``train_unlabeled/images/`` (distractors NOT filtered).

    With ``return_two_views`` returns two independently augmented views of the same
    image for consistency-based SSL; otherwise a single augmented view.
    """

    def __init__(
        self,
        images_dir: str | Path,
        transform: Callable,
        return_two_views: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.filenames = list_images(self.images_dir)
        self.transform = transform
        self.return_two_views = return_two_views

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        image = load_image(self.images_dir / self.filenames[idx])
        if self.return_two_views:
            return self.transform(image), self.transform(image)
        return self.transform(image)


class ValDataset(Dataset):
    """Public validation split: image-level + pixel labels, for model selection.

    Joins ``val/classification.json`` (``{class_id, image}`` with bare filenames)
    with masks at ``val/masks/<stem>.png``. The image is resized for the model, but
    the mask is kept at **original resolution** and ``orig_hw`` is returned so
    ``evaluate.py`` can upsample predictions back and score at full res.

    Returns ``(image, mask_long, class_id, orig_hw, filename)``. Never train on this.
    """

    def __init__(
        self,
        data_root: str | Path,
        classification_json: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        eval_size: int = 256,
    ):
        self.data_root = Path(data_root)
        self.val_dir = self.data_root / "val"
        if classification_json is None:
            classification_json = self.val_dir / "classification.json"
        self.records = read_json(classification_json)
        self.transform = transform or EvalImageTransform(eval_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        filename = rec["image"]
        stem = Path(filename).stem
        image = load_image(self.val_dir / "images" / filename)
        orig_hw = (image.height, image.width)
        seg_ids = decode_mask_to_seg_ids(self.val_dir / "masks" / f"{stem}.png")
        mask = torch.from_numpy(seg_ids_to_train_target(seg_ids))
        return self.transform(image), mask, int(rec["class_id"]), orig_hw, filename


class TestDataset(Dataset):
    """Hidden test images. Returns ``(image, orig_hw, filename)`` for inference.

    ``orig_hw`` lets the predictor resize masks back to exact input dims with
    nearest-neighbor before RLE encoding.
    """

    def __init__(self, images_dir: str | Path, transform: Optional[Callable] = None, eval_size: int = 256):
        self.images_dir = Path(images_dir)
        self.filenames = list_images(self.images_dir)
        self.transform = transform or EvalImageTransform(eval_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = load_image(self.images_dir / filename)
        orig_hw = (image.height, image.width)
        return self.transform(image), orig_hw, filename


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #

def eval_collate(batch: list[tuple]) -> tuple:
    """Collate for ValDataset / TestDataset.

    Stacks the (fixed-size) image tensors but keeps per-sample ``orig_hw``,
    ``filename`` (and full-res masks for val) as plain lists, since those vary in
    size across the batch.
    """
    if len(batch[0]) == 5:  # ValDataset
        images, masks, class_ids, orig_hws, filenames = zip(*batch)
        return (
            torch.stack(images, 0),
            list(masks),
            torch.tensor(class_ids, dtype=torch.long),
            list(orig_hws),
            list(filenames),
        )
    # TestDataset
    images, orig_hws, filenames = zip(*batch)
    return torch.stack(images, 0), list(orig_hws), list(filenames)
