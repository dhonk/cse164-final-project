"""
Datasets

One Dataset class per split. Every __getitem__ also returns the image filename
(handy for logging / debugging / pseudo-label bookkeeping, even if unused):

Dataset Descriptions:
    train_labeled   -> TrainLabeledDataset    image + class_id           (7,500)
    train_seg       -> TrainSegDataset        image + class_id + mask    (3,000)
    train_unlabeled -> TrainUnlabeledDataset  image only (+ distractors) (50,000)
    val             -> ValDataset             image + class_id + mask    (750)
    test            -> TestDataset            image only                 (3,000)

val is fully labeled but for model selection / scoring ONLY -- never train on it.

Return-type contract (every split, always -- no PIL/numpy leaks):
    image     float32 tensor (3, C, C)        C = transform crop/eval size
    label     int                             (splits with class labels)
    seg_id    long tensor                     TrainSeg: (C, C);
                                              Val: ORIGINAL (H, W) for scoring
    name      str                             image filename
    orig_size tuple(W, H)                     Val/Test, for resizing preds back

A ``transform`` is REQUIRED for every dataset (use ``build_transforms(split)``);
there is no raw-PIL fallback. This keeps return types uniform so loader / collate
/ metric code never has to branch on PIL-vs-tensor or numpy-vs-tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2


from .utils import rgb_to_seg_id

# --- Transform constants -----------------------------------------------------
# Plain arithmetic normalization (scale to [0,1] -> map to [-1,1]). NOT external
# stats: swap in channel mean/std computed from the competition data if desired.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# Ignore id for unscored pixels in the provided masks; the seg loss skips them
# via `ignore_index=IGNORE_INDEX`. Valid seg ids are 0 (background) and 1..300
# (foreground), so the sentinel must sit OUTSIDE 0..300 -- 1000 already does, and
# the masks already encode ignore as 1000, so no remap is needed. NOTE: the usual
# seg convention of 255 CANNOT be used here -- it collides with foreground seg id
# 255 (class_id 254), which would silently mistrain that class.
IGNORE_INDEX = 1000


def load_metadata(data_root: Path, name: str) -> list[dict]:
    """Load one of the data/metadata/*.json manifests."""
    with open(Path(data_root) / "metadata" / f"{name}.json") as f:
        return json.load(f)


class TrainLabeledDataset(Dataset):
    """train_labeled: image -> class_id (image-level labels only)."""

    def __init__(self, data_root: str | Path, transform) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.items = load_metadata(self.data_root, "train_labeled")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(self.data_root / item["image"]).convert("RGB")
        label = int(item["class_id"])
        name = Path(item["image"]).name
        return self.transform(image), label, name


class TrainSegDataset(Dataset):
    """train_seg: image -> (class_id, segmentation-id mask)."""

    def __init__(self, data_root: str | Path, transform) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.items = load_metadata(self.data_root, "train_seg")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(self.data_root / item["image"]).convert("RGB")
        mask_rgb = np.array(Image.open(self.data_root / item["mask"]).convert("RGB"))
        seg_id = rgb_to_seg_id(mask_rgb)
        label = int(item["class_id"])
        name = Path(item["image"]).name
        # Joint image/mask transform: geometric ops hit both (nearest on the
        # mask), photometric/normalize hit the image only; ignore stays IGNORE_INDEX.
        # Returns (float32 image (3,C,C), long target (C,C)).
        image, seg_id = self.transform(image, seg_id)
        return image, label, seg_id, name


class TrainUnlabeledDataset(Dataset):
    """train_unlabeled: image only. Contains distractors -- filter before use."""

    def __init__(self, data_root: str | Path, transform) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.paths = sorted((self.data_root / "train_unlabeled" / "images").glob("*.JPEG"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name


class ValDataset(Dataset):
    """val: image -> (class_id, segmentation-id mask). Selection/scoring ONLY.

    Driven by val/classification.json ({class_id, image}); masks live at
    val/masks/<stem>.png. Also returns filename + original size so predictions
    can be resized back and written to a submission CSV for offline scoring.
    """

    def __init__(self, data_root: str | Path, transform) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        with open(self.data_root / "val" / "classification.json") as f:
            self.items = json.load(f)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        name = item["image"]
        image = Image.open(self.data_root / "val" / "images" / name).convert("RGB")
        orig_size = image.size  # (W, H) -- needed to resize predictions back
        mask_path = self.data_root / "val" / "masks" / f"{Path(name).stem}.png"
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        seg_id = rgb_to_seg_id(mask_rgb)
        label = int(item["class_id"])
        # Image is resized by the (eval) transform; the GT mask is deliberately
        # kept at ORIGINAL resolution as a long tensor so scoring stays exact --
        # resize predictions back to orig_size (nearest) before comparing. Masks
        # are not stacked by default collate (sizes vary); keep them in a list.
        image = self.transform(image)
        seg_id = torch.as_tensor(seg_id, dtype=torch.long)
        return image, label, seg_id, name, orig_size


class TestDataset(Dataset):
    """test: image only. Returns (image, filename, original_size) for inference."""

    def __init__(self, data_root: str | Path, transform) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.paths = sorted((self.data_root / "test" / "images").glob("*.JPEG"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        orig_size = image.size  # (W, H) -- needed to resize predictions back
        return self.transform(image), path.name, orig_size


class SegTransform:
    """Joint image+mask transform for segmentation training.

    Geometric ops are applied to BOTH the image and the mask (v2 auto-uses
    nearest-neighbor on a ``tv_tensors.Mask``, so seg ids are never blended);
    photometric ops + normalization touch the IMAGE ONLY. Ignore pixels keep
    their raw id ``IGNORE_INDEX`` (1000) so the loss can skip them.
    """

    def __init__(self, crop_size: int = 256, train: bool = True) -> None:
        if train:
            self.geom = v2.Compose(
                [
                    v2.RandomResizedCrop(crop_size, scale=(0.4, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )
            self.photo = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        else:
            self.geom = v2.Resize((crop_size, crop_size), antialias=True)
            self.photo = v2.Identity()
        # scale=True only rescales the float image; the int Mask is left alone.
        self.normalize = v2.Compose(
            [
                v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
                v2.Normalize(list(NORM_MEAN), list(NORM_STD)),
            ]
        )

    def __call__(self, image: Image.Image, seg_id: np.ndarray):
        img = tv_tensors.Image(v2.functional.pil_to_tensor(image))
        mask = tv_tensors.Mask(torch.from_numpy(seg_id.astype(np.int64)))
        img, mask = self.geom(img, mask)
        img = self.photo(img)
        img = self.normalize(img)
        # Masks already encode ignore pixels as IGNORE_INDEX (1000), which sits
        # outside the valid 0..300 seg-id range, so the loss skips them directly.
        target = torch.as_tensor(mask, dtype=torch.long)
        return img, target


class ImageTransform:
    """Image-only transform for classification / unlabeled / eval splits."""

    def __init__(self, crop_size: int = 256, train: bool = True) -> None:
        ops = (
            [
                v2.RandomResizedCrop(crop_size, scale=(0.4, 1.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
            if train
            else [v2.Resize((crop_size, crop_size), antialias=True)]
        )
        ops += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(list(NORM_MEAN), list(NORM_STD)),
        ]
        self.transform = v2.Compose(ops)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(tv_tensors.Image(v2.functional.pil_to_tensor(image)))


def build_transforms(split: str, crop_size: int = 256):
    """Return the transform for a split.

    ``split`` in ``{"seg", "cls", "unlabeled", "eval"}``. Seg returns a joint
    image+mask callable; the rest return image-only callables. Eval is
    deterministic (resize only, no random aug). Masks for eval are intentionally
    NOT resized here so val scoring can compare at full resolution.
    """
    if split == "seg":
        return SegTransform(crop_size, train=True)
    if split in {"cls", "unlabeled"}:
        return ImageTransform(crop_size, train=True)
    if split == "eval":
        return ImageTransform(crop_size, train=False)
    raise ValueError(f"unknown split for build_transforms: {split!r}")

def build_dataloaders(batch_size: int):
    l_loader = DataLoader(
        dataset=TrainLabeledDataset("./data", build_transforms("cls")),
        batch_size=batch_size,
        num_workers=2,
    )
    seg_loader = DataLoader(
        dataset=TrainSegDataset("./data", build_transforms("seg")),
        batch_size=batch_size,
        num_workers=2,
    )
    ul_loader = DataLoader(
        dataset=TrainUnlabeledDataset("./data", build_transforms("unlabeled")),
        batch_size=batch_size,
        num_workers=2,
    )
    val_loader = DataLoader(
        dataset=ValDataset("./data", build_transforms("eval")),
        batch_size=batch_size,
        num_workers=2,
    )
    test_loader = DataLoader(
        dataset=TestDataset("./data", build_transforms("eval")),
        batch_size=batch_size,
        num_workers=2,
    )

    return l_loader, seg_loader, ul_loader, val_loader, test_loader