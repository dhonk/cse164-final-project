"""
dataset.py: load images from directory and prepare to be used by project pipeline

dataset file structure (from kaggle competition)

    data/
    ├── metadata/
    │   ├── class_map.json
    │   ├── train_labeled.json
    │   └── train_seg.json
    ├── test/images/
    ├── train_labeled/images/
    ├── train_seg/
    │   ├── images/
    │   └── masks/
    ├── train_unlabeled/images/
    └── val/
        ├── images/
        ├── masks/
        └── classification.json

    train_labeled   -> TrainLabeledDataset    image + class_id           (7,500)
    train_seg       -> TrainSegDataset        image + class_id + mask    (3,000)
    train_unlabeled -> TrainUnlabeledDataset  image                      (50,000)
    val             -> ValDataset             image + class_id + mask    (750)
    test            -> TestDataset            image                      (3,000)

"""



from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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


class ImageOnlyDataset(Dataset):
    """Strip any split down to image tensors only, for FCMAE pre-training.

    The image is ALWAYS element 0 of every split's ``__getitem__`` return
    (TrainUnlabeled ``(image, name)``, TrainLabeled ``(image, label, name)``,
    TrainSeg ``(image, label, seg_id, name)``), so taking ``[0]`` works
    uniformly. FCMAE wants pixels only -- no labels/masks -- and a uniform
    image-only item lets the default collate stack a clean ``(N, 3, C, C)``
    batch (mixed-length tuples would break it).
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.dataset[idx][0]


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

def build_pretrain_loader(
    data_root: str | Path = "./data",
    batch_size: int = 64,
    crop_size: int = 224,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Pooled image-only loader for FCMAE pre-training (~60.5k images).

    Concatenates the three TRAINING splits with labels/masks stripped:
    ``train_unlabeled`` (50k) + ``train_labeled`` (7.5k) + ``train_seg`` (3k).
    val/test are deliberately walled off (never pooled) to avoid leakage. Every
    item is a ``(3, C, C)`` float tensor with ``C % 32 == 0`` (default 224) so
    FCMAE's patch/mask grid divides evenly. Each split is wrapped in
    ``ImageOnlyDataset`` -- including ``TrainUnlabeledDataset`` so the concat
    yields uniform image-only items the default collate can stack.

    NOTE: train_seg still reads its mask PNG (SegTransform needs it for the joint
    geometric crop) and we discard it -- a small, acceptable I/O cost over 3k imgs.
    """
    img_tf = ImageTransform(crop_size, train=True)
    seg_tf = SegTransform(crop_size, train=True)
    pool = ConcatDataset(
        [
            ImageOnlyDataset(TrainUnlabeledDataset(data_root, img_tf)),
            ImageOnlyDataset(TrainLabeledDataset(data_root, img_tf)),
            ImageOnlyDataset(TrainSegDataset(data_root, seg_tf)),
        ]
    )
    return DataLoader(
        pool,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


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