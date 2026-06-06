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
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import v2


from .utils import rgb_to_seg, project_root

# --- Transform constants -----------------------------------------------------
# Plain arithmetic normalization (scale to [0,1] -> map to [-1,1]). NOT external
# stats: swap in channel mean/std computed from the competition data if desired.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def load_metadata(data_root: Path, name: str) -> list[dict]:
    """Load one of the data/metadata/*.json manifests."""
    with open(data_root.resolve() / "metadata" / f"{name}.json") as f:
        return json.load(f)


class Cse164Dataset(Dataset):
    """big class, given dataset root, create datasets"""
    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        self.data_root: Path = Path(data_root)
        self.transform: v2.Transform = transform
        self.items: list[dict]
        self.img_names: list[str]

    def __len__(self) -> int:
        return len(self.items)


class TrainLabeledDataset(Cse164Dataset):
    """
    turn directory of labeled images into a DataSet
    
    image, label
    """
    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        super().__init__(data_root=data_root, transform=transform)
        self.items: list[dict] = load_metadata(self.data_root, "train_labeled")
        self.img_names = []
        for i in self.items:
            i["image"] = Path(data_root) / i["image"]
            self.img_names.append(i["image"].name)

    def __getitem__(self, idx: int):
        """
        for training cls labeled set returns: image, label, filename
        """
        item = self.items[idx]
        image = Image.open(item["image"]).convert("RGB")
        label = int(item["class_id"])
        name = Path(item["image"]).name
        return self.transform(image), torch.tensor(label, dtype=torch.long) # , name


class TrainMaskedDataset(Cse164Dataset):
    """
    turn directory of segmentation masked images into a Dataset
    
    image, mask, segmentation_label
    """

    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        super().__init__(data_root=data_root, transform=transform)
        self.items: list[dict] = load_metadata(self.data_root, "train_seg")
        self.img_names = []
        for i in self.items:
            i["image"] = Path(data_root) / i["image"]
            self.img_names.append(Path(i["image"]).name)

    def __getitem__(self, idx: int):
        """
        for training seg masked set returns: image, mask, SEGMENTATION label, filename

        note that metadata comes with the foreground class encoded as both class_id and segmentation_id
        to better separate that this is in fact segmentation data - will extract segmentation_id
        """
        item = self.items[idx]
        image = Image.open(item["image"]).convert("RGB")
        mask_rgb = np.array(Image.open(self.data_root / item["mask"]).convert("RGB"))
        mask = rgb_to_seg(mask_rgb)
        label = int(item["segmentation_id"])
        name = Path(item["image"]).name
        image, mask = self.transform(image, mask)
        return image, mask, torch.tensor(label, dtype=torch.long) # , name


class TrainUnlabeledDataset(Cse164Dataset):
    """
    turn directory of unlabeled images into a Dataset
    
    image
    """
    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        super().__init__(data_root=data_root, transform=transform)
        images = sorted((self.data_root / "train_unlabeled" / "images").glob("*.JPEG"))
        self.items = []
        self.img_names = []
        for i in images:
            self.items.append({"image" : i})
            self.img_names.append(Path(i).name)

    def __getitem__(self, idx: int):
        """
        for unlabeled returns: image, filename
        """
        item = self.items[idx]
        path = item["image"]
        image = Image.open(path).convert("RGB")
        return self.transform(image) # , path.name


class ValDataset(Cse164Dataset):
    """
    turn directory of validation images into a Dataset
    
    image, mask, class_label
    """
    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        super().__init__(data_root=data_root, transform=transform)
        
        with open(self.data_root / "val" / "classification.json") as f:
            self.items = json.load(f)
        
        self.img_names = [Path(i["image"]).name for i in self.items]

    def __getitem__(self, idx: int):
        """
        for validation set returns: image, segmentation mask, CLASSIFIER label, name, original size
        """
        item = self.items[idx]
        name = item["image"]
        image = Image.open(self.data_root / "val" / "images" / name).convert("RGB")
        orig_size = image.size  # (W, H) -- needed to resize predictions back
        mask_path = self.data_root / "val" / "masks" / f"{Path(name).stem}.png"
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        seg_mask = rgb_to_seg(mask_rgb)
        cls_label = int(item["class_id"])

        image = self.transform(image)
        seg_mask = torch.as_tensor(seg_mask, dtype=torch.long)
        return image, seg_mask, torch.tensor(cls_label, dtype=torch.long), torch.tensor(orig_size) # , name


class TestDataset(Cse164Dataset):
    """
    turn directory of test images into a Dataset
    
    image, original_size
    """

    def __init__(self, data_root: str | Path, transform: v2.Transform) -> None:
        super().__init__(data_root=data_root, transform=transform)
        images = sorted((self.data_root / "test" / "images").glob("*.JPEG"))
        self.items = []
        self.img_names = []
        for i in images:
            self.items.append({"image": i})
            self.img_names.append(Path(i).name)

    def __getitem__(self, idx: int):
        """
        for test set returns: image, filename, original size
        """
        item = self.items[idx]
        path = item["image"]
        image = Image.open(path).convert("RGB")
        orig_size = image.size  # (W, H) -- needed to resize predictions back
        return self.transform(image), torch.tensor(orig_size) # , path.name

class PretrainDataset(Cse164Dataset):
    def __init__(self, data_root: str | Path, transform: v2.Transform, sets: list[Cse164Dataset]) -> None:
        super().__init__(data_root, transform)
        self.img_names = []
        self.items = []
        for p in sets:
            self.img_names += p.img_names
            self.items += p.items

    def __getitem__(self, idx: int):
        """
        for unlabeled returns: image, filename
        """
        item = self.items[idx]
        path = item["image"]
        image = Image.open(path).convert("RGB")
        return self.transform(image) # , path.name