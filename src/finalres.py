from __future__ import annotations
from collections.abc import Callable
from pathlib import Path

import datetime
import logging
import os
import pandas as pd
import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from .core.utils import *
from .core.dataset import (TestDataset, NORM_MEAN, NORM_STD, build_transform_seg_eval,
                           build_transform_cls_tta)

from .models import convnextv2 as cls
from .models import upernet as seg

from .engines.cls_engine import predict_test as predict_cls
from .engines.seg_engine import predict_test as predict_seg

from .utils import load_checkpoint


# directory to pull final eval checkpoints from 
ckpt_dir = Path(CHECKPOINT_DIR) / "final"

# logger
logger = logging.getLogger(__name__)


def build_cls_model(config: ClsConfigs) -> cls.ConvNeXtV2:
    """
    build_cls_model: pass in configs, get a classifier model
    """
    match config.model_size:
        case "atto":
            builder = cls.atto
        case "femto":
            builder = cls.femto
        case "pico":
            builder = cls.pico
        case "nano":
            builder = cls.nano
        case "tiny":
            builder = cls.tiny
        case "base":
            builder = cls.base
        case "large":
            builder = cls.large
        case "huge":
            builder = cls.huge
    return builder() # TODO: all defaults for now, change later


# TODO
def build_seg_model(config: SegConfigs) -> seg.ConvNeXt_UPerNet:
    """build_cls_model: pass in configs, get a classifier model"""
    match config.model_size:
        case "atto":
            builder = seg.atto
        case "femto":
            builder = seg.femto
        case "pico":
            builder = seg.pico
        case "nano":
            builder = seg.nano
        case "tiny":
            builder = seg.tiny
        case "base":
            builder = seg.base
        case "large":
            builder = seg.large
        case "huge":
            builder = seg.huge
    return builder() # TODO: all defaults for now, change later


def load_cls(model: cls.ConvNeXtV2) -> None:
    """
    load_cls: load checkpoints into model
    """
    ckpts = list(ckpt_dir.glob("checkpoint-clsfinetune-*.pth"))
    latest_ckpt = max(ckpts, key = os.path.getmtime)
    load_results = load_checkpoint(model, latest_ckpt, test = True)

    logger.info(f"Loaded checkpoint: {latest_ckpt} (epoch {load_results.get("epoch", -1)})")
    logger.info("Loaded: %d | Unexpected: %d | Randomized (not found in ckpt): %d | EMA weights: %s",
                load_results.get("loaded", -1),
                load_results.get("unexpected", -1),
                load_results.get("random", -1),
                load_results.get("used_ema", False),)


# TODO
def load_seg(model: seg.ConvNeXt_UPerNet) -> None:
    """
    load_seg: load checkpoints into model
    """
    ckpts = list(ckpt_dir.glob("checkpoint-segfinetune-*.pth"))
    latest_ckpt = max(ckpts, key = os.path.getmtime)
    load_results = load_checkpoint(model, latest_ckpt, test = True)

    logger.info(f"Loaded checkpoint: {latest_ckpt} (epoch {load_results.get("epoch", -1)})")
    logger.info("Loaded: %d | Unexpected: %d | Randomized (not found in ckpt): %d",
                load_results.get("loaded", -1),
                load_results.get("unexpected", -1),
                load_results.get("random", -1),)


def build_transform_cls(config: ClsConfigs) -> v2.Transform:
    return v2.Compose([
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(config.size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean = NORM_MEAN, std = NORM_STD)
    ])

def build_transform_seg(config: SegConfigs) -> v2.Transform:
    """Test transform for segmentation: reuse the dataset eval transform (square
    resize + normalize). hflip TTA needs no special transform -- predict_seg flips
    the batched tensor inside the forward pass and un-flips the logits."""
    return build_transform_seg_eval(config)


def run_cls(device: torch.Device, config: ClsConfigs) -> dict: # TODO: change to cls predict configs...?
    """
    run_cls: run classification on the test set with 10-crop TTA, return
    {image_name: class_id}. TenCrop produces all 10 views per image in a single
    pass and predict_test averages their softmax, so no outer TTA loop is needed.
    """
    logger.info(f"Generating Final Class Predictions | model size - {config.model_size}")

    dataset = TestDataset(DATA_DIR, build_transform_cls_tta(config))
    img_names = dataset.img_names # TODO: check names

    # NOTE: each sample is 10 crops, so the model sees batch_size * 10 images per
    # forward -- lower batch_size if this OOMs.
    loader = DataLoader(
        dataset = dataset,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
        prefetch_factor = 2,
    )

    model = build_cls_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_params / 1e6)

    # load params
    load_cls(model)

    probs = predict_cls(loader, model, device)   # [N, num_classes], TTA-averaged
    outputs = probs.argmax(dim=1).tolist()

    # sanity check
    if len(outputs) != len(img_names):
        logger.critical(f"Number of outputs {len(outputs)} doesn't match number of filenames {len(img_names)}")
        raise ValueError("Number of outputs predicted doesn't match number of filenames.")

    return {img_names[i]: outputs[i] for i in range(len(outputs))}


# TODO
def run_seg(device: torch.Device, config: SegConfigs):
    logger.info(f"Generating Final Segmentation Predictions | model size - {config.model_size}")

    dataset = TestDataset(DATA_DIR, build_transform_seg(config))
    img_names = dataset.img_names # TODO: check names

    # NOTE: predict_seg runs hflip TTA (2 forwards per batch), so the model sees
    # batch_size * 2 images per step -- lower batch_size if this OOMs.
    loader = DataLoader(
        dataset = dataset,
        batch_size = config.batch_size,
        shuffle = False,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
        prefetch_factor = 2,
    )

    model = build_seg_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM", n_params / 1e6)

    # load params
    load_seg(model)

    # Step 1: get pixel-wise labels -- predict_seg returns a list of per-image
    # seg-id maps (np.uint16, ids 0..300) already upsampled to each image's
    # original H x W.
    id_maps = predict_seg(loader, model, device)

    # Step 2: encode each id-map as a row-major 1-indexed RLE string
    # (encode_mask_ids emits "" for an all-background mask).
    outputs = [encode_mask_ids(id_map) for id_map in id_maps]

    # sanity check
    if len(outputs) != len(img_names):
        logger.critical(f"Number of outputs {len(outputs)} doesn't match number of filenames {len(img_names)}")
        raise ValueError("Number of outputs predicted doesn't match number of filenames.")

    return {img_names[i]: outputs[i] for i in range(len(outputs))}


def main():
    setup_logging()
    seed_everything(RAND_SEED)

    device = get_device()

    clsconfig = ClsConfigs() # NOTE: not final
    segconfig = SegConfigs() # NOTE: not final

    cls_outputs = run_cls(device, clsconfig)   # {img_name: class_id}
    seg_outputs = run_seg(device, segconfig)   # {img_name: segmentation_rle}

    # err i'm sure there's a better way to do this but o well
    images = sorted([img.name for img in (Path(DATA_DIR) / "test" / "images").glob("*.JPEG")])
    rows = [{"image": img,
             "class_id": cls_outputs[img],
             "segmentation_rle": seg_outputs[img]} for img in images]

    fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_submission.csv")
    output = Path(SAVE_DIR) / fname
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info(f"Saved output to {fname}")

if __name__ == "__main__":
    main()