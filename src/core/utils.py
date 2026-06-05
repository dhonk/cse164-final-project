"""
utils.py: provide shared helpers and constants

"""


from __future__ import annotations

import argparse
import datetime
import math
import random
import time
import logging
import sys
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

import numpy as np
import torch

from dataclasses import dataclass

NUM_CLASSES = 300
IGNORE_IDX = 1000
BATCH_SIZE = 16
RAND_SEED = 0

CHECKPOINT_DIR = "./checkpoints"
CHKPT_FREQ = 20

SAVE_DIR = "./outputs"
DATA_DIR = "./data"

PRINT_FREQ = 10

@dataclass(frozen=True, slots=True) # TODO: update with any more needed params
class PretrainConfigs:
    """
    Stores important configs for pretraining
    """

    """Model info / Params"""
    model_size: str = "atto" # handles depth, dims 
    # Possible inputs:
    #    "atto" "femto" "pico" "nano"
    #    "tiny" "base" "large" "huge"
    decoder_depth: int = 1
    decoder_embed_dim: int = 512

    # patch_size and mask_ratio should ideally be fixed throughout
    patch_size: int = 32
    mask_ratio: float = 0.6
    norm_pix_loss: bool = False

    """Image parameters info"""
    channels: int = 3
    size: int = 224

    """Run length info"""    
    epochs: int = 800
    warmup_epochs: int = 40
    
    """Learning rate info"""
    min_lr: float = 1e-6    
    blr: float = 1.5e-4

    """Hyperparameters"""
    batch_size: int = BATCH_SIZE
    weight_decay: float = 0.05
    optim_momentum: tuple[float, float] = (0.9, 0.95) # (alpha, beta) values of optimizer momentum

    """Run Info"""
    num_workers: int = 2
    limit: int = 0 # if > 0, use only first `limit` images
    save_every: int = 10 # save the checkpoint every _ epochs
    update_freq: int = 1 # gradient-accumulation steps (effective batch = batch_size * update_freq)
    use_amp: bool = True # use AMP loss-scaling via LossScaler
    viz_every: int = 0 # epochs between show_modeled_image reconstructions (0 = off)

def seed_everything(seed: int = 0) -> None:
    """
    Seed Python / NumPy / Torch RNGs for reproducible runs.
    """

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# from starter code provided through kaggle - kaggle_metric.py:
def encode_mask_ids(mask_ids: np.ndarray) -> str:
    """
    Encode a 2D id mask as row-major 1-indexed RLE triples.

    The encoded string is a space-separated sequence of:

        start length value start length value ...

    Only non-background pixels are stored. `start` is 1-indexed after row-major
    flattening, `length` is the run length, and `value` is the segmentation id.
    """

    flat = np.asarray(mask_ids, dtype=np.int64).reshape(-1)
    nonzero = flat != 0
    if not np.any(nonzero):
        return ""
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


# from starter code provided through kaggle - kaggle_metric.py:
def _is_missing_rle(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# modified starter code provided through kaggle - kaggle_metric.py:
def decode_rle_to_mask(
    rle: object,
    height: int,
    width: int,
    num_classes: int = NUM_CLASSES,
    allow_ignore: bool = False,
) -> np.ndarray:
    """
    Decode row-major RLE triples into a dense segmentation-id mask.
    """
    
    total = int(height) * int(width)
    mask = np.zeros(total, dtype=np.uint16)
    if _is_missing_rle(rle) or str(rle).strip() in {"", "0"}:
        return mask.reshape((int(height), int(width)))

    try:
        tokens = [int(tok) for tok in str(rle).split()]
    except ValueError as exc:
        raise ValueError("segmentation_rle must contain integer tokens only") from exc

    if len(tokens) % 3 != 0:
        raise ValueError("segmentation_rle must contain start length value triples")

    used = np.zeros(total, dtype=bool)
    for start, length, value in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if start < 1 or length < 1:
            raise ValueError("RLE starts and lengths must be positive")
        
        if value < 1 or (value > num_classes and not (allow_ignore and value == IGNORE_IDX)):
            allowed = f"1..{num_classes}" + (f" or {IGNORE_IDX}" if allow_ignore else "")
            raise ValueError(f"RLE values must be in {allowed}")
        
        begin = start - 1
        end = begin + length
        
        if end > total:
            raise ValueError("RLE run extends past the image size")
        
        if used[begin:end].any():
            raise ValueError("RLE runs must not overlap")
        
        used[begin:end] = True
        mask[begin:end] = value
        
    return mask.reshape((int(height), int(width)))


# modified starter code provided through kaggle - kaggle_metric.py:
def cls_metrics(pred: dict[str, int], gt: dict[str, int], num_classes: int = NUM_CLASSES) -> tuple[float, float]:
    """
    calculate accuracy and macro accuracy of classification task given predictions and ground truths.

    args:
        pred (dict[str, int]): dictionary of {filename: label}
        gt   (dict[str, int]): dictionary of {filename: label}
    returns:
        accuracy, macor_accuracy (tuple[float, float]): accuracy, macro_accuracy
    """

    images = sorted(gt)
    correct = np.array([pred.get(image) == gt[image] for image in images], dtype=np.float64)
    accuracy = float(correct.mean()) if len(correct) else 0.0

    per_class = []
    for class_id in range(num_classes):
        class_images = [image for image in images if gt[image] == class_id]
        if class_images:
            per_class.append(float(np.mean([pred.get(image) == class_id for image in class_images])))

    macro_accuracy = float(np.mean(per_class)) if per_class else 0.0

    return accuracy, macro_accuracy


def mask_check(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> bool:
    """
    check that segmentation-id mask only has valid class labels.

    args:
        mask (np.ndarray): the segmentation mask
        num_classes (int): the number of classes
    returns:
        valid (bool): whether or not provided segmentation mask has only valid classes
    """
    valid = np.all((mask >= 0) & (mask <= NUM_CLASSES))
    valid = valid and np.all(mask != IGNORE_IDX) 
    return bool(valid)


# modified starter code provided through kaggle - kaggle_metric.py:
def mask_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """
    create mask confusion matrix

    args:
        pred:        predicted segmentation mask
        gt:          ground truth segmentation mask
        num_classes: number of classes
    returns:
        hist.reshape(num_classes + 1, num_classes + 1)
    """
    pred = np.where((pred >= 0) & (pred <= num_classes), pred, 0)
    valid = gt != IGNORE_IDX
    valid &= gt >= 0
    valid &= gt <= num_classes
    labels = (num_classes + 1) * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    hist = np.bincount(labels, minlength=(num_classes + 1) ** 2)
    return hist.reshape(num_classes + 1, num_classes + 1)


# modified starter code provided through kaggle - kaggle_metric.py:
def iou_from_confusion(hist: np.ndarray, class_ids: list[int]) -> tuple[float, dict[int, float]]:
    """
    get miou from confusion matrix

    args:
        hist:      histogram
        class_ids: ground truth class ids
    returns:
        mean_iou:  mean iou metric
        per_class: per-class iou
    """
    per_class: dict[int, float] = {}
    for class_id in class_ids:
        tp = hist[class_id, class_id]
        fp = hist[:, class_id].sum() - tp
        fn = hist[class_id, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            per_class[class_id] = float(tp / denom)
    mean_iou = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_iou, per_class


# modified starter code provided through kaggle - kaggle_metric.py:
def boundary_map(ids: np.ndarray) -> np.ndarray:
    """
    Mark every pixel that sits on a class boundary

    args:
        ids: segmentation map
    returns:
        boundary: np.ndarray boundary map
    """
    
    valid = ids != IGNORE_IDX
    boundary = np.zeros(ids.shape, dtype=bool)
    boundary[:-1, :] |= (ids[:-1, :] != ids[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[1:, :] |= (ids[:-1, :] != ids[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[:, :-1] |= (ids[:, :-1] != ids[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    boundary[:, 1:] |= (ids[:, :-1] != ids[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    return boundary


# modified starter code provided through kaggle - kaggle_metric.py:
def dilate_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Given a boundary map, apply dialation by radius

    args:
        mask:   a boundary map
        radius: radius to dialate by (default 2)
    returns:
        result: the dialated boundary map
    """
    result = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y_src_start = max(0, -dy)
            y_src_end = mask.shape[0] - max(0, dy)
            x_src_start = max(0, -dx)
            x_src_end = mask.shape[1] - max(0, dx)
            y_dst_start = max(0, dy)
            y_dst_end = mask.shape[0] - max(0, -dy)
            x_dst_start = max(0, dx)
            x_dst_end = mask.shape[1] - max(0, -dx)
            result[y_dst_start:y_dst_end, x_dst_start:x_dst_end] |= mask[
                y_src_start:y_src_end,
                x_src_start:x_src_end,
            ]
    return result


# modified starter code provided through kaggle - kaggle_metric.py:
def boundary_f_score(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> float:
    """
    Given predicted and ground truth segmentation maps, apply dilation by radius 
    and calculate f-score.

    args:
        pred:    predicted segmentation map
        gt:      ground truth segmentation map
        radius:  radius to dilate by (default = 2)
    returns:
        f-score: the f score of the predicted boundary map
    """
    
    pred = pred.copy()
    pred[gt == IGNORE_IDX] = IGNORE_IDX
    pred_boundary = boundary_map(pred)
    gt_boundary = boundary_map(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    pred_match = pred_boundary & dilate_boundary(gt_boundary, radius)
    gt_match = gt_boundary & dilate_boundary(pred_boundary, radius)
    precision = pred_match.sum() / max(1, pred_boundary.sum())
    recall = gt_match.sum() / max(1, gt_boundary.sum())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# modified starter code provided through kaggle - validate_submission.py:
def rgb_to_seg(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Decode an (H, W, 3) RGB mask into (H, W) segmentation labels.
    
    args:
        mask_rgb: (H, W, 3) RGB mask
    returns:
        (H, W) segementation-id array
    """

    r = mask_rgb[..., 0].astype(np.int64)
    g = mask_rgb[..., 1].astype(np.int64)
    return r + g * 256


def label_seg_to_cls(seg_label: int) -> int:
    """
    Map a foreground segmentation class label to a classification label

    Foreground seg label (1 ... 300) -> cls label (0 ... 299).
    Special rules: background label (0) IGNORE_IDX (1000) ignored

    args:
        seg_label: a segmentation label
    returns:
        cls_label: a classification label
    """
    if (seg_label > 0) and (seg_label <= 300):
        return seg_label - 1
    return IGNORE_IDX


def label_cls_to_seg(cls_label: int) -> int:
    """
    Map a classification label to a segmentation foreground class label.

    cls label (0 ... 299) -> foreground seg label (1 ... 300).
    special rules: invalid labels get treated with IGNORE_IDX
    
    args:
        cls_label: a classification label
    return
        seg_fg_label: a segmentation foreground label
    """
    if (cls_label >= 0) and (cls_label < 300):
        return cls_label + 1
    return IGNORE_IDX


def mask_seg_to_cls(seg_mask: np.ndarray) -> np.ndarray:
    """
    Map a segmentation mask to an image-level class label mask.

    Foreground seg id k (1..300) -> class id k-1 (0..299); background (0) and
    ignore (1000) -> IGNORE_IDX so a CrossEntropy loss skips them.

    NOTE: realistically the program should never use this!

    args:
        seg_mask: a segmentation mask
    returns:
        cls_mask: a segmentation mask translated into class labels 
    """

    cls_mask = np.full(seg_mask.shape, IGNORE_IDX, dtype=np.int64)
    fg = (seg_mask > 0) & (seg_mask <= NUM_CLASSES)
    cls_mask[fg] = seg_mask[fg] - 1
    return cls_mask


def mask_cls_to_seg(cls_mask: np.ndarray) -> np.ndarray:
    """
    Map a class label mask to a segmentation mask.

    Foreground seg id k (1..300) -> class id k-1 (0..299); background (0) and
    ignore (1000) -> IGNORE_IDX so a CrossEntropy loss skips them.

    NOTE: realistically the program should never use this!

    NOTE: inference does NOT need this -- the model's seg head emits seg ids
    directly (301 channels)

    NOTE: this also doesn't behave properly: inability to actually generate background!

    args:
        cls_mask: a segmentation mask
    returns:
        seg_mask: a segmentation mask translated into class labels 
    """
    
    return np.asarray(cls_mask, dtype=np.int64) + 1


def check_cuda() -> bool:
    """
    check if cuda is available on this device for torch to use
    
    returns:
        True if cuda is available, False if cpu
    """
    return torch.cuda.is_available()


def get_device() -> torch.Device:
    """
    get the available device for pytorch to use.

    returns:
        available torch device
    """

    import torch
    return torch.device("cuda" if check_cuda() else "cpu")
    

def project_root() -> Path:
    """
    Returns the project root directory as a pathlib.Path
    """
    return Path(__file__).parents[2].resolve()


# next section has convnext utils
# everything up is very good. below is... not as good

# taken from ConvNeXt-V2 -> https://github.com/facebookresearch/ConvNeXt-V2/
def str2bool(v) -> bool:
    """
    convert string to bool type for command argument parsing

    args:
        v: a string
    returns:
        a bool corresponding to the string
    raises:
        argparse.ArgumentTypeError if an invalid input is given
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

logging.basicConfig(
    filename = project_root() / "logs" / (str(datetime.date.today()) + ".log"),
    format = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt = "%H:%M:%S"
)

logging.info("Started")