"""
utils.py: provide shared helpers

"""


from __future__ import annotations

import argparse
import datetime
import math
import random
import time
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

import numpy as np
import torch


NUM_CLASSES = 300
IGNORE_ID = 1000


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
        
        if value < 1 or (value > num_classes and not (allow_ignore and value == IGNORE_ID)):
            allowed = f"1..{num_classes}" + (f" or {IGNORE_ID}" if allow_ignore else "")
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
    valid = valid and np.all(mask != IGNORE_ID) 
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
    valid = gt != IGNORE_ID
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
    
    valid = ids != IGNORE_ID
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
    pred[gt == IGNORE_ID] = IGNORE_ID
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
    Special rules: background label (0) IGNORE_ID (1000) ignored

    args:
        seg_label: a segmentation label
    returns:
        cls_label: a classification label
    """
    if (seg_label > 0) and (seg_label <= 300):
        return seg_label - 1
    return IGNORE_ID


def label_cls_to_seg(cls_label: int) -> int:
    """
    Map a classification label to a segmentation foreground class label.

    cls label (0 ... 299) -> foreground seg label (1 ... 300).
    special rules: invalid labels get treated with IGNORE_ID
    
    args:
        cls_label: a classification label
    return
        seg_fg_label: a segmentation foreground label
    """
    if (cls_label >= 0) and (cls_label < 300):
        return cls_label + 1
    return IGNORE_ID


def mask_seg_to_cls(seg_mask: np.ndarray) -> np.ndarray:
    """
    Map a segmentation mask to an image-level class label mask.

    Foreground seg id k (1..300) -> class id k-1 (0..299); background (0) and
    ignore (1000) -> IGNORE_ID so a CrossEntropy loss skips them.

    NOTE: realistically the program should never use this!

    args:
        seg_mask: a segmentation mask
    returns:
        cls_mask: a segmentation mask translated into class labels 
    """

    cls_mask = np.full(seg_mask.shape, IGNORE_ID, dtype=np.int64)
    fg = (seg_mask > 0) & (seg_mask <= NUM_CLASSES)
    cls_mask[fg] = seg_mask[fg] - 1
    return cls_mask


def mask_cls_to_seg(cls_mask: np.ndarray) -> np.ndarray:
    """
    Map a class label mask to a segmentation mask.

    Foreground seg id k (1..300) -> class id k-1 (0..299); background (0) and
    ignore (1000) -> IGNORE_ID so a CrossEntropy loss skips them.

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
    return Path(__file__).resolve().parents[1].parents[1]


# next section has convnext utils
# TODO: refactor as convnext gets adopted!

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


# taken from ConvNeXt-V2 -> https://github.com/facebookresearch/ConvNeXt-V2/
# TODO: UPDATE AS CONVNEXT CODE GETS REFACTORED
class SmoothedValue(object):
    """
    Tracks a series of values with a windowed median/avg and a global avg.

    attributes:
        deque: stores the individual values
        total: stores the sum of values
        count: stores the number of values
        fmt:   the format string
    """

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n: int = 1) -> None:
        """
        Update stored values
        
        args:
            value: the value
            n:     number of ocurrences (default = 1)
        returns:
            None 
        """
        
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """
        currently building for single gpu, kept for matching
        """
        return None

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg,
                               global_avg=self.global_avg, max=self.max, value=self.value)


# taken from ConvNeXt-V2 -> https://github.com/facebookresearch/ConvNeXt-V2/
# TODO: UPDATE AS CONVNEXT CODE GETS REFACTORED
class MetricLogger:
    """Aggregates named SmoothedValues and prints periodic progress with ETA."""

    def __init__(self, delimiter: str = "  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):  # single-GPU no-op (kept for API parity)
        return

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_parts = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}",
                     "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            log_parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_parts)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                                         time=str(iter_time), data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                                         time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


# taken from ConvNeXt-V2 -> https://github.com/facebookresearch/ConvNeXt-V2/ 
def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device # type: ignore
    if norm_type == math.inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters) # type: ignore
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), # type: ignore
        norm_type)


class NativeScaler:
    """AMP loss scaler with optional grad clipping; returns the grad norm."""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        norm = None
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, epoch, args):
    """Half-cycle cosine LR decay after a linear warmup (per-iteration `epoch`).

    `epoch` may be fractional (step / steps_per_epoch + epoch). Each param group's
    LR is scaled by its ``lr_scale`` (set by the layer-decay assigner) if present.
    Requires args.lr, args.min_lr, args.warmup_epochs, args.epochs.
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr


def save_checkpoint(path, model, optimizer=None, scaler=None, epoch=None, extra=None):
    """Save {model, [optimizer], [scaler], epoch, **extra} to ``path``."""
    ckpt = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if extra:
        ckpt.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, model, optimizer=None, scaler=None, strict=True, map_location="cpu"):
    """Load a checkpoint into ``model`` (+ optionally optimizer/scaler). Returns the
    raw checkpoint dict (read ``['epoch']`` to resume). ``strict=False`` tolerates
    missing/extra keys (e.g. loading an FCMAE encoder into the dense backbone)."""
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=strict)
    if not strict:
        print(f"load_checkpoint(strict=False): missing={list(msg.missing_keys)} "
              f"unexpected={list(msg.unexpected_keys)}")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def remap_checkpoint_keys(ckpt):
    """Remap sparse-FCMAE encoder weights to the dense ConvNeXtV2 layout.

    Strips the ``encoder.`` prefix, converts MinkowskiEngine ``*.kernel`` tensors
    into ``*.weight`` conv kernels (3-D standard / 2-D depthwise), drops the ``ln``/
    ``linear`` infixes, and reshapes GRN affine params. Verbatim from the reference;
    needed before loading an FCMAE checkpoint into the dense backbone for finetuning.
    """
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove 'encoder' prefix
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove 'kernel'
            new_k = k + ".weight"
            if len(v.shape) == 3:  # standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2:  # depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove 'ln'/'linear' infix
            new_k = ".".join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    for k, v in new_ckpt.items():  # reshape GRN affine params / biases
        if k.endswith("bias") and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt


# next section has upernet utils
# TODO: port over UPerNet utils once port is finalized