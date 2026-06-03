"""Shared helpers: reproducibility, mask <-> id encoding, RLE.

Mask spec (see CLAUDE.md):
    segmentation_id = R + G * 256
    0          -> background / non-target
    1..300     -> foreground, segmentation_id = class_id + 1
    1000       -> ignore region (ground-truth only, never predicted)
"""

from __future__ import annotations

import datetime
import math
import random
import time
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

import numpy as np
import torch

IGNORE_ID = 1000
NUM_CLASSES = 300

# Logging helpers now LIVE in core.logging (their canonical home). Import from
# there -- e.g. ``from src.core.logging import setup_logging, get_logger``.


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


# ===========================================================================
# Training infra (single-GPU port of ConvNeXt-V2/utils.py; all DDP removed).
# ===========================================================================


class SmoothedValue:
    """Track a series of values with a windowed median/avg and a global avg."""

    def __init__(self, window_size: int = 20, fmt: str | None = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):  # single-GPU no-op (kept for API parity)
        return

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


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
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
