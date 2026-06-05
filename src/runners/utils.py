from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from pathlib import Path

from ..models.convnext.convnextv2 import ConvNeXtV2
from ..models.convnext.fcmae import FCMAE


class LossScaler:
    """
    Lightweight AMP loss-scaler.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, use_amp: bool = True, device: str = "cuda"):
        self._scaler = torch.amp.GradScaler(device, enabled=use_amp) # type:ignore

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters=None,
        update_grad: bool = True,
    ) -> None:
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None, "clip_grad needs parameters"
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self) -> dict:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._scaler.load_state_dict(state_dict)

def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    split parameters into weight-decay/no-decay groups

    biases & 1-d parameters are excluded from weight decay.

    Delegates to timm's helper (a layer/optim utility, not a pretrained-weight
    load) so we stay consistent with the reference; falls back to a manual split
    only if timm's entry points are unavailable.
    """
    try:
        from timm.optim import param_groups_weight_decay
        return param_groups_weight_decay(model, weight_decay=weight_decay)
    except ImportError:
        pass
    try:
        from timm.optim.optim_factory import add_weight_decay
        return add_weight_decay(model, weight_decay)
    except ImportError:
        pass

    # manual fallback: biases & 1-D params (norms) get no weight decay
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

def save_checkpoint(
    model: nn.Module,
    file: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    loss_scaler: LossScaler | None = None,
    epoch: int | None = None,
    config=None,
) -> None:
    """
    save_checkpoint: save the weights/etc of a model as a checkpoint file.

    The on-disk format mirrors the ConvNeXt-V2 reference so downstream cls/seg
    finetune can read `ckpt["model"]` directly:
        {"model", "optimizer", "scaler", "epoch", "config"}
    Only `model` is always present; the rest are written when provided.
    """
    to_save: dict = {"model": model.state_dict()}
    if optimizer is not None:
        to_save["optimizer"] = optimizer.state_dict()
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    if epoch is not None:
        to_save["epoch"] = epoch
    if config is not None:
        to_save["config"] = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_save, file)


def load_checkpoint(
    model: nn.Module,
    file: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    loss_scaler: LossScaler | None = None,
    map_location: str = "cpu",
) -> dict:
    """
    load_checkpoint: restore weights (and optionally optimizer/scaler) from a
    checkpoint produced by `save_checkpoint`. Returns the raw checkpoint dict so
    the caller can read `ckpt["epoch"]` etc.

    - FCMAE: strict full restore (resume pretraining / reconstruction).
    - ConvNeXtV2: encoder warm-start with strict=False -- the seg-path norms are
      new/random and absent from the FCMAE checkpoint (documented finetune path).
    - anything else: best-effort strict=False load of the model weights.
    """
    ckpt = torch.load(file, map_location=map_location)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(model, FCMAE):
        model.load_state_dict(state)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if loss_scaler is not None and "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
    elif isinstance(model, ConvNeXtV2):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    return ckpt

