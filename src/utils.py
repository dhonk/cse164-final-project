from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from .core.utils import IGNORE_IDX

from .models.convnextv2 import ConvNeXtV2
from .models.fcmae import FCMAE
from .models.upernet import ConvNeXt_UPerNet


class LossScaler:
    """
    Lightweight AMP loss-scaler.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, use_amp: bool = True, device: str | torch.Device = "cuda"):
        if device != "cuda":
            raise ValueError("Should be cuda")
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


class DiceCELoss(nn.Module):
    """
    CrossEntropy + multiclass soft-Dice. Region-overlap Dice counters the bg
    pixel-dominance that plain CE collapses to. `+1` smoothing zeroes out classes
    absent from a batch. Drop-in for nn.CrossEntropyLoss: same
    `crit(logits[B,C,H,W], target[B,H,W])` signature.
    """

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_IDX,
                 label_smoothing: float = 0.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        valid = (target != self.ignore_index).unsqueeze(1)                  # [B,1,H,W]
        tgt = F.one_hot(target.masked_fill(~valid.squeeze(1), 0),
                        self.num_classes).permute(0, 3, 1, 2).float()       # [B,C,H,W]
        probs = logits.softmax(1) * valid
        tgt = tgt * valid
        inter = (probs * tgt).sum((0, 2, 3))
        denom = probs.sum((0, 2, 3)) + tgt.sum((0, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return ce + self.dice_weight * dice


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    split parameters into weight-decay/no-decay groups

    biases & 1-d parameters are excluded from weight decay.

    Delegates to timm's helper (a layer/optim utility, not a pretrained-weight
    load) so we stay consistent with the reference; falls back to a manual split
    only if timm's entry points are unavailable.
    """
    try:
        from timm.optim._param_groups import param_groups_weight_decay
        return param_groups_weight_decay(model, weight_decay=weight_decay)
    except ImportError:
        pass
    try:
        from timm.optim.optim_factory import add_weight_decay # type:ignore
        return add_weight_decay(model, weight_decay)
    except ImportError:
        pass

    # manual fallback: biases & 1-D params (norms) get no weight decay
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name.endswith(".gamma") or name.endswith(".beta"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def _convnext_layer_id(name: str, depths: list[int]) -> int:
    """
    Map a ConvNeXt-V2 parameter name to a depth index ("single" scheme).

    Ported from the reference get_num_layer_for_convnext_single
    (cooking/references/ConvNeXt-V2/optim_factory.py:32): the stem +
    every block gets its own id (1..sum(depths)); everything outside the
    backbone (norm/head/seg_norms) lands in the trailing id sum(depths)+1,
    which gets lr_scale = layer_decay**0 = 1.0 (full lr).

    Handles both the bare ConvNeXtV2 (names `downsample_layers.*`/`stages.*`)
    and the wrapped ConvNeXt_UPerNet, whose backbone params are prefixed
    `encoder.*` and whose from-scratch `decoder.*` params correctly fall
    through to the trailing full-lr id.
    """
    if name.startswith("encoder."):
        name = name[len("encoder."):]
    if name.startswith("downsample_layers"):
        stage_id = int(name.split(".")[1])
        return sum(depths[:stage_id]) + 1
    if name.startswith("stages"):
        stage_id = int(name.split(".")[1])
        block_id = int(name.split(".")[2])
        return sum(depths[:stage_id]) + block_id + 1
    return sum(depths) + 1


def build_param_groups_lrd(
    model: nn.Module,
    weight_decay: float,
    layer_decay: float,
    no_weight_decay_list: tuple[str, ...] = (),
) -> list[dict]:
    """
    Layer-wise LR-decay parameter groups for ConvNeXt-V2 finetune.

    Faithful port of the reference optim_factory.get_parameter_groups +
    LayerDecayValueAssigner (cooking/references/ConvNeXt-V2/optim_factory.py).
    Each group carries an `lr_scale` that `engines.utils.adjust_learning_rate`
    multiplies the base lr by, so earlier layers train slower than later ones.

    - Decay split: biases, 1-D params (norms), and GRN `.gamma`/`.beta` (4-D,
      so caught by NAME not dimensionality) get weight_decay=0.
    - Scale ladder: id i -> layer_decay ** (num_layers + 1 - i), matching
      main_finetune.py:320 (`layer_decay ** (num_layers + 1 - i)`).

    `model` must expose `.depths`, either directly (ConvNeXtV2) or on a
    wrapped backbone at `.encoder.depths` (ConvNeXt_UPerNet). Returns a list
    of param groups ready for `torch.optim.AdamW(...)`.
    """
    if hasattr(model, "depths"):
        depths = list(model.depths)  # type: ignore[attr-defined]
    else:
        depths = list(model.encoder.depths)  # type: ignore[attr-defined]
    num_layers = sum(depths)
    # ids run 0..num_layers+1; id 0 is unused by the single scheme but kept so
    # the ladder is indexable by layer id (mirrors the reference value list).
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]

    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (param.ndim == 1 or name.endswith(".bias")
                or name.endswith(".gamma") or name.endswith(".beta")
                or name in no_weight_decay_list):
            decay_tag, this_wd = "no_decay", 0.0
        else:
            decay_tag, this_wd = "decay", weight_decay

        layer_id = _convnext_layer_id(name, depths)
        group_name = f"layer_{layer_id}_{decay_tag}"

        if group_name not in groups:
            groups[group_name] = {
                "params": [],
                "weight_decay": this_wd,
                "lr_scale": scales[layer_id],
            }
        groups[group_name]["params"].append(param)

    return list(groups.values())


def save_checkpoint(
    model: nn.Module,
    file: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    loss_scaler: LossScaler | None = None,
    epoch: int | None = None,
    config=None,
    model_ema=None,
) -> None:
    """
    save_checkpoint: save the weights/etc of a model as a checkpoint file.

    The on-disk format mirrors the ConvNeXt-V2 reference so downstream cls/seg
    finetune can read `ckpt["model"]` directly:
        {"model", "optimizer", "scaler", "epoch", "config", "model_ema"}
    Only `model` is always present; the rest are written when provided. `model_ema`
    is a timm `ModelEma` whose shadow weights live at `.ema` (a plain nn.Module).
    """
    to_save: dict = {"model": model.state_dict()}
    if optimizer is not None:
        to_save["optimizer"] = optimizer.state_dict()
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    if epoch is not None:
        to_save["epoch"] = epoch
    if config is not None:
        to_save["config"] = config
    if model_ema is not None:
        to_save["model_ema"] = model_ema.ema.state_dict()

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
    test: bool = False,
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
    if not file:
        raise FileNotFoundError(f"Checkpoint file: {file} not found.")

    ckpt = torch.load(file, map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    if not state:
        raise ValueError(f"Checkpoint file: {file} failed to load state dict")
    
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint file: {file} loaded malformed state dict")

    num_loaded = 0
    unexpected = 0
    num_randomized = 0
    used_ema = False

    if test:
        # prefer EMA shadow weights when the checkpoint carries them -- the EMA
        # model is what we submit; falls back to the raw weights otherwise.
        ema_state = ckpt.get("model_ema") if isinstance(ckpt, dict) else None
        load_state = ema_state if ema_state else state
        used_ema = ema_state is not None

        result = model.load_state_dict(load_state, strict=False)

        unexpected = len(result.unexpected_keys)
        num_loaded = len(load_state) - len(result.unexpected_keys)
        num_randomized = len(result.missing_keys)
    else:
        if isinstance(model, FCMAE):
            result = model.load_state_dict(state)
            if optimizer is not None and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if loss_scaler is not None and "scaler" in ckpt:
                loss_scaler.load_state_dict(ckpt["scaler"])
            num_loaded = len(state)

        elif isinstance(model, ConvNeXtV2):
            encoder_state = {k[len("encoder."):] : v
                            for k, v in state.items() if k.startswith("encoder.")}

            if encoder_state is None:
                raise ValueError(f"Issue reading encoder state from {file}")

            result = model.load_state_dict(encoder_state, strict = False)
            unexpected = len(result.unexpected_keys) # should be 0
            num_loaded = len(encoder_state) - len(result.unexpected_keys)
            num_randomized = len(result.missing_keys)

        elif isinstance(model, ConvNeXt_UPerNet):
            # FCMAE stores the backbone under `encoder.*`, which maps 1:1 onto the
            # seg model's `self.encoder.*`. Keep only those keys (the prefix is
            # preserved) and drop the FCMAE decoder-only keys (`proj`/`decoder`/
            # `pred`/`mask_token`) -- some collide on shape with UPerNet's decoder.
            encoder_state = {k: v for k, v in state.items()
                             if k.startswith("encoder.")}

            if not encoder_state:
                raise ValueError(f"Issue reading encoder state from {file}")

            result = model.load_state_dict(encoder_state, strict=False)
            unexpected = len(result.unexpected_keys)  # should be 0
            num_loaded = len(encoder_state) - len(result.unexpected_keys)
            # the seg head (decoder) + seg-path norms are random by design, so
            # report only the backbone tensors missing relative to the encoder load
            num_randomized = sum(1 for k in result.missing_keys
                                 if k.startswith("encoder."))

        else:
            result = model.load_state_dict(state, strict=False)
            unexpected = len(result.unexpected_keys)
            num_loaded = len(state) - len(result.unexpected_keys)
            num_randomized = len(result.missing_keys)

    return {"epoch": ckpt.get("epoch", "?"),
            "loaded": num_loaded,
            "unexpected": unexpected,
            "random": num_randomized,
            "used_ema": used_ema}

