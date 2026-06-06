from __future__ import annotations

from typing import Any, Sequence, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# modified from -> https://github.com/facebookresearch/ConvNeXt-V2/
# basically unchanged, just type hints
class LayerNorm(nn.Module):
    """
    LayerNorm with two data formats: channels_last (default) or channels_first. 
    Ordering of dimensions in inputs. 
    
    channels_last - inputs shape (batch_size, height, width, channels)
    channels_first - inputs shape (batch_size, channels, height, width)

    """
    def __init__(self, normalized_shape: int, eps: float=1e-6, channels_last: bool=True) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channels_last = channels_last
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

# modified from -> https://github.com/facebookresearch/ConvNeXt-V2/
# basically unchanged, just type hints
class GRN(nn.Module):
    """
    GRN (Global Repsonse Normalization) layer
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

