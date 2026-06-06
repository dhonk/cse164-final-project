"""
convnextv2_sparse_redux.py: "pseudo-sparse" implementation of ConvNeXtV2

Uses a binary mask to mimic usage of sparse convolutions
Intuition: I want to use CUDA 13.2
Also, mask ratio is ~ 0.6, sparse convolution doesn't save much overhead,
especially when considering how much overhead the current sparse implementation brings.

ConvNeXt V1 Paper -> https://arxiv.org/pdf/2201.03545
ConvNeXt V2 Paper -> https://arxiv.org/pdf/2301.00808
"""


# Modified from -> https://github.com/facebookresearch/ConvNeXt-V2/

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations
from typing import cast, Sequence

import torch
import torch.nn as nn

from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath

from .utils import LayerNorm, GRN


class Block(nn.Module):
    """
    pseudosparse convnext v2 block, regular dense convolutions with masked matrices
    
    args:
        dim (int): number of input channels
        drop_path (float): stochastic depth rate (default =  0.0)
        layer_scale_init_value (float): init value for layer scale, (default = 1e-6)

    order of ops inside of pseudosparse block:
    - mask input
    - depthwise conv
    - mask output
    - layernorm
    - pointwise conv
    - mask output
    - gelu
    - grn
    - pointwise conv
    - mask output
    """
    def __init__(self, dim: int, drop_path: float = 0., layer_scale_init_value: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        # keep = (1. - mask)                # (N, C, H, W) # TODO: see if this works, if nah, ggs
        keep_l = keep.permute(0, 2, 3, 1)   # (N, H, W, C)

        x = x * keep                        # (N, C, H, W) 
        in_tensor = x
        x = self.dwconv(x)
        x = x * keep                        # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)           # (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = x * keep_l
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)                 # (N, H, W, C)
        x = x.permute(0, 3, 1, 2)           # (N, C, H, W)
        x = x * keep
        x = in_tensor + self.drop_path(x)
        return x
    
class SparseConvNeXtV2(nn.Module):
    """
    pseudosparse convnextv2, regular dense convolutions with masked

    args:
        in_channels (int): number of input image channels (default = 3)
        num_classes (int): number of classes for classification head (default = 300 for this project)
        depths (Sequence[int]): number of blocks at each stage (default = [3, 3, 9, 3])
        dims (Sequence[int]): feature dimension at each stage. (default = [96, 192, 384, 768])
        drop_path_rate (float): stochastic depth rate (default = 0)
        head_init_scale (float): init scaling value for classifier weights and biases. (default: 1)

    order of ops inside 
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 300,
                 depths: Sequence[int] = [3, 3, 9, 3],
                 dims: Sequence[int] = [96, 192, 384, 768],
                 drop_path_rate: float = 0.,
                 ):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem & 3 downsampling layers
        
        # stem
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, channels_last=False),
        )
        self.downsample_layers.append(stem)

        # intermediate downsampling
        for i in range(3):
            layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, channels_last=False),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True),
            )
            self.downsample_layers.append(layer)

        # 4 resolution stages, each with convnext blocks. MUST MASK
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.ModuleList(
                [Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module | nn.Conv2d | nn.Linear):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0) # type: ignore

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2 ** (num_stages - 1))
        mask = mask.unsqueeze(1).type_as(x) # (N, 1, H, W)
        keep = (1. - mask)

        # patch embedding
        x = self.downsample_layers[0](x)
        x = x * keep

        for i in range(4):
            if i > 0:
                x = self.downsample_layers[i](x)
                keep = keep[:, :, ::2, ::2]   # stride-2 subsample to match downsample conv
            for blk in cast(nn.ModuleList, self.stages[i]):
                x = blk(x, keep)

        return x