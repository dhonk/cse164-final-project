# Modified from -> https://github.com/facebookresearch/ConvNeXt-V2/

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

'''
V1 Paper -> https://arxiv.org/pdf/2201.03545
V2 Paper -> https://arxiv.org/pdf/2301.00808
'''

import torch
import torch.nn as nn

from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath

from .utils import (
    LayerNorm,
    SparseLayerNorm,
    SparseGRN,
    SparseDropPath,
    SparseGELU,
    SparseLinear,
)

import spconv.pytorch as sp

# note - submanifold sparse convolution as detailed by the convnext paper.
# just as the dense block uses nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
# spconv can directly map this to spconv.SubMConv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
# the groups=dim portion makes this a depthwise port


class Block(nn.Module):
    """
    sparse ConvNeXt V2 block in spconv port

    args:
        dim (int): number of input channels
        drop_path (float): stochastic depth rate (default =  0.0)
        layer_scale_init_value (float): init value for layer scale, (default = 1e-6)
    """
    def __init__(self, dim: int, drop_path: float = 0., layer_scale_init_value: float = 0.):
        super().__init__()
        self.dwconv = sp.SubMConv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = SparseLayerNorm(dim, 1e-6)
        self.pwconv1 = SparseLinear(dim, 4 * dim)
        self.act = SparseGELU()
        self.pwconv2 = SparseLinear(4 * dim, dim)
        self.grn = SparseGRN(4 * dim)
        self.drop_path = SparseDropPath(drop_path)

    def forward(self, x: sp.SparseConvTensor) ->  sp.SparseConvTensor:
        in_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x
    
class SparseConvNeXtV2(nn.Module):
    """
    Sparse ConvNeXt V2 ported to spconv

    args:
        channels (int): number of input image channels (default = 3)
        num_classes (int): number of classes for classification head (default = 300 for this project)
        depths (tuple[int] | list[int]): number of blocks at each stage (default = [3, 3, 9, 3])
        dims (tuple[int] | list[int]): feature dimension at each stage. (default = [96, 192, 384, 768])
        drop_path_rate (float): stochastic depth rate (default = 0)
        head_init_scale (float): init scaling value for classifier weights and biases. (default: 1)
    """
    def __init__(self,
                 channels: int = 3,
                 num_classes: int = 300,
                 depths: tuple[int] | list[int] = [3, 3, 9, 3],
                 dims: tuple[int] | list[int] = [96, 192, 384, 768],
                 drop_path_rate: float = 0.,
                 ):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # stem
        stem = nn.Sequential(
            nn.Conv2d(channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, channels_last=False)
        )
        self.downsample_layers.append(stem)

        # intermmediate downsampling
        for i in range(3):
            downsample_layer = nn.Sequential(
                SparseLayerNorm(dims[i], eps=1e-6),
                sp.SparseConv3d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True) # note: change coordinates intentionally...?
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each with convnextv2 blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, sp.SparseConv3d):
            trunc_normal_(m.linear.weight, std=.02) # type:ignore
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, sp.SubMConv3d):
            trunc_normal_(m.linear.weight, std=.02) # type:ignore
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)        
        if isinstance(m, SparseLinear):
            trunc_normal_(m.linear.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # type: ignore

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2**(num_stages-1))
        mask = mask.unsqueeze(1).type_as(x)          # (N, 1, H, W)

        # patch embedding
        x = self.downsample_layers[0](x)             # (N, C, H, W), dense
        x *= (1. - mask)                             # zero out masked sites

        # sparse encoding
        x_sparse = sp.SparseConvTensor.from_dense(x.permute(0, 2, 3, 1).contiguous())
        for i in range(4):
            x_sparse = self.downsample_layers[i](x_sparse) if i > 0 else x_sparse
            x_sparse = self.stages[i](x_sparse)

        # densify
        x = x_sparse.dense()                                # (N, C, H, W), channels-first
        return x