"""
convnextv2.py: implementation of ConvNeXt V2
"""


# Taken from -> https://github.com/facebookresearch/ConvNeXt-V2/

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations
from typing import Sequence

'''
V1 Paper -> https://arxiv.org/pdf/2201.03545
V2 Paper -> https://arxiv.org/pdf/2301.00808
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath
from .utils import LayerNorm, GRN


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim: int, drop_path: float=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) # GRN applied
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x) 
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (Sequence[int]): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (Sequence[int]): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self,
        in_channels: int=3, 
        num_classes: int=300, # default changed from 1000 to 300 for this project
        depths: Sequence[int]=[3, 3, 9, 3],
        dims: Sequence[int]=[96, 192, 384, 768], 
        drop_path_rate: float=0.,
        head_init_scale: float=1.
    ):
        # initialization
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # stem ("patchify" layer)
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, channels_last=False)
        )

        self.downsample_layers.append(stem)

        # downsample layers before passing into each block
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, channels_last=False),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        
        # drop path rates
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        cur = 0
        
        # ConvNeXt V2 blocks
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # final norm layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # weight initialization 
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # for semantic segmentation skip connection
        self.seg_norms = nn.ModuleList(
            [LayerNorm(dims[i], eps=1e-6, channels_last=False) for i in range (4)]
        )

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0) # type: ignore

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    # create an updated forward_features for segmentation task
    def forward_features_seg(self, x: torch.Tensor) -> tuple:
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(self.seg_norms[i](x))
        return tuple(outs)

    # base ConvNeXt V2 is configured for image classification
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 atto size
        depths = [2, 2, 6, 2]
        dims = [40, 80, 160, 320]
    """
    model = ConvNeXtV2(in_channels, num_classes, [2, 2, 6, 2], [40, 80, 160, 320], drop_path_rate, head_init_scale)
    return model

def convnextv2_femto(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 femto size
        depths = [2, 2, 6, 2]
        dims = [48, 96, 192, 384]
    """
    model = ConvNeXtV2(in_channels, num_classes, [2, 2, 6, 2], [48, 96, 192, 384], drop_path_rate, head_init_scale)
    return model

def convnext_pico(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 pico size
        depths = [2, 2, 6, 2]
        dims = [64, 128, 256, 512]
    """
    model = ConvNeXtV2(in_channels, num_classes, [2, 2, 6, 2], [64, 128, 256, 512], drop_path_rate, head_init_scale)
    return model

def convnextv2_nano(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 nano size
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
    """
    model = ConvNeXtV2(in_channels, num_classes, [2, 2, 8, 2], [80, 160, 320, 640], drop_path_rate, head_init_scale)
    return model

def convnextv2_tiny(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 tiny size
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
    """
    model = ConvNeXtV2(in_channels, num_classes, [3, 3, 9, 3], [96, 192, 384, 768], drop_path_rate, head_init_scale)
    return model

def convnextv2_base(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 base size
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
    """
    model = ConvNeXtV2(in_channels, num_classes, [3, 3, 27, 3], [128, 256, 512, 1024], drop_path_rate, head_init_scale)
    return model

def convnextv2_large(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 large size
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
    """
    model = ConvNeXtV2(in_channels, num_classes, [3, 3, 27, 3], [192, 384, 768, 1536], drop_path_rate, head_init_scale)
    return model

def convnextv2_huge(in_channels: int=3, num_classes: int=300, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 huge size
        depths = [3, 3, 27, 3]
        dims = [352, 704, 1408, 2816]
    """
    model = ConvNeXtV2(in_channels, num_classes, [3, 3, 27, 3], [352, 704, 1408, 2816], drop_path_rate, head_init_scale)
    return model