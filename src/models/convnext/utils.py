from __future__ import annotations

from typing import Any, Sequence, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as sp

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
    

# modified from -> https://github.com/facebookresearch/ConvNeXt-V2/
# changed to support spconv
# TODO: this is currently applying batch-wise!! to keep an eye on
class SparseGRN(nn.Module):
    """
    GRN (Global Response Normalization) for sparse tensors
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        Gx = torch.norm(x.features, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        new_features = self.gamma * (x.features * Nx) + self.beta + x.features
        return x.replace_feature(new_features)
    

# modified from -> https://github.com/facebookresearch/ConvNeXt-V2/
# changed to support spconv
class SparseDropPath(nn.Module):
    """
    Drop path for sparse tensors
    """
    def __init__(self, drop_prob: float=0., scale_by_keep: bool=True):
        super(SparseDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        if self.drop_prob == 0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob # for scale_by_keep
        batch_idx = x.indices[:, 0].long() # index tracking

        # logic check:
        #   torch.rand generates [0, 1), one per batch because of x.batch_size
        #   keep mask becomes a tensor of what batches to not drop from
        keep = (torch.rand(x.batch_size, device=x.features.device) >  self.drop_prob)
        keep = keep.to(x.features.dtype) # type enforcement

        if self.scale_by_keep and keep_prob > 0.0:
            keep = keep / keep_prob
        
        mask = keep[batch_idx].unsqueeze(1)
        return x.replace_feature(x.features * mask)


# modified from -> https://github.com/facebookresearch/ConvNeXt-V2/
# changed to support spconv
class SparseLayerNorm(nn.Module):
    """
    channel-wise layer normalization for sparse tensors
    """
    def __init__(self, normalized_shape: int, eps: float=1e-6):
        super(SparseLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape,  eps=eps)
    
    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        output = self.ln(x.features)
        return x.replace_feature(output)
    

class SparseGELU(nn.Module):
    """
    GeLU for sparse tensor
    """
    def __init__(self):
        super().__init__()
        self.act = nn.GELU()
    
    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        return x.replace_feature(self.act(x.features))
    

class SparseLinear(nn.Module):
    """
    Linear layer for sparse tensor
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    
    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        return x.replace_feature(self.linear(x.features))
    

class SparseDepthwiseConv(nn.Module):
    """
    depthwise sparse submanifold convolution
    
    simple approach of just iterating over channels, not best approach! 
    """
    def __init__(self, in_channels: int, kernel_size: int, padding: int,  bias: bool=True):
        super().__init__()
        self.convs = nn.ModuleList([
            # NOTE: large_kernel_fast_algo must stay False here — it triggers an NVRTC
            # build failure for single-channel (depthwise) SubMConv2d kernels.
            # bias=True gives each channel its own learnable scalar bias, matching the
            # reference MinkowskiDepthwiseConvolution(bias=True) and the dense
            # nn.Conv2d depthwise; the bias is NOT absorbed by the following channel-wise
            # LayerNorm, so it affects the computation.
            sp.SubMConv2d(1, 1, kernel_size, bias=bias, indice_key="dw")  # SAME key for all
            for _ in range(in_channels)
        ])

    def forward(self, x: sp.SparseConvTensor) -> sp.SparseConvTensor:
        outs = [conv(x.replace_feature(x.features[:, c:c+1])).features for c, conv in enumerate(self.convs)]
        out = torch.cat(outs, dim=1)
        return x.replace_feature(out)
    
