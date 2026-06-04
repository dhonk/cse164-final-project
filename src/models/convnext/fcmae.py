"""
fcmae.py: implementation of fully masked autoencoder
"""

# Modified from -> https://github.com/facebookresearch/ConvNeXt-V2/


from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

import spconv.pytorch as sp

from timm.layers.weight_init import trunc_normal_
from .convnextv2_sparse import SparseConvNeXtV2, SparseLinear
from .convnextv2 import Block

class FCMAE(nn.Module):
    """
    fully convolutional masked autoencoder, convnext v2 backbone
    """
    def __init__(
            self,
            img_size: int = 224,
            in_channels: int = 3,
            depths: Sequence[int] = [3, 3, 9, 3],
            dims: Sequence[int] = [96, 192, 384, 768],
            decoder_depth: int = 1,
            decoder_embed_dim: int = 512,
            patch_size: int = 32,
            mask_ratio: float = 0.6,
            norm_pix_loss=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.depths = depths
        self.dims = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = SparseConvNeXtV2(in_channels, depths=depths, dims=dims)

        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1],
            out_channels=decoder_embed_dim,
            kernel_size=1
        )

        # mask_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim,
            drop_path=0.) 
        for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)

        # prediction
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_channels,
            kernel_size=1,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, sp.SparseConv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, sp.SubMConv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)        
        if isinstance(m, SparseLinear):
            trunc_normal_(m.linear.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
        
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        patchify images

        imgs:  (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def gen_random_mask(self, x: torch.Tensor, mask_ratio: float = 0.6) -> torch.Tensor:
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate binary mask: 0 = keep 1 = remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
    
    def upsample_mask(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float = 0.6) -> tuple[torch.Tensor, torch.Tensor]:
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask
    
    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, imgs: torch.Tensor, labels=None, mask_ratio: float=0.6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
def convnextv2_atto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model