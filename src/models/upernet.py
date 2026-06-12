# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py 

from __future__ import annotations
from typing import Any, Sequence

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_

from .convnextv2 import ConvNeXtV2


# upernet
class UPerNet(nn.Module):
    def __init__(self, 
                 num_classes: int=301, # +1 for background class 
                 fc_dim: int=768, # channel dimension of output feature map, default 768 for convnext tiny
                 pool_scales: Sequence[int]=(1, 2, 3, 6), # output grid sizes, 1 = global avg pool, 2 = 2 regions, 3 = 3... etc
                 fpn_inplanes: Sequence[int]=[96, 192, 384, 768], # feature pyramid network in planes - tldr skip connections
                 fpn_dim: int=512,
                 segSize: Any=None): # internal channel width
        super().__init__()

        # PPM - pyramid pooling module, deep semantic information
        self.ppm_pool = nn.ModuleList() # pool down to s x s
        self.ppm_conv = nn.ModuleList() # 1x1 conv, reduce channels to 512

        for scale in pool_scales: # create each s x s pooling layer
            self.ppm_pool.append(nn.AdaptiveAvgPool2d(scale)) # s x s pooling layer
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),  # convolution
                nn.BatchNorm2d(512),                                # batch norm
                nn.ReLU(inplace=True),                              # activation
            ))

        self.ppm_last_conv = nn.Sequential(                             # turns back into fpn_dim channels
            nn.Conv2d(fc_dim + len(pool_scales) * 512, fpn_dim,         # the channel reduction part
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),                                    # err batch norm? idk why
            nn.ReLU(inplace=True),                                      # activation
        )

        # FPN module - feature pyramid module, improves resolution
        self.fpn_in = nn.ModuleList([   # take skip connection feature, squeeze to fpn_dim channels
            nn.Sequential( 
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False), 
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ) for fpn_inplane in fpn_inplanes[:-1]]) # each sequential block handles channel deep info

        self.fpn_out = nn.ModuleList([ # point of this is to remove antialiasing
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(len(fpn_inplanes) - 1)])
        
        self.conv_last = nn.Sequential(     # per pixel class scores
            nn.Sequential(
                nn.Conv2d(len(fpn_inplanes) * fpn_dim, fpn_dim, 
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),  # the classifier
        )

        # the decoder is trained from scratch (no pretrained weights allowed), so
        # init deliberately rather than relying on PyTorch defaults. Mirrors the
        # encoder's scheme (convnextv2.py _init_weights): trunc_normal_(std=.02) on
        # conv/linear weights, zero bias; BN as identity affine (weight 1, bias 0).
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:  # most decoder convs are bias=False
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, conv_out: Sequence[torch.Tensor], segSize = None):
        # PPM portion
        conv5 = conv_out[-1]    # grabs last feature map from ConvNeXt

        input_size = conv5.size()   # get the size
        ppm_out = [conv5]           # save full resolution feature
        for pool_scale, pool_conv in zip(self.ppm_pool, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode="bilinear",
                align_corners=False
            )))
        ppm_out = torch.cat(ppm_out, 1)     # big boi - big channel-wise concat
        f = self.ppm_last_conv(ppm_out)     # result is output from PPM

        # FPN portion
        fpn_feature_list = [f]              # deepest feature is result from PPM
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]            # get the features from the encoder, reversed to go from most downsampled to least
            conv_x = self.fpn_in[i](conv_x) # perform fpn operation on that jawn

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )
            f = conv_x + f                  # interpolate / upscale

            fpn_feature_list.append(self.fpn_out[i](f))     # feature list for FPN is created
        
        # order of fpn_feature_list is currently in deep to shallow
    
        fpn_feature_list.reverse()  # back to shallow to coarse
        output_size = fpn_feature_list[0].size()[2:]    # size of P2, finest resolution
        fusion_list = [fpn_feature_list[0]]             # shallowest layer as is, already at output dims
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(   # upsample other layers to clearest resolution
                fpn_feature_list[i],
                output_size,
                mode="bilinear",
                align_corners=False,
            ))
        fusion_out = torch.cat(fusion_list, 1)  # concat along channels, assuming (N, C, H, W)
        x = self.conv_last(fusion_out)          # get class logits

        # Raw logits at stride-4 res: the seg engine owns upsampling + loss/argmax
        # (see SEG_PIPELINE_PATTERNS.md section 4), keeping the head reusable for TTA.
        return x

# combined encoder + decoder head
class ConvNeXt_UPerNet(nn.Module):
    def __init__(
            self,
            num_channels: int = 3,
            num_classes: int = 301, # 301 for classes + background
            depths: Sequence[int] = [3, 3, 9, 3],
            dims: Sequence[int] = [96, 192, 384, 768],
            drop_path_rate: float = 0.,
            head_init_scale: float = 1.,
            segSize: Any = None # TODO: robustify typing later
        ) -> None:
        super().__init__()
        self.encoder = ConvNeXtV2(
            in_channels=num_channels,
            num_classes=num_classes - 1,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            head_init_scale=head_init_scale,
        )
        self.decoder = UPerNet(
            num_classes=num_classes,
            fc_dim = dims[-1],
            pool_scales=(1, 2, 3, 6),
            fpn_inplanes=dims,
            fpn_dim=512,
            segSize = segSize,
        )

    def forward(self, x: torch.Tensor):
        conv_out = self.encoder.forward_features_seg(x)
        logits = self.decoder(conv_out)
        return logits # TODO: double check this

def atto(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 atto size
        depths = [2, 2, 6, 2]
        dims = [40, 80, 160, 320]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [2, 2, 6, 2], [40, 80, 160, 320], drop_path_rate, head_init_scale)
    return model

def femto(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 femto size
        depths = [2, 2, 6, 2]
        dims = [48, 96, 192, 384]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [2, 2, 6, 2], [48, 96, 192, 384], drop_path_rate, head_init_scale)
    return model

def pico(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 pico size
        depths = [2, 2, 6, 2]
        dims = [64, 128, 256, 512]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [2, 2, 6, 2], [64, 128, 256, 512], drop_path_rate, head_init_scale)
    return model

def nano(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 nano size
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [2, 2, 8, 2], [80, 160, 320, 640], drop_path_rate, head_init_scale)
    return model

def tiny(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 tiny size
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [3, 3, 9, 3], [96, 192, 384, 768], drop_path_rate, head_init_scale)
    return model

def base(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 base size
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [3, 3, 27, 3], [128, 256, 512, 1024], drop_path_rate, head_init_scale)
    return model

def large(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 large size
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [3, 3, 27, 3], [192, 384, 768, 1536], drop_path_rate, head_init_scale)
    return model

def huge(in_channels: int=3, num_classes: int=301, drop_path_rate: float=0., head_init_scale: float=1.):
    """
    returns an instance of convnext v2 huge size
        depths = [3, 3, 27, 3]
        dims = [352, 704, 1408, 2816]
    """
    model = ConvNeXt_UPerNet(in_channels, num_classes, [3, 3, 27, 3], [352, 704, 1408, 2816], drop_path_rate, head_init_scale)
    return model