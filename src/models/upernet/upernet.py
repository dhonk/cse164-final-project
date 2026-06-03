# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py 

import torch
import torch.nn as nn

# ================================ FIXING IMPORTS ================================

import src.models.convnext.convnextv2 as cv

r'''
from lib.nn import SynchronizedBatchNorm2d

SynchronizedBatchNorm2d - from original codebase:
"Applies Synchronized Batch Normalization over a 4d input that is seen as a mini-batch 
of 3d inputs"

math:: y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

This module differs from the built-in PyTorch BatchNorm2d as the mean and standard-
deviation are reduced across all devices during training.

For example, when one uses `nn.DataParallel` to wrap the network during training, 
PyTorch's implementation normalize the tensor on each device using the statistics only 
on that device, which accelerated the computation and is also easy to implement, but the 
statistics might be inaccurate. Instead, in this synchronized version, the statistics 
will be computed over all training samples distributed on multiple devices.

Note that, for one-GPU or CPU-only case, this module behaves exactly same as the 
built-in PyTorch implementation."

For this project: using only 1x RTX 3090

Further: torch.nn.SyncBatchNorm rectifies this per-GPU issue.
'''

# ==================== IMPORTANT NOTE ABOUT AdaptiveAvgPool2d ====================
'''
NOTE: UPerNet code releases differ in PPM implementation. 

MIT Semantic Segmentation toolkit: 
https://github.com/CSAILVision/semantic-segmentation-pytorch/
features AdaptiveAvgPool2d (where this code came from)

Unified Parsing Code Release:
https://github.com/CSAILVision/unifiedparsing/tree/master
(technically the real code release of UPerNet)

The unifiedparsing code release features PrRoiPooling - pooling with continuous
integration across. This may/may not have downstream performance impact, something
to keep an eye on if mIoU/Boundary F-score in particular seems low.
'''

# ============================ ANOTHER IMPORTANT NOTE ============================
'''
This file originally contained a lot of flexibility for a bunch of different models.
Consider trying out different encoder/decoder structures.

This version stripped everything away and basically just left UPerNet
'''

# DEVIATION (single-GPU): the CSAILVision port used SyncBatchNorm, which requires
# a DDP process group we never create. We train on one GPU, so plain BatchNorm2d
# is the correct equivalent (SyncBN == BN for a single device anyway).

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        '''pixel by pixel accuracy for segmentation'''
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    '''a generalized segmentation module'''
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    '''
    TODO / an aside: I don't like how the individual training structures (error, loss, etc) is actually
    within the model architecture itself. Consider pulling this out to keep model pure, and have training/
    inference FULLY separated
    '''

    # changed name from segSize to seg_size
    def forward(self, feed_dict, *, seg_size=None):
        # training
        if seg_size is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), seg_size=seg_size)
            return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    # TODO: also to note: this basically couples the encoder and decoder together here. Consider pulling
    # out into src.models.classifier and src.models.segmenter
    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        
       
        net_decoder = UPerNet(
            num_class=num_class,
            fc_dim=fc_dim,
            use_softmax=use_softmax,
            fpn_dim=512)
       
        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, seg_size=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=seg_size, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


def build_upernet(dims, num_class=301, fpn_dim=512):
    """Build a randomly-initialized UPerNet decoder for our ConvNeXt V2 backbone.

    ``dims`` is the backbone's per-stage channel tuple (the same ``dims`` passed
    to the ConvNeXt size factory), e.g. atto = (40, 80, 160, 320). It feeds both
    ``fc_dim`` (top stage, = ``dims[-1]``) and ``fpn_inplanes`` (all four stages),
    so the decoder's lateral/PPM convs line up with ``forward_features_seg``'s
    4-tuple of stride-4/8/16/32 feature maps.

    ``num_class=301`` → seg ids 0..300 (background 0 + foreground 1..300); argmax
    over the 301 channels yields the seg id directly. ``use_softmax=False`` so the
    decoder emits ``log_softmax`` (paired with ``NLLLoss(ignore_index=1000)``);
    weights are randomly initialized (NO pretrained weights — hard project rule).
    """
    dims = tuple(dims)
    net = UPerNet(
        num_class=num_class,
        fc_dim=dims[-1],
        use_softmax=False,
        fpn_inplanes=dims,
        fpn_dim=fpn_dim,
    )
    net.apply(ModelBuilder.weights_init)
    return net