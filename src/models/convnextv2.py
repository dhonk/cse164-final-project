from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Ok, time to figure out what this architecture will look like

Swin architecture from paper: https://arxiv.org/pdf/2103.14030

-> Images: H x W x 3
-> Patch Partition (Patchify)
    -> Paper uses patch size 4 x 4, feature dimension of each patch is 4 x 4 x 3 = 48

STAGE 1:
-> Linear Embedding Layer - project it into arbitrary dimension
-> Swin Transformer Block

STAGE 2:
-> Patch Merging
-> Swin Transformer Block

STAGE 3:
-> 
'''

NUM_SEG_CLASSES = 301  # 0 = background, 1..300 = foreground seg ids

class SwinEncoder(nn.Module):
    def __init__(self, num_seg_classes, num_cls_classes):
        super().__init__()

        