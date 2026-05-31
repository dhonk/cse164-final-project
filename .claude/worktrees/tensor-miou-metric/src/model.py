"""Model definitions.

HARD RULE (CLAUDE.md): NO pretrained weights. Every layer must be randomly
initialized and trained on the provided competition data only. Whenever a
torchvision/timm constructor is used, pass weights=None / pretrained=False.

Multi-task idea: a shared encoder (e.g. Swin-S) feeding both a classification
head (300 classes) and a segmentation decoder (301 logits: bg + 300 classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 300       # classification: 0..299
NUM_SEG_CLASSES = 301   # segmentation:  0 = background, 1..300 = foreground

