# CLAUDE.md — CSE 164 Final Project 2026

Semi-supervised image **classification** + **semantic segmentation** from scratch.
For every hidden test image, predict (1) one image-level `class_id` and (2) a
pixel-level segmentation mask. **Segmentation dominates the score.**

This file is guidance for any coding assistant working in this repo. Read it
before writing training, inference, or submission code.

---

## ⛔ Hard rule: NO pretrained weights (project fails if violated)

Every weight in a final-submission model must originate from training on the
**provided competition data only**. This is the single easiest way to get a zero,
and it is easy to trigger by accident.

**Forbidden**

- `torchvision.models.*(weights=...)` / `pretrained=True`
- `timm.create_model(..., pretrained=True)` and any `timm` checkpoint download
- Any `from_pretrained(...)` (HF transformers, segmentation_models_pytorch, etc.)
- Loading any `.pth` / `.ckpt` / `.safetensors` downloaded from the internet
- ImageNet / COCO / self-supervised / foundation / vision-language checkpoints
- External labeled data, external annotations, external vision APIs

**Allowed**

- Public *architecture code* (e.g. a U-Net or DeepLab implementation) **as long
  as it is randomly initialized**, not loaded with weights.
- Public Python packages, docs, and coding assistants for programming help.
- **Self-supervised pre-training (FCMAE) on the competition's own images** —
  this is "training on provided data," not a pretrained weight. It is the core
  of our approach (see below).
- **`timm` is used for layer utilities only** (`trunc_normal_`, `DropPath`).
  NEVER call `timm.create_model(..., pretrained=True)` or download any timm
  checkpoint. The dependency is for `timm.models.layers`, nothing else.

**When adding any model code:** explicitly set `weights=None` / `pretrained=False`,
and never call `.load_state_dict()` on anything not produced by this project's
own training run. If a default downloads weights, change it.

The final report must state that **no pretrained weights and no external training
data** were used. Keep code consistent with that statement.

---

## Data layout

```text
data/
├── train_labeled/images/        # 7,500 imgs, IMAGE-LEVEL labels only
├── train_seg/{images,masks}/    # 3,000 imgs WITH segmentation masks
├── train_unlabeled/images/      # 50,000 imgs, NO labels (+ a few distractors)
├── val/{images,masks}/, classification.json   # 750 public, fully labeled
├── test/images/                 # 3,000 hidden test images
└── metadata/{class_map,train_labeled,train_seg}.json
```

- **300 target classes.** Images are `.JPEG`, variable resolution.
- `val/` is the only labeled split with masks besides `train_seg/` — use it for
  model selection, **never train on it**.
- `train_unlabeled/` contains **distractor images from non-target classes**.
  Filter them (confidence thresholding) before pseudo-labeling.

---

## Label & mask encodings (get these exactly right)

**Classification:** integer `class_id` in `0..299`.

**Segmentation masks (PNG, RGB-encoded):**

```python
segmentation_id = R + G * 256
```

| segmentation_id | meaning |
|---|---|
| `0` | background / non-target |
| `1..300` | foreground class, where `segmentation_id = class_id + 1` |
| `1000` | ignore region — **ground-truth only**, these pixels are unscored |

Critical mapping: foreground `class_id k` ↔ `segmentation_id k+1`. So
classification class `0` is segmentation id `1`; background is its own id `0`
with no classification equivalent. **Off-by-one here silently tanks the score.**

**Prediction constraints**

- Predicted ids must be in `0..300` only. **Never emit `1000`** (rejected).
- When training on masks, treat `1000` pixels as **ignore** in the loss
  (e.g. `ignore_index`), do not let them become a class.
- Each predicted mask must match its input image's **exact** width/height.
  Resize predictions back to original size with **nearest-neighbor** (never
  bilinear — it invents fractional class ids).

---

## Submission format

`submission.csv`, one row per test image:

```csv
image,class_id,segmentation_rle
test_00000.JPEG,17,1 20 18 210 5 18
test_00001.JPEG,3,
```

- `image` — exact test filename (`.JPEG`).
- `class_id` — predicted image-level class in `0..299`.
- `segmentation_rle` — **row-major, 1-indexed RLE triples**:
  `start length value start length value ...`
  - `start` is 1-indexed into the row-major flattened mask.
  - `value` is a segmentation id in `1..300` (foreground only; background is
    just the gaps).
  - Empty string **or** `0` = all-background mask.

Validation rejects: missing/duplicate rows, bad filenames, invalid class ids,
malformed RLE, **overlapping runs**, or out-of-range segmentation ids. Always
run the validator before submitting.

---

## What the score rewards (optimize for this, not raw mIoU)

```text
Kaggle = 70% segmentation + 20% classification        (10% report graded off-Kaggle)

segmentation = 70% mean IoU
             + 20% boundary F-score
             + 10% rare-class mIoU
```

- **mIoU and rare-class mIoU are foreground-only** — background is excluded from
  the per-class average. Predicting background well earns nothing directly.
- **Boundary F-score is 20% of seg** → boundary quality matters; the description
  lists boundary-aware losses or post-processing as a possible direction.
- **Rare-class mIoU + macro accuracy** → both metrics are class-balanced, so
  common classes won't carry you. The description lists class-balanced sampling
  for rare classes as a possible direction.

Implications: prioritize segmentation effort, protect the long tail of rare
classes, and don't let boundaries blur.

---

## Suggested directions (from the project description)

The description notes that strong projects will usually need more than direct
supervised training on the segmentation-labeled images. Listed possibilities:

- Training a segmentation model from scratch with strong augmentation.
- Multi-task learning with a shared encoder.
- Pseudo-labeling high-confidence unlabeled images.
- Confidence filtering for distractor images.
- Boundary-aware losses or post-processing.
- Class-balanced sampling for rare classes.
- Test-time augmentation and model ensembling.

You are not required to use any particular architecture.

---

## Current direction & architecture (COMMITTED)

We have committed to a **shared-encoder, self-supervised pipeline**. Rationale:
limited training time + a small labeled set means we must squeeze every bit of
signal out of the 50k unlabeled images. (In lecture, masked image modeling /
MIM was called the strongest SSL technique.)

**The pipeline:**

1. **Backbone — ConvNeXt V2** (from-scratch architecture code adapted from
   facebookresearch/ConvNeXt-V2). Comes in two forms:
   - *dense* (`convnextv2.py`) — used for fine-tuning / inference.
   - *sparse* (`convnextv2_sparse.py`, needs MinkowskiEngine) — used as the
     FCMAE encoder during pre-training, so masked patches cost no compute.
   - Key component: **GRN (Global Response Normalization)** — V2's fix for the
     feature-collapse V1 suffers under FCMAE. Keep GRN in for the whole run
     (removing it for fine-tuning hurts). LayerScale is dropped in V2.
2. **Pre-training — FCMAE** (`fcmae.py`): Fully Convolutional Masked
   Autoencoder. Mask ~60% of patches, reconstruct pixels, MSE loss on masked
   patches only. Architecture + framework are **co-designed** — FCMAE alone or
   GRN alone barely helps; together they pop off.
3. **Segmentation head — UPerNet** (`upernet/upernet.py`). This is the V2
   paper's ADE20k recipe and our most important downstream task
   (seg = 70% of score). Emits **301 channels** (seg ids 0..300) so `argmax`
   yields the seg id directly.
4. **Classification head** — linear head on pooled backbone features, fine-tuned
   on the 7,500 image-level labels.

**How the dense backbone serves two heads** (one shared encoder, two read-outs):

- **Classification** uses `forward_features(x)` → global-average-pool + final
  `nn.LayerNorm` → `(N, C)` vector → `head` (`nn.Linear`, set `num_classes=300`).
- **Segmentation** uses `forward_features_seg(x)` → returns a **4-tuple of
  multi-scale feature maps** (NCHW, strides 4/8/16/32, channels `dims`), each
  passed through a per-stage channels-first `LayerNorm` (`seg_norms`). UPerNet
  consumes all four. The plain classification `forward_features` is **unusable**
  for seg — it pools away the spatial grid and keeps only the last stage.
- The seg-specific norm layers are **new, randomly-initialized, and only on the
  seg path** — they are NOT in the FCMAE checkpoint, so the pre-trained weights
  load into this backbone with **`strict=False`** (encoder weights match via
  `remap_checkpoint_keys`; `seg_norms` + `head` stay fresh). Renaming/altering
  these seg layers has **no checkpoint implications** (the remap only touches
  `downsample_layers` / `stages`).
- This multi-scale extractor is the **only structural change** ConvNeXt V2
  needed for the whole project — everything else is a separate module (UPerNet),
  config (`num_classes`, `drop_path_rate`, model size), weight-load glue, or
  training procedure. The dense/sparse encoders stay structurally identical so
  the remap lines up.

**Data usage plan for pre-training:** pool the 50k unlabeled + 7.5k labeled
(labels dropped) + 3k seg (masks dropped) ≈ **60.5k images**, but **only the
training splits** — val/test are walled off first to avoid leakage. Fine-tune
each head on its own labeled subset afterward; pseudo-labeling on the unlabeled
pool is a later stretch goal.

**Scale note:** dataset is small, so favor small backbones (Atto ≈ 3.7M params
is the recommended starting point) — large-scale gains don't transfer here;
small-scale gains matter more. Size factories (`convnextv2_atto` … `_huge`)
exist in both `convnextv2.py` and `fcmae.py`.

`resnet/` and `unet/` are from-scratch **baselines** (the original simpler
direction) kept for comparison; the ConvNeXt V2 + FCMAE + UPerNet path is the
real plan.

---

## Repository structure

**Check this tree FIRST before searching the filesystem for where something
lives** — it exists so assistants don't have to re-traverse the whole project.
Keep it current when structure moves. It is **structure only**: for per-file
completion status see the Phase 1–6 plan below and `cooking/CHANGELOG.md` — do
not re-annotate status here. (Excludes `data/`, `__pycache__/`,
`.ipynb_checkpoints/`, `venv/`, and `cooking/src/` image assets.)

> The reference ConvNeXt-V2 repo now lives at `cooking/references/ConvNeXt-V2/`
> (a **local clone for observation only** — NOT part of our project). Read it to
> learn from; never import from it or copy weights. Breakdown:
> `cooking/CONVNEXTV2_CODEBASE.md`.

```text
cse164-final-project/
├── CLAUDE.md                      # this file — read FIRST before searching the repo
├── README.md                     # Docker usage (CLI/deps sections TODO)
├── requirements.txt              # torch>=2.12 (cu126), torchvision, timm, tf, jupyter…
├── compose.yaml                  # `notebook` service (JupyterLab on :8888)
├── notebook.ipynb                # interactive sanity-checks / scratchpad
├── checkpoints/                  # saved model weights (best/last per stage)
├── outputs/                      # submission .csv files (timestamped)
├── logs/                         # stdout + error logs per run
├── cache/                        # scratch cache
├── tests/                        # project-wide unit tests (mirrors src/)
├── docker/
│   ├── Dockerfile                # CUDA 12.6 + PyTorch 2.12 + MinkowskiEngine build
│   └── entrypoint.sh             # task dispatch (default: notebook)
├── cooking/                      # research notes (Obsidian-style), NOT code
│   ├── PROJECT_SPEC.md           # the official competition spec
│   ├── IDEATION.md               # committed approach + live roadmap (high-level authority)
│   ├── CHANGELOG.md              # detailed per-goal change log (what/why/files/test)
│   ├── CONVNEXT.md               # ConvNeXt V1/V2 + FCMAE PAPER notes
│   ├── CONVNEXTV2_CODEBASE.md    # reference SOURCE-CODE breakdown + our design choices
│   ├── CONVERSATIONSNOTES.md     # notes/insight from presentations, meetings, etc
│   ├── LECTURES.md               # FCN / segmentation lecture notes
│   ├── NOTES.md
│   ├── src/                      # image assets for the notes (ignore)
│   └── references/               # reference source code, read-only — observe, never import
│       └── ConvNeXt-V2/          # local clone of the FB repo
├── starter/                      # course-provided, DO NOT EDIT
│   ├── make_sample_submission_csv.py
│   ├── validate_submission_csv.py   # ALWAYS run before submitting
│   └── kaggle_metric.py             # reference scorer (decode_rle_to_mask etc.)
└── src/                          # package; namespace pkgs (NO __init__.py)
    ├── core/                     # shared-infra subpackage
    │   ├── utils.py              # logging, seeding, rgb↔seg_id, RLE, MetricLogger, AMP, LR-sched, ckpt, remap
    │   ├── optim.py              # AdamW + layer-wise LR-decay factory (timm.optim guarded)
    │   ├── dataset.py            # all 5 split Datasets + transforms + loaders + build_pretrain_loader
    │   ├── evaluate.py           # kaggle_metric glue: write_submission/score_val/resize_mask_nearest
    │   └── logging.py            # OWN home for setup_logging/get_logger (+LOG_FORMAT/DATEFMT)
    ├── engines/                  # per-epoch train loops (one file per task)
    │   ├── convnext_cls.py             # cls train_one_epoch + evaluate (top-1 + macro-acc)
    │   ├── convnext_upernet_seg.py     # seg train_one_epoch + seg_forward + pixel_accuracy
    │   └── fcmae_pretrain.py           # FCMAE train_one_epoch (import-clean; full run Docker-only)
    ├── trainers/                 # per-stage drivers (N epochs, ckpt, selection)
    │   ├── train_pretrain.py     # FCMAE pre-training (Docker-only)
    │   ├── train_cls.py          # cls driver: train→val top-1/macro-acc→best/last ckpt
    │   └── train_seg.py          # UPerNet seg driver: train→val-score→best/last ckpt
    ├── tests/                    # prediction → submission
    │   ├── test_cls.py           # predict class_id per test image
    │   └── test_seg.py           # predict mask, resize nearest, RLE
    └── models/
        ├── convnext/             # ConvNeXt V2 + FCMAE (adapted from FB repo)
        │   ├── convnextv2.py     #    dense backbone + Block + size factories
        │   ├── convnextv2_sparse.py  # sparse encoder (needs MinkowskiEngine)
        │   ├── fcmae.py          #    FCMAE pre-trainer + size factories
        │   └── utils.py          #    LayerNorm, GRN, Minkowski* sparse layers
        ├── upernet/
        │   └── upernet.py        # CSAILVision port adapted: BN (not SyncBN) + build_upernet
        ├── resnet/resnet.py      # build_resnet — from-scratch cls baseline
        └── unet/unet.py          # build_unet  — from-scratch seg baseline
```

The `src/` layout mirrors the reference repo's **models / engine / driver**
separation (see `cooking/CONVNEXTV2_CODEBASE.md`): pure models in `models/`,
reusable per-epoch loops in `engines/` (one file per task — `convnext_cls`,
`convnext_upernet_seg`, `fcmae_pretrain`; the prefix leaves room for other
backbones later), one thin entrypoint per training stage in `trainers/`
(`train_pretrain` / `train_cls` / `train_seg`), prediction→submission scripts in
`tests/`, and shared infra in `core/` (`utils`, `optim`, `dataset`, `evaluate`,
`logging`). **Completion status for every file lives in one place — the Phase 1–6
plan below (+ `cooking/CHANGELOG.md` for the detailed log).** Don't duplicate it
here or in the tree.

**Module gotchas (learned the hard way):**

- **No `__init__.py` anywhere** — we rely on Python 3.3+ implicit namespace
  packages (less clutter). `src` is imported as a **package**, so use
  **package-relative imports** (`from .core.utils import rgb_to_seg_id`,
  `from .models.convnext.fcmae import ...`). Run modules from the **project
  root** as `python -m src.trainers.train_seg` (NOT from inside `src/`). There are
  no convenience re-exports, so import builders by full path, e.g.
  `from src.models.resnet.resnet import build_resnet`.
- **Shared helpers live in `src/core/utils.py`** (canonical). The legacy
  `src/utils.py` duplicate has been **deleted**; `optim.py` and `dataset.py` also
  moved under `src/core/`. Nothing references the old flat paths.
- **Logging helpers live in `src/core/logging.py`** (NOT `utils.py`) — it OWNS
  `setup_logging`/`get_logger`/`LOG_FORMAT`/`LOG_DATEFMT`. Import them from there
  (`from src.core.logging import setup_logging, get_logger`). The module is named
  `logging` but its own top-level `import logging` resolves to the **stdlib** via
  absolute import (package path is `src.core.logging`); never use a relative
  `from . import logging`.
- `convnext/convnextv2_sparse.py`, `fcmae.py`, and `convnext/utils.py` import
  **MinkowskiEngine** (and `MinkowskiOps`), which is **only installed in the
  Docker image** (built from source there) and is **not in `requirements.txt`**.
  Sparse/FCMAE code will `ImportError` on a bare `pip install`. The dense
  `convnextv2.py` path does not need it.
- `core.utils.encode_rle` emits `"0"` (not `""`) for an all-background mask, so
  the CSV never carries NaN/null fields — the official decoder treats them same.
- Loss ignore id is **1000**, NOT 255 (255 collides with foreground seg id 255 /
  class 254). Defined as `IGNORE_INDEX` in `core/dataset.py` and `IGNORE_ID` in
  `core/utils.py`.

---

## Detailed implementation plan (RESUME HERE — read this to start fresh)

Single authoritative roadmap. `cooking/IDEATION.md` is the high-level roadmap
authority and `cooking/CHANGELOG.md` holds the *detailed* per-goal change log
(what/why/files/test result); the full approved plan lives at
`C:\Users\ddani\.claude\plans\take-a-look-at-vectorized-planet.md`.

Goal: fill the `src/` skeleton into a working **train → predict → submit**
pipeline via small, unit-tested vertical slices.

**Snapshot:** Phases 1–3 done; Phase 4 underway — `core/` infra complete, UPerNet
adapted, all three engines written, and the seg + cls trainers (4.1/4.2) run
end-to-end locally on real data from random init (both finetune ckpts exist).
**Next: 4.3 `train_pretrain.py` (FCMAE harness, Docker-only) or jump to Phase 5
`tests/` (prediction→submission).** Stretch work (pseudo-labeling, distractor
filtering, TTA, ensembling, boundary post-processing) stays last.

**Working agreement (per-goal cadence — do this at the end of EVERY goal):**
1. Update this CLAUDE.md (terse status edit; tree/gotchas if structure moved).
2. Update `cooking/CHANGELOG.md` (the *detailed* change log — what/why/files/test result).
3. Flag a "good compaction point" (the user runs `/compact`).
4. Continue to the next goal.
Break each function into smaller subgoals first; run a small unit test to prove a
goal before moving on. Don't erase large pasted chunks silently — note big
deviations here + in CHANGELOG.md (see the Sprint-4 truncation precedent).

**Environment & how to run:** dense cls/seg path is locally testable on Windows
via `venv\Scripts\python.exe` (real GPU on the 3090, `torch 2.12+cu126`). Run from
project root: `venv\Scripts\python.exe -m src.<module>`. **MinkowskiEngine/FCMAE is
Docker-only** — sparse/pretrain imports `ImportError` on Windows; that's expected,
leave it. `TORCH_CUDA_ARCH_LIST=8.6` is set for the 3090 — **change it for other
GPUs**. Strategy: build & verify the dense finetune path **from random-init
backbone first**, then slot in FCMAE weights (produced in Docker) via
`remap_checkpoint_keys` + `load_state_dict(strict=False)`.

**Status legend:** ✅ done · 🔄 doing (current) · ⬜ todo.

### Phase 1 — Complete `core/` (foundation)
- ✅ **1.0** Env + import sanity check (venv GPU confirmed).
- ✅ **1.1** `core/utils.py` single-GPU training infra (MetricLogger, NativeScaler,
  adjust_learning_rate, save/load_checkpoint, remap_checkpoint_keys). Unit-tested.
- ✅ **1.3** `core/optim.py` imports cleanly (timm.optim guarded). AdamW + layer
  decay verified.
- ✅ **1.4** `core/dataset.py` → **`build_pretrain_loader`** added (+ generic
  `ImageOnlyDataset` wrapper taking element `[0]`). `ConcatDataset` of all three
  TRAINING splits — `train_unlabeled` (50k) + `train_labeled` (7.5k) + `train_seg`
  (3k), labels/masks stripped — wrapped uniformly (incl. unlabeled) so default
  collate stacks a clean `(N,3,C,C)` batch. `ImageTransform(crop_size=224)`
  (train_seg uses `SegTransform`; its mask is read then discarded — minor I/O).
  *Tested:* `len==60500`; items `(3,224,224)` float32, `C%32==0`; batch→`(4,3,224,224)`.
- ✅ **1.5** `core/evaluate.py` — done. Imports `starter/kaggle_metric.py` by file
  path (lazy+cached `_kaggle_metric()`; never edited). `write_submission(rows,path)`
  + `score_val(preds,data_root)` share a `_normalize_rows` row contract (dict-keyed
  OR list; each row gives `segmentation_rle` **or** a `seg_mask` array → `encode_rle`).
  `score_val` builds a cached `_solution_frame` (val GT masks at native res → RLE
  with H/W + ignore-1000 preserved) and calls `detailed_score`. Added
  `resize_mask_nearest(mask, (W,H))` helper (nearest, exact ids). *Tested on all
  750 val:* perfect preds → mIoU/boundary/rare/macro-acc all 1.0 (automated 0.9
  ceiling); all-bg → seg≈0, cls 1.0; submission roundtrip 750 rows, no NaN.
- ✅ **1.2** `core/logging.py` is now its **own self-contained module** that OWNS
  `setup_logging`/`get_logger`/`LOG_FORMAT`/`LOG_DATEFMT` — they were MOVED out of
  `core/utils.py` (which no longer defines or re-exports them, and dropped its now-
  unused `import logging`/`import sys`). `logging.py` keeps a top-level stdlib
  `import logging` (resolves to stdlib via absolute import; module path is
  `src.core.logging`). No circular import (logging.py imports nothing from utils;
  no current caller imported these from utils). Import gate re-run green; logging
  behavior + stdlib-unshadowed asserted. *(The prior thin-re-export attempt is
  superseded.)*

### Phase 2 — Adapt `models/upernet/upernet.py` (port is gold; touch minimally) — ✅ DONE
- ✅ **2.1** `BatchNorm2d = torch.nn.SyncBatchNorm` → **`nn.BatchNorm2d`**
  (single-GPU; SyncBN needs a DDP process group we never create — and SyncBN == BN
  on one device). Deviation noted inline + here. Also raw-stringified the LaTeX
  comment block to silence a `SyntaxWarning`.
- ✅ **2.2** Added `build_upernet(dims, num_class=301, fpn_dim=512)` →
  `UPerNet(num_class=301, fc_dim=dims[-1], fpn_inplanes=tuple(dims), fpn_dim=512,
  use_softmax=False)` + `apply(ModelBuilder.weights_init)` (random init — no
  pretrained). `num_class=301` ⇒ seg ids 0..300; `fpn_inplanes` = backbone stage
  dims (atto: 40/80/160/320).
- ✅ **2.3** Seg path wired & smoke-tested (do NOT use `SegmentationModule` — it
  bakes loss in and calls `encoder(..., return_feature_maps=True)` our backbone
  lacks). Path: `feats = backbone.forward_features_seg(img)` (4-tuple = UPerNet
  `conv_out`) → `decoder(feats)` (emits `log_softmax` at **stride-4**, i.e.
  `(N,301,56,56)` for 224 in) → **upsample to crop size** (bilinear) →
  `nn.NLLLoss(ignore_index=1000)`. *Tested:* atto backbone + build_upernet,
  `(2,3,224,224)` → decoder `(2,301,56,56)` → upsample `(2,301,224,224)`; one
  NLLLoss backward (with some `1000` ignore pixels) → finite backbone grads. PASS.

### Phase 3 — Engines (single-epoch loops; templates in `ConvNeXt-V2/engine_*.py`)
- ✅ **3.1** `engines/convnext_upernet_seg.py` — `train_one_epoch(backbone, decoder,
  criterion, loader, optimizer, device, epoch, loss_scaler, args, max_norm, use_amp)`:
  `seg_forward` (features→decoder→bilinear-upsample) → `NLLLoss(ignore_index=1000)`;
  per-iter `adjust_learning_rate` (fractional epoch); AMP via `NativeScaler`;
  `MetricLogger`(loss, pixel_acc, lr, grad_norm). `pixel_accuracy` excludes 1000.
  Loader yields `(image,label,seg_id,name)`; uses image+seg_id. *Tested (GPU):*
  overfits a tiny synthetic set, loss 6.23→5.69 monotonic, ignore-only target→acc
  0.0 no crash, `seg_forward` out `(2,301,96,96)`. PASS.
- ✅ **3.2** `engines/convnext_cls.py` — `train_one_epoch` (CE over 300, drives
  `backbone(x)` directly; AMP/cosine/grad-clip; loss/acc1/lr/grad_norm) +
  `evaluate` (top-1 **and** macro/class-balanced acc; reads batch[0:2] so handles
  both 3-tuple labeled + 5-tuple val contracts). *Tested (GPU):* overfits a tiny
  set loss 5.71→0.0085, acc1→1.0; evaluate returns acc1/macro_acc. PASS.
- ✅ **3.3** `engines/fcmae_pretrain.py` — `train_one_epoch(model, data_loader,
  optimizer, device, epoch, loss_scaler, args)`: image-only batches (no labels);
  `loss,_,_ = model(samples, mask_ratio)`; optional `update_freq` grad-accum;
  `torch.cuda.empty_cache()` on each optimizer step (ME hygiene). Engine FILE
  imports only `core.utils` (model passed in) so it's **import-clean on Windows**;
  full run is Docker-only (FCMAE model needs MinkowskiEngine). *Verified:* imports
  green on Windows, signature checked. *(No local run — sparse model needs Docker.)*

### Phase 4 — Trainers (N epochs, checkpointing, model selection)
- ✅ **4.1** `trainers/train_seg.py` — argparse driver: `MODELS` registry
  (atto/femto/nano/tiny → factory+dims) → dense backbone + `build_upernet`;
  `--finetune` optionally warm-starts the encoder (remap + strict=False; seg_norms/
  head stay fresh); `TrainSegDataset` loader (+`--limit-train` for smoke); seg engine
  per epoch; `evaluate_val` iterates `ValDataset` directly (variable-size masks →
  no DataLoader), forward→argmax→`resize_mask_nearest`→`score_val`; saves best
  (by `segmentation_score`) + last ckpt (backbone+decoder+optim+scaler+scores).
  **DEVIATION:** plain `AdamW` over backbone+decoder, NOT `create_optimizer`+layer
  decay (layer decay matters for *pretrained* finetune, not random init; wire it
  when FCMAE weights land). *Tested (GPU, real data):* 1 epoch / 32 imgs → loss
  6.33→5.90, full 750-val `score_val` returns the metric dict (seg=0.0075 from
  ~untrained), best/last ckpt written **and reload strict into fresh
  backbone+decoder**. PASS.
- ✅ **4.2** `trainers/train_cls.py` — argparse driver mirroring `train_seg.py`:
  `MODELS` registry → dense backbone (`num_classes=300`); `--finetune` warm-starts
  the encoder (remap + strict=False; `head` stays fresh); `TrainLabeledDataset`
  loader (+`--limit-train` smoke); cls engine per epoch (CE over 300); val eval via
  a `DataLoader(ValDataset, collate_fn=_val_collate)` that keeps only image+label
  (drops the variable-size GT masks default collate can't stack); selects **best by
  `macro_acc`** (class-balanced, the leaderboard cls metric) + last ckpt. Same
  **DEVIATION** as 4.1: plain `AdamW`, no layer-decay. *Tested (GPU, real data):*
  1 epoch / 64 imgs → loss 6.08→5.94, full 750-val eval acc1 0.0040 / macro_acc
  0.0033 (≈1/300 chance, untrained), best/last ckpt written and **reload strict
  into a fresh backbone (incl. head)**. PASS.
- ⬜ **4.3** `trainers/train_pretrain.py` — FCMAE harness: fcmae size factory +
  pooled loader; AdamW (no layer decay, betas 0.9/0.95, linear-scaled lr); cosine;
  **save encoder ckpt** (the artifact finetune consumes). *Docker-only.*

### Phase 5 — `tests/` (prediction → submission)
- ⬜ **5.1** `tests/test_seg.py` — load seg ckpt; per test image predict, **resize
  to original (W,H) nearest**, `encode_rle` → rows.
- ⬜ **5.2** `tests/test_cls.py` — load cls ckpt; predict class_id per test image.
- ⬜ **5.3** Merge into one `submission.csv` (`write_submission`) and **run
  `starter/validate_submission_csv.py`** before any upload.

### Phase 6 — `pipeline.py` (root CLI)
- ⬜ **6.1** Thin argparse dispatcher: `train {pretrain,cls,seg}`,
  `predict {cls,seg}`, `submit`. Calls the trainers/tests above.

### Housekeeping conventions (keep honoring these)
- Repo tree + gotchas track the real `src/` subpackage layout (`core/` /
  `engines/` / `trainers/` / `tests/`) — update them when structure moves.
- `src/core/utils.py` is the canonical shared-helpers home; the legacy flat
  `src/utils.py` is deleted — don't reintroduce it.

### Verification gates (run as you go)
- **Imports (local, root):** `venv\Scripts\python.exe -c "import src.core.optim,
  src.core.utils, src.core.evaluate, src.models.upernet.upernet"`. Minkowski
  failures are expected on Windows.
- **Overfit smoke (dense seg):** `... -m src.trainers.train_seg` 1–2 epochs on a
  handful of `train_seg` images; loss drops, ckpt written, resume loads it.
- **Eval wiring:** `core.evaluate.score_val` on `val/` returns
  mIoU/boundary/rare/macro-acc on the kaggle_metric scale.
- **Submission:** small `submission.csv` → `python starter/validate_submission_csv.py
  --submission submission.csv --data-root data --split val` passes.

### Deviations to record (already decided)
- Single-GPU: `nn.BatchNorm2d` in UPerNet (not SyncBN); no DDP anywhere.
- Seg loss path: decoder `log_softmax` + upsample-to-crop + `NLLLoss(ignore_index=1000)`.
- `core/optim.py` kept whole; only unused breaking `timm.optim` imports trimmed.
- `core/utils.py` truncated to the clean single-GPU port (verbatim ConvNeXt-V2
  paste dropped; user approved — "leave the clean tested port for now").
- Recommended starting backbone: `atto`/`femto` (small dataset).

---

## Workflow commands

```bash
# --- Dev environment ---------------------------------------------------------
# Preferred: Docker (provides CUDA 12.6 + PyTorch 2.12 + MinkowskiEngine, which
# pip alone cannot install). JupyterLab serves on http://127.0.0.1:8888/lab .
docker compose up notebook -d
docker compose run notebook
docker compose --profile notebook down       # when done

# Bare-metal pip works ONLY for the dense (non-FCMAE) path — MinkowskiEngine,
# and therefore sparse ConvNeXt / FCMAE pre-training, is NOT pip-installable.
pip install -r requirements.txt

# Baseline / format reference submission
python starter/make_sample_submission_csv.py --data-root data --split test  --output sample_submission.csv

# Local scoring loop on the public val split
python starter/make_sample_submission_csv.py --data-root data --split val --output val_sample_submission.csv
python starter/validate_submission_csv.py    --submission val_sample_submission.csv --data-root data --split val

# ALWAYS validate before submitting
python starter/validate_submission_csv.py --submission submission.csv --data-root data --split test
```

---

## Common pitfalls checklist

- [ ] No pretrained/external weights anywhere; `pretrained=False` set explicitly.
- [ ] `timm` used for layers only — no `pretrained=True`, no checkpoint download.
- [ ] `class_id` ↔ `segmentation_id` off-by-one handled (`seg = class + 1`).
- [ ] `1000` (ignore) treated as ignore in loss; **never** emitted in predictions.
- [ ] Predicted ids strictly in `0..300`.
- [ ] Masks resized to original image dims with **nearest-neighbor**.
- [ ] RLE is row-major, 1-indexed, non-overlapping runs; empty/`0` = background.
- [ ] Did not train on `val/` or `test/`.
- [ ] mIoU computed foreground-only when checking against the leaderboard metric.
- [ ] Submission passes `validate_submission_csv.py`.
- [ ] Report lists all data/architectures/packages/tools and states no pretrained
      weights or external data were used.