# CLAUDE.md — CSE 164 Final Project 2026

Semi-supervised image **classification** + **semantic segmentation** from scratch.
For every hidden test image predict (1) one image-level `class_id` and (2) a
pixel-level segmentation mask. **Segmentation dominates the score.**

Guidance for any coding assistant in this repo — read before writing training,
inference, or submission code. These instructions OVERRIDE default behavior.

---

## ⛔ Hard rule: NO pretrained weights (project fails if violated)

Every weight in a final-submission model must come from training on the
**provided competition data only**. Easiest way to get a zero, easy to trigger by
accident.

**Forbidden:** `torchvision`/`timm`/HF `pretrained=True` / `from_pretrained(...)`;
loading any `.pth`/`.ckpt`/`.safetensors` from the internet; ImageNet/COCO/SSL/
foundation checkpoints; external labeled data or annotations.

**Allowed:** public *architecture code* if **randomly initialized**; public
packages/docs/assistants; **self-supervised FCMAE pre-training on the competition's
own images** (this is "training on provided data," and is the core of our plan);
**`timm` for layer utils only** (`trunc_normal_`, `DropPath` — used in
`models/convnext/convnextv2.py`). NEVER `timm.create_model(..., pretrained=True)`.

When adding model code: set `weights=None`/`pretrained=False` explicitly; never
`load_state_dict()` anything not produced by this project's own training. The final
report must state no pretrained weights and no external data were used.

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
- `val/` is for model selection only — **never train on it.** Same for `test/`.
- `train_unlabeled/` has **distractors from non-target classes** — filter them
  (confidence thresholding) before pseudo-labeling.

---

## Label & mask encodings (get these exactly right)

**Classification:** integer `class_id` in `0..299`.

**Segmentation masks (PNG, RGB-encoded):** `segmentation_id = R + G * 256`.

| segmentation_id | meaning |
|---|---|
| `0` | background / non-target |
| `1..300` | foreground, where `segmentation_id = class_id + 1` |
| `1000` | ignore region — GT only, unscored |

Critical: foreground `class_id k` ↔ `segmentation_id k+1` (cls class 0 = seg id 1;
background is its own id 0). **Off-by-one silently tanks the score.**

**Prediction constraints:** ids in `0..300` only (**never emit `1000`**); treat
`1000` GT pixels as **ignore** in the loss (`IGNORE_IDX = 1000`, NOT 255 — 255
collides with foreground id 255 / class 254); each predicted mask must match its
input's exact W/H — resize back with **nearest-neighbor** (never bilinear).

---

## Submission format

`submission.csv`, one row per test image:

```csv
image,class_id,segmentation_rle
test_00000.JPEG,17,1 20 18 210 5 18
test_00001.JPEG,3,
```

- `segmentation_rle` — **row-major, 1-indexed RLE triples** `start length value …`;
  `value` in `1..300` (background is just the gaps); empty / `0` = all-background.
- `core.utils.encode_mask_ids` emits `""` for all-background; `decode_rle_to_mask`
  reverses it (the official decoder treats `""`/`0`/NaN the same).
- Validation rejects: missing/dup rows, bad filenames, invalid ids, malformed RLE,
  **overlapping runs**, out-of-range ids. **Always run `starter/validate_submission_csv.py`
  before submitting.**

---

## What the score rewards (optimize for this, not raw mIoU)

```text
Kaggle = 70% segmentation + 20% classification     (10% report graded off-Kaggle)
segmentation = 70% mean IoU + 20% boundary F-score + 10% rare-class mIoU
```

- mIoU and rare-class mIoU are **foreground-only** (background excluded from the
  per-class average) — predicting background well earns nothing directly.
- Boundary F-score is 20% of seg → boundaries matter (boundary-aware loss / post-proc).
- Rare-class mIoU + macro accuracy are class-balanced → protect the long tail
  (class-balanced sampling). Prioritize seg effort; don't let boundaries blur.

Suggested directions (from the spec): strong augmentation, multi-task shared
encoder, pseudo-labeling high-confidence unlabeled images, distractor confidence
filtering, boundary-aware loss/post-proc, class-balanced sampling, TTA, ensembling.

---

## Committed architecture — shared-encoder self-supervised pipeline

Limited training time + small labeled set ⇒ squeeze signal from the 50k unlabeled
images via masked image modeling (the strongest SSL technique per lecture).

1. **Backbone — ConvNeXt V2** (from-scratch arch code adapted from
   facebookresearch/ConvNeXt-V2). Two forms sharing identical structure (so weights
   line up): *dense* (`convnextv2.py`, finetune/inference) and *sparse*
   (`convnextv2_sparse.py`, the FCMAE encoder — masked patches cost no compute).
   Keep **GRN** (V2's fix for FCMAE feature-collapse) in for the whole run;
   LayerScale is dropped in V2.
2. **Pre-training — FCMAE** (`fcmae.py`): mask ~60% of patches, reconstruct pixels,
   MSE on masked patches only. Architecture + framework are co-designed (FCMAE or
   GRN alone barely help; together they pop).
3. **Segmentation head — UPerNet** (`upernet/upernet.py`, `build_upernet`): emits
   **301 channels** (seg ids 0..300) so `argmax` gives the seg id directly. Seg =
   70% of score, the most important downstream task.
4. **Classification head** — linear head on pooled backbone features.

**Dense backbone serves two heads (one shared encoder):** classification uses
`forward_features(x)` → GAP + LayerNorm → `(N,C)` → `head`; segmentation uses
`forward_features_seg(x)` → **4-tuple of multi-scale maps** (strides 4/8/16/32) for
UPerNet. The seg-path norms are new/random and only on the seg path (not in the
FCMAE checkpoint), so pre-trained weights load with **`strict=False`**. This
multi-scale extractor is the only structural change V2 needed.

**Pre-training data:** pool the three TRAINING splits only — 50k unlabeled + 7.5k
labeled (labels dropped) + 3k seg (masks dropped) ≈ **60.5k images** (val/test walled
off). Finetune each head on its own labeled subset afterward; pseudo-labeling is a
later stretch goal. **Favor small backbones** (`atto` ≈ 3.7M) — the dataset is small.

`resnet/` + `unet/` are from-scratch baselines kept for comparison; the ConvNeXt V2
+ FCMAE + UPerNet path is the real plan.

---

## Repository structure (check this FIRST before filesystem-searching)

Structure only — for status see the table below; for the detailed log see
`cooking/CHANGELOG.md`. Excludes `data/`, `__pycache__/`, `venv/`, asset dirs.
`src/` uses **implicit namespace packages (no `__init__.py` anywhere)**.

```text
cse164-final-project/
├── CLAUDE.md                     # this file — read FIRST
├── pipeline.py                   # root CLI entrypoint (stub)
├── requirements.txt              # torch (cu126), torchvision, timm, spconv, …
├── compose.yaml / docker/        # CUDA 12.6 + PyTorch 2.12 image
├── checkpoints/ outputs/ logs/ cache/
├── cooking/                      # research notes (NOT code)
│   ├── PROJECT_SPEC.md  IDEATION.md  CHANGELOG.md  CONVNEXT*.md  LECTURES.md …
│   └── references/ConvNeXt-V2/    # local clone, read-only — observe, never import
├── starter/                      # course-provided, DO NOT EDIT
│   ├── validate_submission_csv.py # ALWAYS run before submitting
│   └── kaggle_metric.py  make_sample_submission_csv.py
└── src/
    ├── core/
    │   ├── utils.py              # constants, PretrainConfigs, seed/device/root,
    │   │                         #   ALL kaggle metrics (RLE, IoU, boundary, cls),
    │   │                         #   seg↔cls id maps; configures stdlib logging on import
    │   └── dataset.py            # 5 split Datasets + PretrainDataset + NORM_MEAN/STD
    ├── engines/                  # per-epoch loops (one file per task)
    │   ├── fcmae_pretrain.py     #   train_one_epoch (image-only, AMP, cosine) — DONE
    │   ├── convnext_cls.py       #   cls loop — STUB
    │   ├── convnext_upernet_seg.py  # seg loop — EMPTY
    │   └── utils.py              #   adjust_learning_rate (warmup + cosine)
    ├── runners/                  # per-stage drivers (N epochs, ckpt, selection)
    │   ├── pretrain.py           #   FCMAE harness — DONE (Docker/GPU run)
    │   ├── classifier.py         #   cls driver — EMPTY
    │   ├── segmenter.py          #   seg driver — EMPTY
    │   └── utils.py              #   LossScaler (AMP grad scaler)
    ├── tests/                    # prediction → submission — STALE (old-structure imports)
    │   ├── test_cls.py  test_seg.py
    └── models/
        ├── convnext/             # convnextv2.py (dense) · convnextv2_sparse.py (spconv, WIP)
        │                         #   · fcmae.py · utils.py (LayerNorm, GRN, Sparse* spconv wrappers)
        ├── upernet/upernet.py    # build_upernet (BN not SyncBN; num_class=301)
        ├── resnet/resnet.py      # build_resnet — cls baseline
        └── unet/unet.py          # build_unet  — seg baseline
```

---

## Status & next steps

| Area | State |
|---|---|
| `core/utils.py`, `core/dataset.py` | ✅ done (metrics, datasets, configs, logging) |
| `engines/utils.py`, `engines/fcmae_pretrain.py` | ✅ done |
| `runners/pretrain.py`, `runners/utils.py` | ✅ done (`run`/`build_transform`/`build_model`/`show_modeled_image`/argparse `main`; `build_param_groups` via timm, `save`/`load_checkpoint`). End-to-end blocked only by the sparse encoder below |
| `models/convnext/convnextv2.py` (dense), `upernet/upernet.py` | ✅ usable |
| `models/convnext/convnextv2_sparse.py` (FCMAE encoder) | 🔄 **spconv port WIP — current blocker** |
| `engines/convnext_cls.py`, `convnext_upernet_seg.py` | ⬜ stub / empty |
| `runners/classifier.py`, `segmenter.py` | ⬜ empty |
| `src/tests/*` | ⬜ stale — rewrite against current `core/`/`runners/` |

**Immediate blocker:** the sparse port uses `spconv.pytorch.SubMConv2d` for the
depthwise 7×7 conv, but spconv asserts `groups == 1` ("don't support groups") — so
`FCMAE.convnextv2_atto()` can't instantiate and pretraining can't run end-to-end
yet. Resolve the depthwise-conv strategy in `convnextv2_sparse.py` before anything
downstream of FCMAE.

**Order of work:** (1) fix sparse depthwise conv → FCMAE runs; (2) flesh out
`engines/convnext_cls.py` + `convnext_upernet_seg.py`; (3) write `runners/classifier.py`
+ `segmenter.py` (warm-start encoder from the FCMAE ckpt, `strict=False`); (4) rewrite
`tests/` → `submission.csv`; (5) wire `pipeline.py`. Stretch (pseudo-labeling,
distractor filtering, TTA, ensembling, boundary post-proc) stays last.

**Per-goal cadence:** at the end of every goal — update this CLAUDE.md status +
`cooking/CHANGELOG.md` (detailed what/why/files/test), then flag a `/compact` point.

---

## Module gotchas (learned the hard way)

- **No `__init__.py`** — use package-relative imports (`from ..core.utils import …`)
  and run from the **project root** as `python -m src.runners.pretrain` (NOT from
  inside `src/`). No convenience re-exports.
- **Shared helpers + metrics + constants all live in `src/core/utils.py`** (also
  configures stdlib `logging` to `logs/<date>.log` on import). There is **no**
  `core/log.py`, `core/optim.py`, `core/evaluate.py`, or `core/logging.py` — log via
  `logging.getLogger(__name__)`; build optimizer param groups inline; metrics are in
  `utils.py`. (The old CLAUDE.md referenced those modules — they don't exist.)
- **`adjust_learning_rate`** lives in `engines/utils.py`; **`LossScaler`** (AMP) in
  `runners/utils.py`. There is no `NativeScaler` / `MetricLogger` / `remap_checkpoint_keys`.
- **Sparse stack uses `spconv` (not MinkowskiEngine).** spconv **is** pip-installable
  and is in the venv, so both dense and sparse import on Windows — but the sparse port
  is **WIP** (see blocker above), so FCMAE can't run end-to-end locally yet.
- `IGNORE_IDX = 1000` (NOT 255). Loss must ignore it; never emit it.
- `PretrainConfigs` (in `core/utils.py`) carries `epochs`/`warmup_epochs`/`min_lr`/
  `mask_ratio`; the FCMAE engine takes it plus a constant base `lr` and anneals from
  there — do **not** overwrite that base `lr` with the live group lr inside the loop.

---

## How to run

```bash
# Dev env. Docker provides CUDA 12.6 + PyTorch 2.12. Bare-metal venv also works for
# the dense + (once fixed) sparse path since spconv is pip-installable.
docker compose up notebook -d        # JupyterLab on :8888
venv\Scripts\python.exe -m src.runners.pretrain --epochs 1 --limit 64 --num-workers 0

# Import / compile gates (run from project root)
venv\Scripts\python.exe -c "import src.engines.fcmae_pretrain"
venv\Scripts\python.exe -m py_compile src\runners\pretrain.py

# Submission validation — ALWAYS before submitting
python starter/validate_submission_csv.py --submission submission.csv --data-root data --split test
```

---

## Common pitfalls checklist

- [ ] No pretrained/external weights; `pretrained=False` set explicitly.
- [ ] `timm` used for layers only — no `pretrained=True`, no checkpoint download.
- [ ] `class_id` ↔ `segmentation_id` off-by-one handled (`seg = class + 1`).
- [ ] `1000` (ignore) treated as ignore in loss; **never** emitted in predictions.
- [ ] Predicted ids strictly in `0..300`.
- [ ] Masks resized to original image dims with **nearest-neighbor**.
- [ ] RLE row-major, 1-indexed, non-overlapping; empty/`0` = background.
- [ ] Did not train on `val/` or `test/`.
- [ ] mIoU computed foreground-only when checking against the leaderboard metric.
- [ ] Submission passes `validate_submission_csv.py`.
- [ ] Report lists all data/architectures/packages and states no pretrained weights
      or external data were used.
