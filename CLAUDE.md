# CLAUDE.md — CSE 164 Final Project 2026

Semi-supervised image **classification** + **semantic segmentation** from scratch.
Per hidden test image, predict one `class_id` + a pixel mask. **Segmentation
dominates the score.** Read before writing training/inference/submission code.
These instructions OVERRIDE default behavior.

---

## ⛔ Hard rule: NO pretrained weights (project fails if violated)

Every final-model weight must come from training on **provided competition data
only**. Easy to trigger by accident → instant zero.

- **Forbidden:** `pretrained=True` / `from_pretrained(...)`; any `.pth`/`.ckpt`/
  `.safetensors` from the internet; ImageNet/COCO/SSL/foundation checkpoints;
  external labeled data.
- **Allowed:** public *architecture code* if **randomly initialized**; packages/
  docs/assistants; **self-supervised FCMAE pretraining on the competition's own
  images** (core of our plan); **`timm` for layer utils only** (`trunc_normal_`,
  `DropPath`) — NEVER `timm.create_model(..., pretrained=True)`.

Always set `pretrained=False`/`weights=None` explicitly; never `load_state_dict()`
anything not produced by this project. Report must state no pretrained weights / no
external data.

---

## Data layout

```text
data/
├── train_labeled/images/        # 7,500 imgs, IMAGE-LEVEL labels only
├── train_seg/{images,masks}/    # 3,000 imgs WITH masks
├── train_unlabeled/images/      # 50,000 imgs, NO labels (+ distractors)
├── val/{images,masks}/, classification.json   # 750, fully labeled
├── test/images/                 # 3,000 hidden test images
└── metadata/{class_map,train_labeled,train_seg}.json
```

- **300 target classes.** `.JPEG`, variable resolution.
- **Never train on `val/` or `test/`** (model selection only).
- `train_unlabeled/` has **distractors** — confidence-filter before pseudo-labeling.

---

## Label & mask encodings (off-by-one silently tanks the score)

- **Classification:** `class_id` in `0..299`.
- **Masks (PNG, RGB-encoded):** `segmentation_id = R + G*256`.
  - `0` = background · `1..300` = foreground (`segmentation_id = class_id + 1`) ·
    `1000` = ignore (GT only, unscored).
- foreground `class_id k` ↔ `segmentation_id k+1`; background is its own id `0`.
- **Predictions:** ids in `0..300` only (**never emit `1000`**); treat `1000` GT as
  ignore in loss (`IGNORE_IDX = 1000`, NOT 255 — 255 collides with class 254);
  resize masks back to exact input W/H with **nearest-neighbor**.

---

## Submission format

`submission.csv`, one row per test image:

```csv
image,class_id,segmentation_rle
test_00000.JPEG,17,1 20 18 210 5 18
test_00001.JPEG,3,
```

- `segmentation_rle` — **row-major, 1-indexed RLE triples** `start length value …`;
  `value` in `1..300` (background = gaps); empty/`0` = all-background.
- `core.utils.encode_mask_ids` emits `""` for all-background; `decode_rle_to_mask`
  reverses it.
- **Always run `starter/validate_submission_csv.py` before submitting** (rejects
  dup/missing rows, bad filenames, invalid ids, overlapping/malformed RLE).

---

## Scoring (optimize for this, not raw mIoU)

```text
Kaggle = 70% segmentation + 20% classification     (10% report off-Kaggle)
segmentation = 70% mean IoU + 20% boundary F-score + 10% rare-class mIoU
```

- mIoU + rare-class mIoU are **foreground-only** (background excluded) — predicting
  background earns nothing directly.
- Boundaries matter (20% of seg) → boundary-aware loss/post-proc.
- Rare-class mIoU + macro-acc are class-balanced → protect the long tail
  (class-balanced sampling).

Suggested levers: strong aug, multi-task shared encoder, pseudo-labeling,
distractor filtering, boundary-aware loss, class-balanced sampling, TTA, ensembling.

---

## Committed architecture — shared-encoder SSL pipeline

Small labeled set ⇒ mine the 50k unlabeled imgs via masked image modeling.

1. **Backbone — ConvNeXt V2** (from-scratch arch, adapted from
   facebookresearch/ConvNeXt-V2). Two structurally identical forms: *dense*
   (`convnextv2.py`, finetune/inference) and *sparse* (`convnextv2_sparse.py`, FCMAE
   encoder). Keep **GRN**; LayerScale dropped in V2.
2. **Pretrain — FCMAE** (`fcmae.py`): mask ~60%, reconstruct pixels, MSE on masked
   patches only. GRN + FCMAE are co-designed (either alone barely helps).
3. **Seg head — UPerNet** (`upernet/upernet.py`, `build_upernet`): **301 channels**
   (ids 0..300) so `argmax` = seg id.
4. **Cls head** — linear on pooled features.

**One shared encoder, two heads:** cls uses `forward_features(x)` → GAP+LayerNorm →
`head`; seg uses `forward_features_seg(x)` → **4 multi-scale maps** (strides
4/8/16/32) for UPerNet. Seg-path norms are new/random → load FCMAE weights with
**`strict=False`**.

**Pretrain data:** the 3 TRAINING splits only (50k unlabeled + 7.5k + 3k, labels/
masks dropped) ≈ **60.5k imgs** (val/test walled off). **Favor small backbones**
(`atto` ≈ 3.7M). `resnet/` + `unet/` are from-scratch baselines for comparison.

---

## Repository structure (check FIRST before filesystem-searching)

Structure only; detailed log in `cooking/CHANGELOG.md`. `src/` uses **implicit
namespace packages (no `__init__.py`)** — run from project root as
`python -m src.runners.pretrain`.

```text
cse164-final-project/
├── CLAUDE.md  pipeline.py(stub)  requirements.txt  compose.yaml / docker/
├── checkpoints/ outputs/ logs/ cache/
├── cooking/                      # research notes (NOT code); references/ConvNeXt-V2/ read-only — never import
├── starter/                      # course-provided, DO NOT EDIT (validate_submission_csv.py, kaggle_metric.py)
└── src/
    ├── core/
    │   ├── utils.py              # constants, PretrainConfigs, seed/device/root, ALL kaggle metrics
    │   │                         #   (RLE/IoU/boundary/cls), seg↔cls maps; configs logging on import
    │   └── dataset.py            # 5 split Datasets + PretrainDataset + NORM_MEAN/STD
    ├── engines/                  # per-epoch loops: fcmae_pretrain.py(done) convnext_cls.py(stub)
    │   │                         #   convnext_upernet_seg.py(empty); utils.py = adjust_learning_rate
    ├── runners/                  # per-stage drivers: pretrain.py(done) classifier.py(empty)
    │   │                         #   segmenter.py(empty); utils.py = LossScaler (AMP)
    ├── tests/                    # prediction→submission — STALE (old imports)
    └── models/
        ├── convnext/             # convnextv2.py(dense) · convnextv2_sparse.py(spconv, FCMAE encoder runs)
        │                         #   · fcmae.py · utils.py (LayerNorm, GRN, Sparse* wrappers)
        ├── upernet/upernet.py    # build_upernet (BN; num_class=301)
        ├── resnet/resnet.py  unet/unet.py   # from-scratch cls/seg baselines
```

---

## Status & next steps

Done: `core/*`, `engines/utils.py`, `engines/fcmae_pretrain.py`,
`runners/pretrain.py`+`utils.py`, `convnextv2.py`(dense), `convnextv2_sparse.py`+
`fcmae.py` (FCMAE forward+backward runs on CUDA), `upernet.py`.
WIP/empty: `engines/convnext_cls.py`+`convnext_upernet_seg.py`,
`runners/classifier.py`+`segmenter.py`, `tests/`(stale).

**Sparse depthwise — RESOLVED.** spconv's `SubMConv2d` asserts `groups == 1`, so the
depthwise 7×7 is done as a per-channel `nn.ModuleList` of 1-ch `SubMConv2d`
(`SparseDepthwiseConv`, `convnext/utils.py`). Two fixes were needed to run:
(a) `SparseLinear` exposes no `.bias` (only `.linear`) — `_init_weights` must guard on
`m.linear.bias`, not `m.bias` (was an `AttributeError` at construction);
(b) `large_kernel_fast_algo=True` triggers an NVRTC build failure for single-channel
kernels — must stay `False`. `FCMAE.convnextv2_atto()` now runs end-to-end on GPU.
NOTE: spconv kernels are **CUDA-only** (CPU forward asserts `implicit gemm only
support cuda`); the per-channel loop is functional but not perf-optimal.

**Order:** (1) ~~fix sparse depthwise → FCMAE runs~~ ✓; (2) `engines/convnext_cls.py` +
`convnext_upernet_seg.py`; (3) `runners/classifier.py` + `segmenter.py` (warm-start
from FCMAE ckpt, `strict=False`); (4) rewrite `tests/` → `submission.csv`; (5) wire
`pipeline.py`. Stretch (pseudo-labeling, distractor filtering, TTA, ensembling,
boundary post-proc) last.

**Cadence:** end of every goal — update this status + `cooking/CHANGELOG.md`, flag a
`/compact` point.

---

## Gotchas

- **No `__init__.py`** — package-relative imports (`from ..core.utils import …`); run
  from project root. No re-exports.
- **All helpers/metrics/constants live in `src/core/utils.py`** (also configs stdlib
  `logging` to `logs/<date>.log` on import). No `core/log.py`/`optim.py`/`evaluate.py`/
  `logging.py` — log via `logging.getLogger(__name__)`; build param groups inline.
- `adjust_learning_rate` in `engines/utils.py`; `LossScaler` in `runners/utils.py`.
  No `NativeScaler`/`MetricLogger`/`remap_checkpoint_keys`.
- Sparse uses **`spconv`** (not Minkowski); pip-installable, in venv. **CUDA-only** —
  spconv forward asserts `implicit gemm only support cuda`, so the FCMAE encoder can't
  run/import-test on CPU; verify on GPU. Depthwise 7×7 is a per-channel `ModuleList`
  (`SparseDepthwiseConv`); keep `large_kernel_fast_algo=False` (NVRTC build fails for
  1-ch kernels). `SparseLinear` has no `.bias` attr — use `m.linear.bias` in init.
- `IGNORE_IDX = 1000` (NOT 255) — ignore in loss, never emit.
- `PretrainConfigs` carries `epochs`/`warmup_epochs`/`min_lr`/`mask_ratio`; FCMAE
  engine anneals from a constant base `lr` — do **not** overwrite it with live group lr.

---

## How to run (from project root)

```bash
docker compose up notebook -d        # JupyterLab :8888 (CUDA 12.6 + PyTorch 2.12)
venv\Scripts\python.exe -m src.runners.pretrain --epochs 1 --limit 64 --num-workers 0
venv\Scripts\python.exe -c "import src.engines.fcmae_pretrain"   # import gate
python starter/validate_submission_csv.py --submission submission.csv --data-root data --split test
```
