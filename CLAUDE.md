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
   (`convnextv2.py`, finetune/inference) and *pseudo-sparse* (`convnextv2_pseudosparse.py`,
   FCMAE encoder — plain dense convs/linears with a binary keep-mask multiplied in,
   NOT spconv/Minkowski). Same param names/shapes in both, so warm-start is a clean
   prefix-strip (see below). Keep **GRN**; LayerScale dropped in V2.
2. **Pretrain — FCMAE** (`fcmae.py`): mask ~60%, reconstruct pixels, MSE on masked
   patches only. GRN + FCMAE are co-designed (either alone barely helps).
3. **Seg head — UPerNet** (`upernet/upernet.py`, `build_upernet`): **301 channels**
   (ids 0..300) so `argmax` = seg id.
4. **Cls head** — linear on pooled features.

**One shared encoder, two heads:** cls uses `forward_features(x)` → GAP+LayerNorm →
`head`; seg uses `forward_features_seg(x)` → **4 multi-scale maps** (strides
4/8/16/32) for UPerNet. Seg-path norms are new/random → load FCMAE weights with
**`strict=False`**.

**Warm-start mechanics (verified):** the FCMAE checkpoint stores the backbone under
`encoder.*` plus decoder-only keys (`proj`/`decoder`/`pred`/`mask_token`). To load
it into a dense `ConvNeXtV2`, **strip the `encoder.` prefix**, drop the decoder keys,
then `load_state_dict(strict=False)` — `downsample_layers.*`/`stages.*` match 1:1
(for `atto`: 136/136 encoder tensors, 0 unexpected); only `norm`/`head`/`seg_norms`
(12 tensors) are correctly left random. The checkpoint also stores its
`PretrainConfigs`, so read `ckpt["config"].model_size` to build the matching size.

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
    ├── engines/                  # per-epoch loops: fcmae_pretrain.py(done) convnext_cls.py(done:
    │   │                         #   train_one_epoch + predict_one_epoch) convnext_upernet_seg.py(empty);
    │   │                         #   utils.py = adjust_learning_rate
    ├── runners/                  # per-stage drivers: pretrain.py(done) classifier.py(stub — flow
    │   │                         #   currently lives in notebook + tmp_pretrain_finetune.py)
    │   │                         #   segmenter.py(empty); utils.py = LossScaler (AMP) + save/load_checkpoint
    ├── tests/                    # prediction→submission — STALE (old imports)
    └── models/
        ├── convnext/             # convnextv2.py(dense) · convnextv2_pseudosparse.py(FCMAE encoder,
        │                         #   dense+mask, NO spconv) · fcmae.py · utils.py (LayerNorm, GRN only)
        ├── upernet/upernet.py    # build_upernet (BN; num_class=301)
        ├── resnet/resnet.py  unet/unet.py   # from-scratch cls/seg baselines
```

`tmp_pretrain_finetune.py` (project root, scratch) = the notebook's pretrain→finetune→
val flow as a plain script (no notebook overhead).

---

## Status & next steps

Done: `core/*`, `engines/utils.py`, `engines/fcmae_pretrain.py`,
`engines/convnext_cls.py` (train + eval), `runners/pretrain.py`+`utils.py`,
`convnextv2.py`(dense), `convnextv2_pseudosparse.py`+`fcmae.py` (FCMAE forward+backward
runs on CUDA), `upernet.py`. Cls finetune flow (warm-start + train + val) is wired in
`notebook.ipynb` and `tmp_pretrain_finetune.py`.
WIP/empty: `engines/convnext_upernet_seg.py`, `runners/classifier.py`(stub)+
`segmenter.py`, `tests/`(stale).

**spconv DROPPED → pseudo-sparse.** The FCMAE encoder no longer uses spconv/Minkowski.
The old `convnextv2_sparse.py` (per-channel `SubMConv2d` depthwise, `Sparse*` wrappers)
was **deleted**; the encoder is now `convnextv2_pseudosparse.py` — ordinary dense
`nn.Conv2d`/`nn.Linear` with a binary keep-mask multiplied in after each op. All spconv
imports removed (`models/convnext/utils.py`, `fcmae.py`), so the whole pipeline now
**imports + runs on CPU** (no CUDA-only constraint, no Windows install pain). `utils.py`
keeps only `LayerNorm` + `GRN`.

**Cls warm-start — VERIFIED loading real weights.** Empirically confirmed against the
on-disk `checkpoint-*.pth`: strip `encoder.` → `load_state_dict(strict=False)` loads
all 136/136 encoder tensors (0 unexpected), values match the checkpoint and differ from
random init; the 12 `norm`/`head`/`seg_norms` tensors stay random *by design*. See the
"Warm-start mechanics" note above.

**Order:** (1) ~~FCMAE runs~~ ✓; (2) ~~`engines/convnext_cls.py`~~ ✓ + `convnext_upernet_seg.py`;
(3) fold the notebook cls flow into `runners/classifier.py` + build `segmenter.py`
(warm-start from FCMAE ckpt, `strict=False`); (4) rewrite `tests/` → `submission.csv`;
(5) wire `pipeline.py`. Stretch (pseudo-labeling, distractor filtering, TTA, ensembling,
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
- `adjust_learning_rate` in `engines/utils.py`; `LossScaler`/`save_checkpoint`/
  `load_checkpoint` in `runners/utils.py`. No `NativeScaler`/`MetricLogger`/
  `remap_checkpoint_keys` (the `encoder.` prefix-strip is done inline at the call site).
- **No spconv.** The FCMAE encoder is pseudo-sparse (`convnextv2_pseudosparse.py`):
  dense convs × keep-mask, runs on CPU or GPU. Don't reintroduce `import spconv` or the
  `Sparse*` wrappers; `models/convnext/utils.py` is `LayerNorm` + `GRN` only.
- **CUDA cache:** the reference (`cooking/references/ConvNeXt-V2/engine_pretrain.py:52`)
  calls `torch.cuda.empty_cache()` every optimizer step — that was a Minkowski-Engine
  workaround. We do **NOT** need it (pseudo-sparse leaks nothing); it stays commented
  out in `fcmae_pretrain.py` — a per-step `empty_cache()` only adds sync stalls.
- `load_checkpoint`/`torch.load` our own ckpts with **`weights_only=False`** — they
  pickle a `PretrainConfigs`/`ClsFinetuneConfigs` dataclass (PyTorch ≥2.6 defaults to
  `weights_only=True`, which rejects it).
- `IGNORE_IDX = 1000` (NOT 255) — ignore in loss, never emit.
- `PretrainConfigs` carries `epochs`/`warmup_epochs`/`min_lr`/`mask_ratio`; FCMAE
  engine anneals from a constant base `lr` — do **not** overwrite it with live group lr.
- `ValDataset` yields a **4-tuple** `(img, seg_mask, cls_label, orig_size)` with masks
  at ORIGINAL (variable) size — can't default-collate a batch; use a `collate_fn` that
  keeps just `img`+`cls_label` for a cls val loader.

---

## How to run (from project root)

```bash
docker compose up notebook -d        # JupyterLab :8888 (CUDA 12.6 + PyTorch 2.12)
venv\Scripts\python.exe -m src.runners.pretrain --epochs 1 --limit 64 --num-workers 0
venv\Scripts\python.exe -c "import src.engines.fcmae_pretrain"   # import gate (CPU ok now — no spconv)
venv\Scripts\python.exe tmp_pretrain_finetune.py                 # pretrain → cls finetune → val
python starter/validate_submission_csv.py --submission submission.csv --data-root data --split test
```

GPU still recommended for real runs, but the pipeline now imports and forward/backward
runs on CPU too (handy for smoke tests on Windows without the Docker GPU image).
