# CHANGELOG — detailed change summary

Granular dev log of code changes (the terse version lives in CLAUDE.md's sprint
log). Newest first. One entry per accomplished goal; each notes *what* changed,
*why*, files touched, and the verification result.

---

## 2026-06-03 — Fix: FCMAE depthwise-conv import (Docker / MinkowskiEngine)

### `MinkowskiDepthwiseConvolution` missing in our CUDA-12 ME fork → aliased to `MinkowskiChannelwiseConvolution`
- **Symptom:** running the sparse/FCMAE path in Docker raised
  `ImportError: cannot import name 'MinkowskiDepthwiseConvolution' from
  'MinkowskiEngine'`. The build's import sanity-check still passed — MinkowskiEngine
  imports fine; only that one symbol was absent.
- **Root cause:** the image builds ME from the `CiSong10/MinkowskiEngine`
  `cuda12-installation` fork, which is v0.5.4 and **predates**
  `MinkowskiDepthwiseConvolution` (a newer NVIDIA addition). Introspection showed
  the fork exposes `MinkowskiChannelwiseConvolution` instead — the equivalent
  per-channel (depthwise) sparse conv.
- **Fix:** in `convnextv2_sparse.py` and `fcmae.py`, import
  `MinkowskiChannelwiseConvolution as MinkowskiDepthwiseConvolution`. Verified the
  channelwise class is a true drop-in: same ctor `(in_channels, kernel_size, bias,
  dimension)`, allocates a real `.bias` Parameter when `bias=True` (not `None`, so
  the `nn.init.constant_(m.bias, 0)` init paths are safe), and `.kernel` shape
  `(K^D, C)` — so **all** downstream construction (sparse `Block.dwconv`),
  `isinstance` checks, and `_init_weights` calls are unchanged.
- **Files:** `src/models/convnext/convnextv2_sparse.py`,
  `src/models/convnext/fcmae.py` (import line + explanatory comment only).
- **Verified (Docker, GPU):** channelwise conv forward OK; full `FCMAE` atto
  (`dims=[40,80,160,320]`) forward on `(2,3,224,224)` → finite loss,
  `pred (2,3072,7,7)` (=32²·3 patch pixels), `mask (2,49)` (=7² patches). This
  unblocks the Phase 4.3 FCMAE pre-training harness.

---

## 2026-06-02 — Phase 4.2: classification trainer (`trainers/train_cls.py`)

### Full train → val (top-1/macro-acc) → checkpoint driver (runs locally on real data)
- New file (was a 0-byte stub). Mirrors `train_seg.py` but for the classification
  task — the second finetune stage that runs end-to-end on the competition data
  from random init.
- **Structure:**
  - `MODELS` registry (same atto/femto/nano/tiny → `(factory, dims)`); dims kept
    only for ckpt provenance/parity (the cls path itself needs just the factory).
  - `build_model`: dense backbone `factory(num_classes=300, drop_path_rate=...)`,
    so `model(x)` = `forward_features` (GAP + final LayerNorm) → linear `head` →
    `(N, 300)` logits, `class_id` 0..299 (no off-by-one — that only bites seg). If
    `--finetune` given, warm-start the **encoder only** from an FCMAE ckpt:
    `remap_checkpoint_keys` + `load_state_dict(strict=False)` so the random-init
    `head` (absent from the FCMAE ckpt) survives.
  - `_val_collate`: a tiny collate for the val `DataLoader` that stacks **only**
    image + label from `ValDataset`'s 5-tuple `(image, label, seg_id, name,
    orig_size)`. The GT `seg_id` masks are native-res / variable-size and default
    collate can't stack them; the cls `evaluate` reads only `batch[0:2]`, so we
    drop the rest. (Cleaner than `train_seg`'s manual `ValDataset` iteration —
    classification needs no masks at all, so a standard batched DataLoader works.)
  - `main`: `TrainLabeledDataset` loader (`--limit-train N` → `Subset` for smoke);
    `nn.CrossEntropyLoss`; `NativeScaler`; per-epoch cls engine `train_one_epoch`;
    every `--eval-every` epochs (and last) run cls engine `evaluate` on all 750
    val, save `last.pth` always and `best.pth` when **`macro_acc`** improves.
    Checkpoints bundle backbone(`model`)+optimizer+scaler+epoch+args+scores+dims.
- **Selection metric = `macro_acc`, NOT top-1.** The leaderboard classification
  metric is class-balanced (mean per-class recall), so a model that nails common
  classes but drops the long tail must not be preferred. Same reasoning as the
  cls engine's dual-metric `evaluate`.
- **DEVIATION (same as 4.1, documented inline + CLAUDE.md):** plain
  `torch.optim.AdamW` over `model.parameters()`, NOT `core.optim.create_optimizer`
  with layer-wise LR decay. Layer decay matters for *pretrained* finetune; defer
  to when FCMAE weights land.
- *Test (GPU, real data, `tmp/cls_ckpt/`):* `--epochs 1 --limit-train 64
  --batch-size 16 --eval-every 1` → train loss 6.08→5.94 (near ln(300)=5.70
  chance, untrained); full 750-val `evaluate` returns acc1 0.0040 / macro_acc
  0.0033 (≈1/300 = chance, as expected from a random-init net over 300 classes);
  `best.pth`/`last.pth` written. Separate `test_cls_resume.py` reloads `best.pth`
  **strict into a fresh atto backbone (incl. `head`)** — PASS. Import check green.
- *Files:* `src/trainers/train_cls.py` (new). No other files touched.
- *Status:* Phase 4.2 ✅. Both finetune trainers (seg 4.1 + cls 4.2) now run
  locally end-to-end from random init. Remaining in Phase 4: 4.3 `train_pretrain`
  (FCMAE harness, Docker-only). Phase 5 (`tests/` → submission) is now also
  unblocked since both finetune checkpoints can be produced locally.

---

## 2026-06-02 — Phase 4.1: segmentation trainer (`trainers/train_seg.py`) — FIRST end-to-end stage

### Full train → val-score → checkpoint driver (runs locally on real data)
- New file (was a 0-byte stub). The first driver that ties the whole stack
  together and **actually runs on the competition data** from random init.
- **Structure:**
  - `MODELS` registry maps `--model` name → `(size factory, per-stage dims)` for
    atto/femto/nano/tiny; dims feed both `build_upernet`'s `fpn_inplanes` and
    `fc_dim`.
  - `build_models`: dense backbone (`num_classes=300` for the unused cls head;
    `drop_path_rate=--drop-path`) + `build_upernet(dims, num_class=301)`. If
    `--finetune` is given, warm-start the **encoder only** from an FCMAE checkpoint:
    `remap_checkpoint_keys(state)` then `load_state_dict(strict=False)` so the
    randomly-init `seg_norms` + `head` (absent from the FCMAE ckpt) survive.
  - `evaluate_val`: iterates `ValDataset` **directly** (NOT via a DataLoader — the
    GT masks are native-resolution / variable-size and default collate can't stack
    them). Per image: forward at crop res → `argmax` seg ids 0..300 →
    `resize_mask_nearest(pred, orig_size)` back to native (H,W). `class_id` = majority
    foreground seg id − 1 (placeholder, clamped 0..299 — seg selection reads
    `segmentation_score`, not cls accuracy). Calls `score_val` over all 750 val.
  - `main`: `TrainSegDataset` loader (`--limit-train N` wraps a `Subset` for smoke
    runs); `nn.NLLLoss(ignore_index=1000)`; `NativeScaler`; per-epoch seg engine;
    every `--eval-every` epochs (and last) score val, save `last.pth` always and
    `best.pth` when `segmentation_score` improves. Checkpoints bundle
    backbone(`model`)+`decoder`+optimizer+scaler+epoch+args+scores+dims.
- **DEVIATION (documented inline + CLAUDE.md):** optimizer is a plain
  `torch.optim.AdamW` over `backbone.parameters() + decoder.parameters()`, NOT
  `core.optim.create_optimizer` with layer-wise LR decay. Layer decay pays off when
  finetuning a *pretrained* backbone; from random init it buys little and
  complicates the two-module param grouping (the assigner keys off ConvNeXt param
  names). TODO: wire `create_optimizer` + layer decay when FCMAE weights land.
- *Test (GPU, real data, `tmp/seg_ckpt/`):* `--epochs 1 --limit-train 32
  --batch-size 8 --warmup-epochs 0 --eval-every 1` → training loss 6.33 → 5.90 over
  4 batches (AMP + cosine + grad-clip 1.0 all active; early `grad_norm` avg shows
  `inf` = normal GradScaler scale-calibration skip, smoothed last value 47.9 real);
  **full 750-image `score_val`** returned `{segmentation_score 0.0075, mean_iou
  0.0001, boundary_f 0.0369, rare_miou 0.0, automated 0.0059}` (sensible for an
  essentially-untrained model); `best.pth`+`last.pth` written. Separate reload check
  (`tmp/test_seg_resume.py`): `best.pth` `model`→fresh atto backbone and `decoder`→
  fresh `build_upernet` both `load_state_dict(strict=True)` clean. PASS.

---

## 2026-06-02 — Phase 3.3: FCMAE pre-training engine (`engines/fcmae_pretrain.py`)

### `train_one_epoch` for FCMAE self-supervised pre-training (Docker-only run)
- New file (was a 0-byte stub). One function **`train_one_epoch(model, data_loader,
  optimizer, device, epoch, loss_scaler, args, print_freq=20)`**, mirroring
  `ConvNeXt-V2/engine_pretrain.py` trimmed to single-GPU (no DDP, no
  `all_reduce_mean`, no `synchronize_between_processes`, no tensorboard
  `log_writer`).
- **Contract differences from the reference (deliberate):**
  - Our loader (`build_pretrain_loader` → `ImageOnlyDataset`) yields **image-only**
    batches `(N,3,C,C)` — the reference iterates `(samples, labels)`. So the loop is
    `for step, samples in enumerate(...)` and the model is called
    `model(samples, mask_ratio=mask_ratio)` (no `labels` positional; our FCMAE
    `forward(imgs, labels=None, mask_ratio=0.6)` returns `(loss, pred, mask)`).
  - `mask_ratio` and `update_freq` are read via `getattr(args, ..., default)`
    (0.6 / 1) so a minimal `args` works.
- **Kept from the reference:** optional gradient accumulation over `update_freq`
  (LR adjusted only on accumulation boundaries; `loss /= update_freq`;
  `loss_scaler(..., update_grad=(step+1)%update_freq==0)`); **`torch.cuda.empty_cache()`
  on each true optimizer step** — MinkowskiEngine's allocator fragments over long
  runs and OOMs without this. `MetricLogger` tracks loss/lr (+grad_norm when a step
  actually happens — `NativeScaler` returns `None` on accumulation steps, guarded).
  Raises on non-finite loss.
- **Why import-clean on Windows despite being "Docker-only":** the engine FILE
  imports only `..core.utils` — the FCMAE model (which pulls in MinkowskiEngine via
  the sparse encoder) is passed in as an argument, never imported here. So the
  module imports fine on a bare Windows venv; only *running* it (building the FCMAE
  model) needs the Docker env.
- *Test:* `venv\Scripts\python.exe -c "import src.engines.fcmae_pretrain ..."` →
  `import OK`, `train_one_epoch` present, signature confirmed. No end-to-end run
  (sparse model needs MinkowskiEngine = Docker). This **closes Phase 3** (all three
  engines written).

---

## 2026-06-02 — Phase 3.2: classification engine (`engines/convnext_cls.py`)

### `train_one_epoch` + `evaluate` for ConvNeXt V2 image classification
- New file (was a 0-byte stub). Three functions:
  - **`accuracy_top1(logits, target)`** (`@torch.no_grad`): plain top-1 fraction;
    empty batch → 0.0.
  - **`train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
    loss_scaler, args, max_norm=None, use_amp=True, print_freq=20)`**: mirrors the
    seg engine. Drives the backbone's own `forward` (`model(x)` =
    `forward_features` GAP+LayerNorm → linear `head`), so logits are `(N, 300)` and
    `class_id` is `0..299` with **no off-by-one** (the +1 only bites segmentation).
    Per-iteration `adjust_learning_rate(optimizer, step/len(loader)+epoch, args)`
    (fractional-epoch cosine; `args` needs `lr/min_lr/warmup_epochs/epochs`). AMP
    via `torch.autocast("cuda", float16)` + `NativeScaler(loss, optimizer,
    clip_grad=max_norm, parameters=model.parameters())`; non-AMP fallback does
    manual backward/clip/step. `MetricLogger` tracks loss/acc1/lr/grad_norm; raises
    on non-finite loss. Returns `{meter: global_avg}`.
  - **`evaluate(model, data_loader, device, use_amp=True, num_classes=300,
    print_freq=50)`** (`@torch.no_grad`): reports top-1 **and macro (class-balanced)
    accuracy** = mean per-class recall over classes that appear (each class weighted
    equally). Rationale: the leaderboard's cls metric is class-balanced, so model
    selection must use `macro_acc`, not plain top-1. Reads only `batch[0:2]`, so it
    accepts both `TrainLabeledDataset` `(image, label, name)` 3-tuples and
    `ValDataset` `(image, label, seg_id, name, orig_size)` 5-tuples. Returns the
    `MetricLogger` averages plus `macro_acc`.
- **Design:** classifier head is just the backbone's existing `head` (build with
  `num_classes=300`); the engine adds no model wiring — keeps the model pure and
  loss/metrics in the engine, consistent with the seg engine.
- *Test (`tmp/test_cls_engine.py`, GPU):* atto backbone (`num_classes=300`), a fixed
  8-image `FakeCls` set (`(image, label, name)`, labels in 0..4) over 12 epochs with
  AMP + AdamW + grad-clip 1.0 → averaged loss **5.71 → 0.0085**, acc1 → 1.0 (tiny
  fixed set is fittable, unlike the seg test's random targets). `evaluate` run on a
  `FakeVal` 5-tuple wrapper returns `acc1=1.0, macro_acc=1.0`; `accuracy_top1`
  spot-checks (all-right → 1.0, all-wrong → 0.0). device=cuda. PASS.

---

## 2026-06-02 — Phase 3.1: segmentation engine (`engines/convnext_upernet_seg.py`)

### `train_one_epoch` for ConvNeXt V2 + UPerNet seg (single-GPU, AMP)
- New file (was a 0-byte stub). Three functions:
  - **`seg_forward(backbone, decoder, samples)`**: `forward_features_seg` 4-tuple →
    `decoder` (`log_softmax` at stride-4) → `F.interpolate(..., mode="bilinear")`
    back to the input's spatial size → `(N, num_class, H, W)` log-probs.
  - **`pixel_accuracy(logits, target)`** (`@torch.no_grad`): fraction of correct
    pixels with `target != IGNORE_ID` (1000) masked out; all-ignore → 0.0 (no
    div-by-zero).
  - **`train_one_epoch(backbone, decoder, criterion, data_loader, optimizer,
    device, epoch, loss_scaler, args, max_norm=None, use_amp=True, print_freq=20)`**:
    mirrors `ConvNeXt-V2/engine_finetune.py`, trimmed. Per-iteration
    `adjust_learning_rate(optimizer, step/len(loader)+epoch, args)` (fractional
    epoch cosine; `args` needs `lr/min_lr/warmup_epochs/epochs`). AMP path:
    `torch.autocast("cuda", float16)` for the forward+loss, then
    `NativeScaler(loss, optimizer, clip_grad=max_norm, parameters=backbone+decoder
    params)`. Non-AMP fallback does manual backward/clip/step. `MetricLogger`
    tracks loss, pixel_acc, lr, grad_norm; raises on non-finite loss. Returns the
    `{meter: global_avg}` dict.
- **Design:** drives backbone+decoder directly (NOT `SegmentationModule`) so the
  model stays pure and the loss lives in the engine. Loader contract =
  `TrainSegDataset` `(image, label, seg_id, name)`; only image + seg_id used.
  Criterion is supplied by the caller (`nn.NLLLoss(ignore_index=1000)` to match the
  decoder's `log_softmax`).
- *Test (`tmp/test_seg_engine.py`, GPU):* atto backbone + `build_upernet`, a fixed
  8-image synthetic `FakeSeg` set (`(image,0,seg_id,name)`, some pixels set to
  1000) over 6 epochs with AMP + AdamW + grad-clip 1.0 → averaged loss decreases
  **monotonically 6.23 → 5.69**, pixel_acc creeps up; `seg_forward` out
  `(2,301,96,96)`; `pixel_accuracy` on an all-1000 target = 0.0 (no crash).
  device=cuda. PASS. *(Targets are random so it can't fully fit — the point is the
  gradient path + AMP + schedule + ignore handling all work end-to-end.)*

---

## 2026-06-02 — Phase 2: adapt UPerNet (`models/upernet/upernet.py`)

### Minimal adaptation of the CSAILVision port (port stays gold)
- **2.1 SyncBN→BN:** `BatchNorm2d = torch.nn.SyncBatchNorm` →
  `BatchNorm2d = nn.BatchNorm2d`. *Why:* SyncBN requires a DDP process group we
  never create; we train single-GPU, where SyncBN ≡ BN anyway. Deviation noted
  inline. Also `r'''`-prefixed the LaTeX (`\frac`/`\sqrt`) comment block to silence
  a `SyntaxWarning: invalid escape sequence` (confirmed clean under
  `-W error::SyntaxWarning`).
- **2.2 `build_upernet(dims, num_class=301, fpn_dim=512)`:** constructs
  `UPerNet(num_class=301, fc_dim=dims[-1], fpn_inplanes=tuple(dims), fpn_dim=512,
  use_softmax=False)` and calls `apply(ModelBuilder.weights_init)` (random init —
  honoring the no-pretrained-weights rule). `dims` is the backbone's per-stage
  channel tuple (atto = 40/80/160/320): `fc_dim` = top stage, `fpn_inplanes` = all
  four. `num_class=301` ⇒ seg ids 0..300, so `argmax` over channels = seg id.
  `use_softmax=False` ⇒ decoder emits `log_softmax` (pairs with NLLLoss).
- **2.3 seg path (NOT `SegmentationModule`):** that class bakes the loss in and
  calls `encoder(..., return_feature_maps=True)` which our backbone doesn't expose,
  so we drive the decoder directly:
  `feats = backbone.forward_features_seg(img)` (4-tuple, = UPerNet `conv_out`) →
  `decoder(feats)` (emits `log_softmax` at **stride-4**) → bilinear upsample to crop
  size → `nn.NLLLoss(ignore_index=1000)`. Keeps model pure (loss lives in the
  engine, per the port's own TODO).
- *Test (`tmp` one-liner):* atto backbone (`num_classes=300`) + `build_upernet`,
  input `(2,3,224,224)` → `forward_features_seg` shapes
  `[(2,40,56,56),(2,80,28,28),(2,160,14,14),(2,320,7,7)]` (strides 4/8/16/32) →
  decoder `(2,301,56,56)` → upsample `(2,301,224,224)`. Built a target with some
  `1000` ignore pixels; `NLLLoss(ignore_index=1000)` backward → finite grad on
  `backbone.downsample_layers[0][0].weight`. Import gate stays green. PASS.

---

## 2026-06-02 — Phase 1.2: logging as its own module (Phase 1 complete)

### `core/logging.py` — self-contained home (reworked from re-export)
- **`core/logging.py` now OWNS** `setup_logging`, `get_logger`, `LOG_FORMAT`,
  `LOG_DATEFMT`. They were **MOVED out of `core/utils.py`** (definitions + the
  two helpers physically relocated), and `utils.py` dropped its now-unused
  top-level `import logging` and `import sys`. *Why the move (not a re-export):*
  user wants logging to be a real, self-contained module, not an alias of utils —
  the initial thin-re-export version was rejected. No code outside utils/logging
  imported these symbols yet, so the move needed no caller updates.
- **Stdlib-shadow handling:** the module is literally named `logging`, but its
  top-level `import logging` resolves to the **stdlib** because the package path
  is `src.core.logging` and Python 3 uses absolute imports. Asserted
  `src.core.logging.logging is <stdlib logging>`. Never use a relative
  `from . import logging` (that would self-reference).
- **No circular import:** `logging.py` imports nothing from `utils`; `utils.py`
  imports nothing from `logging`. Gate confirms both load independently.
- *Test (`tmp` one-liner):* `logging.py` exposes the four symbols; `utils.py` no
  longer has `setup_logging`/`LOG_FORMAT`; stdlib `logging` unshadowed;
  `setup_logging()` + `get_logger("evaluate")` → logger name `cse164.evaluate`,
  emits a formatted line; **Phase-1 import gate** green —
  `src.core.{optim,utils,evaluate,dataset,logging}` + `src.models.upernet.upernet`
  all import on Windows (Minkowski not required by any). PASS.

### Milestone: Phase 1 (entire `core/`) complete
- `core/`: `utils` (infra), `optim` (AdamW+layer-decay), `dataset` (+pretrain
  loader), `evaluate` (kaggle glue), `logging` — all written, import-green, and
  unit-tested on the dense Windows path. Next up: **Phase 2** — adapt the pasted
  CSAILVision `upernet.py` (SyncBN→`nn.BatchNorm2d`, add `build_upernet`, verify a
  `(2,3,224,224)`→`(2,301,224,224)` forward + one NLLLoss backward).

---

## 2026-06-02 — Phase 1.5: evaluation & submission glue

### `core/evaluate.py` — thin wrapper over the reference scorer
- **Decision:** do NOT reimplement mIoU/boundary/rare — import the leaderboard's
  own `starter/kaggle_metric.py` and call `detailed_score`. Loaded by file path
  via `importlib` (lazy + `lru_cache`) under the name `kaggle_metric`, so `src`
  never imports the starter dir as a package and the file stays untouched. The
  starter guards its own `kaggle_metric_utilities` import, so it runs standalone.
- **`_normalize_rows(rows)`**: one row contract shared by both entry points.
  Accepts a dict keyed by image filename OR an iterable of row dicts; each row
  supplies `class_id` plus either a pre-encoded `segmentation_rle` string or a
  `seg_mask` array (encoded via `core.utils.encode_rle`). → list of
  `{image, class_id, segmentation_rle}`.
- **`write_submission(rows, path)`**: DataFrame → CSV (`image,class_id,
  segmentation_rle`). All-bg masks encode to `"0"` (never NaN). Returns the path.
- **`score_val(preds, data_root, split="val")`**: builds the submission frame +
  a cached `_solution_frame` (GT masks read at NATIVE resolution → `rgb_to_seg_id`
  → `encode_rle` with ignore-1000 **preserved**, plus `height`/`width`/`class_id`)
  and calls `km.detailed_score`. `_solution_frame` is `lru_cache`d (750 mask reads
  are wasteful to repeat each selection epoch). Returns the full metric dict.
- **`resize_mask_nearest(mask, (W,H))`**: PIL mode-"I" NEAREST resize to original
  resolution; no-op if already sized. Mandatory before scoring/RLE so no
  fractional/blended ids are invented. (Engines/tests will call this.)
- **Design note:** predictions must already be at original resolution and must
  never contain id 1000 (ignore is GT-only); `score_val` requires preds to cover
  every image in the split (the scorer rejects missing/extra rows).
- *Test (`tmp/test_evaluate.py`, all 750 val):* (1) `resize_mask_nearest` 2×2→(6,4)
  preserves exact ids; (2) **perfect** preds (GT with 1000→0) → `mean_iou`,
  `boundary_f_score`, `rare_class_miou`, `macro_accuracy` all = 1.0,
  `segmentation_score` 1.0, `automated_score` **0.9** (the off-Kaggle 10% report
  is excluded, so 0.9 is the ceiling — sanity-confirms the weighting); (3) all-bg
  preds → `mean_iou` 0, `seg_score` 0.0024, cls 1.0; (4) `write_submission`
  roundtrip → 750 rows, correct columns, no NaN RLE. PASS.

---

## 2026-06-02 — Phase 1.4: FCMAE pre-training loader

### CLAUDE.md reconciled with real `src/` layout (housekeeping)
- The repo tree had drifted: it still listed the old flat layout (`src/dataset.py`,
  `src/optim.py`, root-level `engine_convnext_*.py` / `train_*.py`, a legacy
  `src/utils.py`, and an "EMPTY" `upernet.py`). Reality: everything moved under
  subpackages — `src/core/{dataset,optim,evaluate,logging}.py`, `src/engines/`
  (`convnext_cls`, `convnext_upernet_seg`, `fcmae_pretrain`), `src/trainers/`
  (`train_{pretrain,cls,seg}`), `src/tests/` (`test_{cls,seg}`). Legacy
  `src/utils.py` is **deleted**. `upernet.py` is **252 lines** (CSAILVision port
  pasted, import-fix underway), not empty. Updated the tree, the models/engine/
  driver prose, and the module-gotchas (run cmd → `python -m src.trainers.train_seg`;
  `IGNORE_INDEX` now in `core/dataset.py`).

### `core/dataset.py` — `build_pretrain_loader` + `ImageOnlyDataset` (Phase 1.4)
- **`ImageOnlyDataset`**: generic wrapper returning `dataset[idx][0]`. The image
  is always element 0 of every split's `__getitem__`, so one wrapper strips
  labels/masks uniformly across TrainUnlabeled/TrainLabeled/TrainSeg. *Why:* FCMAE
  wants pixels only, and a uniform image-only item lets default collate stack a
  clean `(N,3,C,C)` batch — mixed-length tuples (unlabeled `(img,name)` vs
  labeled `(img,label,name)`) would break collate.
- **`build_pretrain_loader(data_root, batch_size=64, crop_size=224, ...)`**:
  `ConcatDataset` of the three TRAINING splits — `train_unlabeled` (50k) +
  `train_labeled` (7.5k) + `train_seg` (3k) ≈ 60.5k — each wrapped in
  `ImageOnlyDataset`. val/test deliberately NOT pooled (leakage). `crop_size=224`
  (`224%32==0`) so FCMAE's patch/mask grid divides evenly. `drop_last=True`,
  `pin_memory=True`, `shuffle=True`.
  - **Deviation from the 1.4 spec note:** spec said "reuse `TrainUnlabeledDataset`
    as-is" (which returns `(img, name)`); I wrap it in `ImageOnlyDataset` too so
    all three concat members return a bare tensor and collate works. The class
    itself is unmodified — only wrapped.
  - **Known minor cost:** `train_seg` goes through `SegTransform` (needs the mask
    for the joint geometric crop), so its mask PNG is read then discarded. 3k imgs
    → acceptable; revisit if FCMAE I/O-bound.
- *Test (`tmp/test_pretrain_loader.py`, PYTHONPATH=root):* `len==60500`; items at
  the unlabeled/labeled/seg range boundaries (0/50000/57500) are `(3,224,224)`
  float32 with `C%32==0`; one real batch collates to `(4,3,224,224)`. PASS.

---

## 2026-06-02 — Phase 1 core foundation (Sprint 4)

### Environment confirmation
- Verified `venv/Scripts/python.exe` (`torch 2.12.0+cu126`) runs **real GPU ops**
  on the 3090's CUDA-13.2 driver (backward-compatible) — no PyTorch reinstall
  needed. Dense cls/seg path is locally testable on Windows; MinkowskiEngine/FCMAE
  stays Docker-only. Run from project root.

### Import-blocker fixes (3 bugs)
- **`src/models/convnext/utils.py`**: wrapped top-level
  `from MinkowskiEngine import SparseTensor` in try/except → `SparseTensor=None`.
  *Why:* the dense `LayerNorm`/`GRN` don't need Minkowski; the unguarded import
  broke the dense backbone + UPerNet import on Windows. Minkowski* classes use
  `SparseTensor` only at call time (sparse path, Docker only).
- **`src/core/optim.py`**: wrapped the `timm.optim.*` imports
  (`Nadam`/`RAdam`/`AdamP`/…) in try/except → None fallback. *Why:* timm 1.0.x
  removed several symbols (`ImportError: cannot import name 'Nadam'`); only the
  unused exotic-optimizer branches reference them. AdamW + layer-decay path
  verified working (26 param groups, lr_scales 0.28→1.0).
- **`src/models/convnext/convnextv2.py:129`**: `forward_features_seg` referenced
  `getattr(self, f"norm{i}")` but `__init__` defines `self.seg_norms` (ModuleList)
  → fixed to `self.seg_norms[i](x)`. *Why:* real `AttributeError: no norm0`.
  Confirmed seg features at strides 4/8/16/32, channels = stage `dims`
  (atto: 40/80/160/320 ⇒ UPerNet `fpn_inplanes`).

### `src/core/utils.py` — single-GPU training infra port (no DDP)
- Added (each unit-tested): `SmoothedValue`, `MetricLogger`, `get_grad_norm_`,
  `NativeScaler` (AMP via `torch.amp.GradScaler("cuda")`),
  `adjust_learning_rate` (cosine + warmup, honours `lr_scale`),
  `save_checkpoint`/`load_checkpoint` (+`strict=False`), `remap_checkpoint_keys`
  (sparse→dense weight bridge). Kept the existing mask/RLE/seed/logging helpers.
- Tests passed: LR endpoints (warmup@0=0, peak@2=1e-3 / scaled group 5e-4,
  end@10=1e-6); MetricLogger global_avg; a real AMP step on GPU (finite grad-norm,
  weights changed); checkpoint roundtrip + epoch=7 + strict=False; remap shapes
  (downsample 8,3,4,4 / dwconv 8,1,7,7 / grn 1,1,1,8).
- **Deviation (recoverable):** the file had grown to 951 lines from a verbatim
  ~557-line paste of `ConvNeXt-V2/utils.py` (DDP/TensorBoard/Wandb, broken
  `inf`/`dist`/`SummaryWriter`) that shadowed and crashed the clean port.
  Truncated to the clean 395-line port. **User approved keeping the clean port.**
  Original recoverable from git history + the `ConvNeXt-V2/` clone.
