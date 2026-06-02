# ConvNeXt-V2 Source Code — Breakdown & Design Notes

Notes on the **reference implementation** (facebookresearch/ConvNeXt-V2, cloned
locally at `ConvNeXt-V2/` — observation only, NOT part of our tree) and what we
take from it for our own `src/` layout.

The repo is the companion code for the FCMAE / ConvNeXt V2 paper (see
`CONVNEXT.md` for the paper notes). It does two things end-to-end:
**(1) FCMAE self-supervised pre-training** and **(2) supervised fine-tuning** of
the resulting backbone for ImageNet classification.

---

## 1. The big idea: three layers + shared infra

The repo is flat (everything at the root), but the responsibilities split into
three clean layers, with a task **never** mixing them:

```text
              models/            engine_*.py              main_*.py
          ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐
          │ architecture │ → │ one-epoch loop   │ → │ driver / entrypoint  │
          │ ONLY         │   │ (train / eval)   │   │ (wires everything)   │
          └──────────────┘   └──────────────────┘   └──────────────────────┘
                              shared infra: datasets.py, optim_factory.py, utils.py
```

- **`models/`** — pure `nn.Module` definitions. No data, no optimizer, no loop.
  - `convnextv2.py` — *dense* backbone (`ConvNeXtV2`, `Block`) + size factories
    (`convnextv2_atto` … `_huge`). Used for fine-tuning + inference.
  - `convnextv2_sparse.py` — *sparse* backbone on **MinkowskiEngine** sparse
    tensors, so masked-out patches cost zero compute. Used only as the FCMAE
    encoder during pre-training.
  - `fcmae.py` — the `FCMAE` masked-autoencoder wrapper: sparse encoder + a thin
    dense decoder + the random-masking + patchify/unpatchify + reconstruction
    loss. Its own size factories mirror the backbone's.
  - `utils.py` — `LayerNorm` (channels-first/last), `GRN`, and the Minkowski
    sparse equivalents (`MinkowskiLayerNorm`, `MinkowskiGRN`, `MinkowskiDropPath`).
- **`engine_pretrain.py` / `engine_finetune.py`** — exactly one `train_one_epoch`
  each (and `evaluate` in the finetune engine). Stateless w.r.t. orchestration:
  take `(model, loader, optimizer, device, epoch, loss_scaler, args)`, run one
  pass, return a stats dict. **This is the reusable core.**
- **`main_pretrain.py` / `main_finetune.py`** — the drivers. Arg parsing, build
  dataset/model/optimizer/scheduler, distributed setup, **own the epoch loop**,
  checkpoint save/resume, logging. One driver **per task** because the two tasks
  genuinely differ.
- **Shared infra**: `datasets.py` (build dataset + transforms), `optim_factory.py`
  (optimizer + layer-wise LR decay), `utils.py` (metrics, checkpointing, LR
  schedule, distributed, AMP scaler). `submitit_*.py` are SLURM launchers — not
  relevant to us.

---

## 2. Pre-training flow (FCMAE)

`main_pretrain.py` → `engine_pretrain.train_one_epoch`:

1. Build a plain `ImageFolder` of unlabeled images (just `RandomResizedCrop` +
   `flip` + normalize — MAE wants minimal augmentation).
2. Build the model via `fcmae.__dict__[args.model](mask_ratio, decoder_depth,
   decoder_embed_dim, norm_pix_loss)`.
3. Optimizer: `AdamW`, **no layer decay** here, betas `(0.9, 0.95)`. LR is set by
   the **linear scaling rule** `lr = blr * eff_batch_size / 256`.
4. Per-epoch loop: `loss, _, _ = model(samples, labels, mask_ratio)`. The model
   returns the reconstruction loss directly (MSE on **masked patches only**).
   - Per-**iteration** cosine LR via `adjust_learning_rate`.
   - Gradient accumulation via `update_freq`.
   - `torch.cuda.empty_cache()` every step — a quirk needed to keep the
     MinkowskiEngine sparse network from leaking GPU memory.
5. Checkpoint every `save_ckpt_freq` epochs.

Key hyperparameters (paper recipe): mask ratio **0.6**, AdamW, blr 1.5e-4, wd
0.05, cosine schedule, 40 warmup epochs, 800–1600 total epochs, batch 4096.

---

## 3. Fine-tuning flow (and the sparse→dense bridge)

`main_finetune.py` → `engine_finetune.train_one_epoch` / `evaluate`.

The interesting part is **loading the pre-trained backbone into the dense model**
(`main_finetune.py` ~L257–282). This is the crux of "use our own SSL weights":

1. Build the **dense** `convnextv2` model with a fresh classification head.
2. `torch.load(args.finetune)` the FCMAE checkpoint, then **strip everything that
   isn't the encoder**: keys containing `decoder`, `mask_token`, `proj`, `pred`
   are deleted, and `head.*` is dropped if shapes mismatch.
3. **`remap_checkpoint_keys`** (`utils.py` ~L545) rewrites the *sparse* encoder
   weights into *dense* `nn.Conv2d` layout: strips the `encoder.` prefix, turns
   Minkowski `*.kernel` tensors into `*.weight` and **reshapes/permutes** them
   into standard conv kernels (3-D for normal conv, 2-D for depthwise), and
   reshapes GRN affine params. Sparse and dense store conv weights differently,
   so this remap is mandatory — you cannot `load_state_dict` raw.
4. `utils.load_state_dict` loads with `strict=False`, manually re-inits the head.

Fine-tuning then adds the heavy supervised-training machinery the pretrain loop
doesn't have:

- **Layer-wise LR decay** (`optim_factory.LayerDecayValueAssigner`): earlier
  layers get a smaller LR (`lr_scale`), so the pre-trained backbone is nudged
  gently while the new head learns fast. Two modes: `single` (per-block) and
  `group` (3 blocks share an id — used for Base/Large). *Our CONVNEXT.md notes
  flag this as mattering for small models — keep it.*
- **AMP** (`use_amp` + `NativeScalerWithGradNormCount`) with grad clipping.
- **timm Mixup/CutMix** + **label smoothing** (→ `SoftTargetCrossEntropy` or
  `LabelSmoothingCrossEntropy`), and **EMA of weights** (`ModelEma`).
- `evaluate` is just top-1/top-5 accuracy + CE loss.

---

## 4. Shared infra worth reusing

- **`optim_factory.py`** — `get_parameter_groups` splits params into decay /
  no-decay (1-D params, biases, `.gamma`, `.beta` get **no weight decay**) and
  tags each with a `layer_id` + `lr_scale`. `create_optimizer` is a big
  if/else over optimizer names — we only need the AdamW branch.
- **`utils.py`**:
  - `SmoothedValue` / `MetricLogger` — windowed running averages + a nice
    `log_every` iterator that prints ETA/throughput.
  - `NativeScalerWithGradNormCount` — AMP loss scaler that also returns grad norm.
  - `cosine_scheduler` (value array) and `adjust_learning_rate` (per-step,
    applies `lr_scale` for layer decay) — **half-cycle cosine after linear
    warmup**. We want the latter.
  - `save_model` / `auto_load_model` — checkpoint {model, optimizer, scaler,
    epoch, (ema)}; auto-resume from `output_dir`.
  - `init_distributed_mode`, `all_reduce_mean`, etc. — **DDP; drop for us.**

---

## 5. What WE keep / adapt / drop  (design choices)

We adopt the **models / engine / driver** separation, but our task is different
in three ways: (a) segmentation is the primary task and the reference has **no
segmentation at all**, (b) we run on a single GPU, (c) we have three training
stages, not two.

| Reference | Our plan |
|---|---|
| `models/` (dense, sparse, fcmae, utils) | ✅ already ported to `src/models/convnext/` |
| `engine_pretrain.py` | → `src/engine_convnext_pretrain.py` (FCMAE loop) |
| `engine_finetune.py` (cls only) | adapt → `src/engine_convnext_finetune.py` for **both** cls and seg one-epoch loops |
| `main_pretrain.py` / `main_finetune.py` | split into **3 drivers**: `train_pretrain.py`, `train_cls.py`, `train_seg.py` (run via `python -m src.<driver>`) |
| `optim_factory.py` | port the **layer-decay + param-group** logic → `src/optim.py`; keep AdamW only |
| `utils.py` infra | port `MetricLogger`, `NativeScaler`, cosine LR, save/load into `src/core/`; **drop all DDP/submitit** |
| `datasets.py` (ImageFolder/CIFAR) | already have richer `src/dataset.py` (5 splits, joint img+mask seg transform) |
| `evaluate` = top-1 acc | rewrite for **our metric**: foreground mIoU + boundary-F + rare-class, plus RLE/submission writing |
| Mixup / CutMix / EMA / AMP | optional; AMP + light aug likely worth it, mixup is awkward with seg masks |
| `remap_checkpoint_keys` (sparse→dense) | **MUST port** — this is how our FCMAE-pretrained sparse encoder loads into the dense backbone for cls/seg fine-tuning |
| **(none)** | **NEW: UPerNet** seg decoder on top of the dense backbone's multi-scale features |

**Biggest gap vs. the reference:** there is no segmentation head and no seg
metric in the source — that's ours to build (UPerNet + the Kaggle-style scorer).
The reference is the *pre-train + classification* template; the segmentation half
is original work layered on the same backbone + the same engine/driver pattern.

> The sparse encoder only exists to make FCMAE cheap. Once pre-training is done,
> everything downstream uses the **dense** backbone — the sparse weights are
> remapped in once and never touched again.
