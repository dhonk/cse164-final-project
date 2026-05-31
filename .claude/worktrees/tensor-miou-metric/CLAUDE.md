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

## Workflow commands

```bash
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