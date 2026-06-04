# Time to Cook
## General Ideas
**Committing to ConvNeXt backbone!!** 

Will be taking the shared encoder approach for now for the following reasons:
- Limited time to train
- Small dataset, pretraining and squeezing as much as I can from data crucial

In Class, MIM was described as the "best" technique for SSL.

## Pre-Training Ideas
**Some training pipeline thoughts!** FCMAE on 50k unlabeled + 7.5k classification labeled WITHOUT labels to treat it as unlabeled + 3k semantic segmentation masked WITHOUT masks to treat it as unlabeled
    - =60.5k images...?

**From Gemini 3.1 Pro:**
> **Approach 1: Pretraining Only on the 50k Unlabeled Samples:** The model learns entirely from the 50k pool and is later fine-tuned on the separate 7.5k (classification) and 3k (segmentation) datasets.
>
> **Domain Shift (Distribution Mismatch):** This is the biggest risk. If your 50k unlabeled images were scraped from a different source or collected under different conditions than your labeled data, the MAE will learn a representation space that doesn't align with your downstream tasks. When you move to fine-tuning, the model will struggle to adapt.
>
> **Underutilization of Data:** MAEs scale exceptionally well with data volume. By completely walling off 10.5k high-quality images during the pretraining phase, you are leaving roughly 17% of your visual diversity on the table.
>
> **Overfitting During Fine-Tuning:** With 300 target classes, 7.5k labeled images leaves you with an average of only 25 images per class (and only 10 per class for segmentation). If the pretrained model hasn't learned robust, universal features from the 50k set, it will rapidly overfit to these tiny labeled subsets during fine-tuning.
>
> **Approach 2: Pretraining on All 60.5k Samples (Ignoring Labels)** You pool everything together for the MAE pretext task, maximizing your pretraining data, and then fine-tune on the labeled subsets.
>
> **Memorization vs. Generalization:** If the model sees the exact same images in pretraining that it will later be fine-tuned on, it risks memorizing specific image patches rather than learning generalized, semantic features. When fine-tuned, the model is essentially acting on "confirmation bias"—relying on memorized textures of those specific images rather than learning the underlying features of the 300 classes.
> 
> **Class Imbalance Amplification:** If the 10.5k labeled images are perfectly balanced across your 300 classes, but the 50k unlabeled images are heavily skewed towards just 10 classes, combining them means your pretraining distribution is heavily skewed. The MAE will become excellent at reconstructing the majority classes while remaining blind to the long-tail minority classes.
>
> **The Verdict: In modern Self-Supervised Learning (SSL), Approach 2 is generally the standard and most effective strategy, but with one absolute condition: You must rigorously separate your validation and test splits first. You should pool the 50k unlabeled images with only the training splits of your 7.5k and 3k datasets. This maximizes your data volume for the MAE to learn good patch reconstructions without leaking your evaluation data into the model's memory.**

앗싸, FCMAE 쓰자. 

## Fine-Tuning Ideas
Directly running end-to-end task on both respective architectures is likely the best bet. Use the 7.5k labeled to train up the classifier model, and use the 3k to train up the semantic segmentation model.

Risk is of overfit/inability to gneralize with such few samples, but later this can be addressed with pesudolabeling on the untrained dataset.

## Model Architecture Ideas
### ConvNeXt Backbone
ConvNeXt V2 will be the backbone of choice, currently SOTA, and also CNNs generally do better with data starvation than ViT. Swin at this size and with this few samples will be very hard.

### UPerNet Decoder for Semantic Segmentation
UPerNet was the decoder architecture of choice for ConvNeXt V2, and was used when benchmarking the semantic segmentation task performance (~50 mIoU on Ade20k)

## Implementation Ideas
### Project Structure

```
cse164-final-project/
│   **the pipeline itself**
├── src/
│   │
│   │   **key deps, used throughout**
│   ├── core/
│   │   ├── dataset.py  # process the images into usable datasets - TODO: add a loader for the big unlabeled set
│   │   ├── evaluate.py # all tools and calculations to find mIoU, macro-average accuracy, loss, etc
│   │   ├── optim.py    # hold optimizers (like implementation in ConvNeXt-V2) - TODO
│   │   ├── logging.py  # this has all logging to stdout and also to (project root)/logs/ - TODO
│   │   └── utils.py    # tools used throughout the project - metadata parsing, seg_id to class_id, etc
│   │                   # add utils from ConvNeXt-V2/utils.py to here - WIP
│   │ 
│   │   **the models, implemented in PyTorch**
│   ├── models/
│   │   │
│   │   │   **convnext v2 - backbone** - DONE...?
│   │   ├── convnext/
│   │   │   ├── convnextv2.py         # dense convnextv2, also convnextv2 pipeline for classification 
│   │   │   ├── convnextv2_sparse.py  # sparse convnextv2, for fcmae
│   │   │   ├── fcmae.py              # pieces together convnextv2 pipeline for fcmae pre training
│   │   │   ├── utils.py              # holds convnext utilities
│   │   │   └── LICENSE 
│   │   │
│   │   │   **upernet - decoder for segmentation** - WIP
│   │   ├── upernet/
│   │   │   ├── upernet.py            # upernet decoder, also full pipeline for segmentation - **TODO**
│   │   │   └── LICENSE 
│   │   │
│   │   │   **resnet-18 and unet - toy samples to check data input/output** - IGNORE
│   │   ├── resnet/
│   │   │   └── resnet.py             # resnet (not using)
│   │   └── unet/
│   │       └── unet.py               # unet (not using)
│   │
│   │   **defines epoch behavior in training**  - TODO
│   ├── engines/       
│   │   ├── fcmae_pretrain.py        # "train one epoch" for fcmae pretraining
│   │   ├── convnext_cls.py          # "train one epoch" for classification by convnext  
│   │   └── convnext_upernet_seg.py  # "train one epoch" for segmentation by convnext upernet pipeline 
│   │
│   │   **training harnesses - run # epochs, save checkpoints, set hyperparameters, load weights, etc** - TODO   
│   ├── trainers/
│   │   ├── train_pretrain.py  # training harness for pretraining - hyperparameters in, learned weights out
│   │   ├── train_cls.py       # training harness for classification - longer epochs for fine tuning
│   │   └── train_seg.py       # training harness for segmentation
│   │
│   │   **tests - create predictions** - TODO
│   └── tests/
│       ├── test_cls.py  # create predictions for classification
│       └── test_seg.py  # create predictions for segmentation
│
│   **high-level users**
├── notebook.ipynb  # interactive sanity-checks / scratchpad - IGNORE
├── pipeline.py         # cli interface for train / test / etc - TODO
│
│   **utilities**
├── checkpoints/  # store trained model weights - ideally timestamped/indexed
│
│   **test suite** - 
├── tests/  # copy project structure of src, write unit tests for each function.
│           # try out 1 epoch, check weight updates
│           # key metrics to look out for are time, FLOPs
│
│   **course-provided code, has some useful implementations to copy from**
├── starter/                      # course-provided, DO NOT EDIT
│   ├── make_sample_submission_csv.py
│   ├── validate_submission_csv.py   # ALWAYS run before submitting
│   └── kaggle_metric.py             # reference scorer (decode_rle_to_mask etc.)
│
│   **data**
├── data/
│   ├── train_labeled/images/        # 7,500 imgs, IMAGE-LEVEL labels only
│   ├── train_seg/{images,masks}/    # 3,000 imgs WITH segmentation masks
│   ├── train_unlabeled/images/      # 50,000 imgs, NO labels (+ a few distractors)
│   ├── val/{images,masks}/, classification.json   # 750 public, fully labeled
│   ├── test/images/                 # 3,000 hidden test images
│   └── metadata/{class_map,train_labeled,train_seg}.json
│
│   **claude**
├── CLAUDE.md                     # this file — read before coding
│
│   **docker/environment tools**
├── compose.yaml                  # `notebook` service (JupyterLab on :8888)
├── docker/
│   ├── Dockerfile                # CUDA 12.6 + PyTorch 2.12 + MinkowskiEngine build
│   └── entrypoint.sh             # task dispatch (default: notebook)
├── requirements.txt              # torch>=2.12 (cu126), torchvision, timm, tf, jupyter…
├── venv/              
│
│   **key insights & project details**
├── cooking/                      # research notes (Obsidian-style), NOT code, has some important details
│   ├── references/               # reference source code! Any libraries copied from will live in here 
│   │   └── ...                   # frequently reference and use as a guide
│   ├── PROJECT_SPEC.md           # the official competition spec
│   ├── IDEATION.md               # committed approach + live roadmap
│   ├── CONVNEXT.md               # ConvNeXt V1/V2 + FCMAE PAPER notes
│   ├── CONVNEXTV2_CODEBASE.md    # reference SOURCE-CODE breakdown + our design choices
│   ├── CONVERSATIONSNOTES.md     # storage for notes and insight picked up in presentations, meetings, etc
│   ├── CHANGELOG.md              # storage for any AI code assistant changes
│   ├── LECTURES.md               # FCN / segmentation lecture notes
│   └── NOTES.md
│
|   **logging: track all stdout and errors** - TODO
├── logs/  # key metrics to log: time, loss, error, metrics, FLOPs
│
|   **outputs: model predictions go here** - TODO
├── outputs/  # store submission .csv files here, timestamp in names
│
|   **info** - IGNORE
└── README.md                     # Docker usage (CLI/deps sections TODO)

```
    
## Starting over - I can't work with sparse convolution libraries

1. Refactor core/ first
2. Rewrite convnextv2 to use binary masking technique (slower but oh well what can I do)