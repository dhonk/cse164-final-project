# CSE 164 Final Project

## Usage
First, must ensure the following project structure:

```
cse164-final-project/
├── src/
├── data/
├── logs/
├── outputs/
├── checkpoints/
│   └── final/
└── ...
```

Start by installing dependencies:
```
pip install -r requirements.txt
```

To run FCMAE pretraining:
```
python -m src.pretrain
```

To run classification finetuning:
```
python -m src.clstrain
```

To run segementation finetuning:
```
python -m src.segtrain
```

To generate Kaggle results:
  First, move the generated checkpoint file from classification training, and segmentation training (called `checkpoint-clsfinetune-{epoch #}.pth` and `checkpoint-segfinetune-{epoch #}.pth` into `checkpoints/final/`, like so:
```
├── checkpoints/
│   └── final/
│       ├── checkpoint-clsfinetune-{epoch #}.pth
│       └── checkpoint-segfinetune-{epoch #}.pth
```

Then run 
```
python -m src.finalres
```

## Dependencies
python 3.12
CUDA 12.6
Check requirements.txt for all other dependencies

## Dataset
Dataset is taken directly from: [Kaggle - CSE164 Final Project 2026](https://www.kaggle.com/competitions/cse-164-final-project-2026/leaderboard)

## Acknowledgment
This repository is built using the [ConvNeXt V2 repository](https://github.com/facebookresearch/ConvNeXt-V2/).

This repository was used as course work for UC Santa Cruz course CSE 164: Computer Vision (Spring '26 Prof. Cihang Xie).
