# CSE 164 Final Project

## Usage
### Docker
The most straightforward way to run this project is in Docker.

#### Interactive Notebook
Train, test, debug, examine features/demonstrations, and evaluate in an interactive notebook environment.

**To start jupyter lab:**
```
docker compose up notebook -d
docker compose run notebook
```
Then go to `http://127.0.0.1:8888/lab`

**When complete** end the container with
```
docker compose --profile notebook down
```

#### CLI
**TODO**

## Dependencies
**TODO**

## Dataset
Dataset is taken directly from: [Kaggle - CSE164 Final Project 2026](https://www.kaggle.com/competitions/cse-164-final-project-2026/leaderboard)

## Acknowledgment
This repository is built using the [ConvNeXt V2 repository](https://github.com/facebookresearch/ConvNeXt-V2/).

And uses the [PyTorch] and [MinkowskiEngine] libraries.

This repository was used as course work for UC Santa Cruz course CSE 164: Computer Vision (Spring '26 Prof. Cihang Xie).