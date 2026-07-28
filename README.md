# CHSG: A Center-Aware Axial Hierarchical Sparse Graph Network for Hyperspectral Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository provides the official PyTorch implementation of **CHSG**, a center-aware axial hierarchical sparse graph network for hyperspectral image classification. The main training entry is `main_v.py`, and the model implementation is in `Model/CHSG.py`.

## Project structure

```text
.
├── main_v.py          # training and evaluation entry
├── Config.py          # command-line arguments
├── DataLoad.py        # PyTorch dataset and preprocessing
├── HyperTools.py      # dataset loading, splitting, training, and inference
├── DrawHyper.py       # classification maps and class metadata
├── Model/
│   └── CHSG.py        # CHSG model
├── Utils/
│   ├── evaluation.py  # accuracy metrics
│   └── scheduler.py   # optimizer and learning-rate scheduler
├── requirements.txt
└── run.sh
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone git@github.com:LanLanZhao/CHSG.git
cd CHSG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install a CUDA-compatible PyTorch build separately when GPU training is required.

## Dataset layout

Datasets are not included in this repository. Put the image and ground-truth `.mat` files in one directory:

```text
dataset/
├── Houston13.mat
├── Houston13_gt.mat
└── ...
```

The supported file names and MATLAB keys are defined in `HyperTools.py`. Use `--data-root` if the datasets are stored elsewhere.

## Training

For a single quick experiment:

```bash
python main_v.py \
  --data-root ./dataset \
  --dataset houston \
  --Experiment_num 1 \
  --epochs 250
```

For the default ten-run evaluation:

```bash
bash run.sh
```

Generated reports, figures, and predictions are written to `results/` by default and are excluded from version control.

Useful arguments:

- `--split_type fixed --train_num 30`: use a fixed number of training samples per class.
- `--split_type ratio --train_ratio 0.05`: use a training ratio.
- `--components 30`: enable PCA with 30 components; `0` disables PCA.
- `--boundary-mode cyclic`: choose `cyclic`, `zero`, `reflect`, or `replicate`.
- `--no-data_aug`: disable training-time augmentation.

Run `python main_v.py --help` for the complete option list.

## Citation

Citation information will be added after the associated paper is publicly available.

## License

This project is released under the [MIT License](LICENSE).
