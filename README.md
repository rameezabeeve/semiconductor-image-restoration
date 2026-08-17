# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Project Overview

This project uses deep learning to restore degraded semiconductor inspection images.

The model takes a degraded 128x128 NoisyLR image as input and generates a restored 256x256 image.

The goal is to improve the quality of degraded semiconductor inspection images and assist downstream inspection and defect analysis.

## Dataset

The dataset contains:

- 3200 training NoisyLR images
- 3200 Ground Truth (GT) images
- 400 test NoisyLR images

All images are stored in NumPy `.npy` format.

### Training Dataset Structure

After extracting the training dataset, arrange it as:

data/
└── train/
    ├── NoisyLR/
    │   ├── 000001.npy
    │   ├── 000002.npy
    │   └── ...
    └── GT/
        ├── 000001.npy
        ├── 000002.npy
        └── ...

The corresponding NoisyLR and GT images use matching filenames.

## Model

The project uses a deep learning image restoration model implemented in PyTorch.

Input:
- Size: 128x128
- Format: NumPy `.npy`

Output:
- Size: 256x256
- Format: NumPy `.npy`

## Repository Structure

```text
semiconductor-image-restoration/
│
├── README.md
├── requirements.txt
├── train.py
├── evaluation.py
│
├── model/
│   └── model.py
│
├── weights/
│   └── best_model.pth
│
├── test_images/
│   └── test `.npy` files
│
└── restored_outputs/
    └── restored `.npy` files
