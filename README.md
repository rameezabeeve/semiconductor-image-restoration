# semiconductor-image-restoration
AI-Based Restoration of Degraded Images for Semiconductor Inspection
# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Project Overview

This project uses deep learning to restore degraded semiconductor inspection images.

The model takes a 128x128 degraded image as input and produces a 256x256 restored image.

## Dataset

The dataset contains:

- 3200 training NoisyLR images
- 3200 Ground Truth images
- 400 test NoisyLR images

Input format: `.npy`

## Model

A deep learning based image restoration and super-resolution model is used.

## Repository Structure

- `train.py` - Training script
- `evaluate.py` - Inference/evaluation script
- `model.py` - Model architecture
- `requirements.txt` - Required Python packages


## Status

Training and evaluation pipeline under development.
