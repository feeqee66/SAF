# SAF – AI-Based Restoration of Degraded Images

## Overview

This repository contains an offline AI-based image restoration system developed for the KLA problem statement.

The system accepts degraded grayscale images in `.npy` format and produces restored images at 2× the input resolution.

## Model

The final submitted model is a lightweight residual CNN trained for image restoration and 2× super-resolution.

### Architecture

Input grayscale image
→ Initial convolution
→ 8 residual blocks
→ Feature refinement
→ PixelShuffle ×2 upsampling
→ Output convolution
→ Restored grayscale image

### Model specifications

- Input: Grayscale `.npy` array
- Expected input resolution: `128 × 128`
- Output resolution: `256 × 256`
- Scale factor: `×2`
- Parameters: 776,705
- Model checkpoint: `models/cnn_l1_best.pt`
- Validation PSNR: 28.0843 dB
- Validation SSIM: 0.7550

## Requirements

- Python
- PyTorch 2.13.0
- NumPy 2.5.2
- NVIDIA GPU is recommended for faster inference

The system does not require internet access, API keys, external model downloads, or user interaction during inference.

## Installation

Install the dependencies using:

```bash
pip install -r requirements.txt