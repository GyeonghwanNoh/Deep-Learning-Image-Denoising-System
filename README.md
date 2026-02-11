# DnCNN Image Denoising

PyTorch implementation of image denoising using DnCNN architecture with EDSR-style residual blocks.

## Project Overview

Built a deep learning-based denoising system using DnCNN with EDSR-style residual blocks. See the **Results** and **Ablation** tables for quantitative performance.

## Architecture

- **Model**: DnCNN with EDSR residual blocks
- **Layers**: 16 ResBlocks with 0.1 residual scaling
- **Input**: 4 channels (RGB + noise level map)
- **Output**: 3 channels (denoised RGB)

## Dataset

- **DIV2K**: 800 high-resolution images (2040×1356)
- **Training**: Images 1-750
- **Testing**: Images 751-800

## Training Details

- **Patch size**: 64×64 random crops
- **Noise range**: Gaussian σ=5-60 (training), σ=10-70 (testing)
- **Loss**: MSELoss
- **Optimizer**: AdamW (lr=1e-4)
- **Epochs**: 500
- **Batch size**: 8

## Results (Ours, 50K iterations, noise map)

| Noise σ | PSNR (dB) |
|--------:|-----------|
| 10 | 37.11 |
| 20 | 33.80 |
| 30 | 31.90 |
| 40 | 30.56 |
| 50 | 29.54 |
| 60 | 28.70 |
| 70 | 28.00 |

## Experiments

### Exp1 vs Exp2

| Exp | Batch | LR | Res scale | Epoch | Iter | σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|-----|------:|----|-----------|------:|-----:|------|------|------|------|------|------|------|
| Exp1 | 16 | 1e-4 | 0.1 | 120 | 6000 | 35.68 | 32.23 | 29.86 | 27.93 | 26.24 | 24.72 | 23.33 |
| Exp2 | 8 | 1e-4 | 0.1 | 300 | 30000 | 36.36 | 32.71 | 30.25 | 28.26 | 26.52 | 24.95 | 23.51 |

### LR Comparison (Batch=8, Res scale=0.1, Epoch=100, Iter=10000)

| LR | σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|----|------|------|------|------|------|------|------|
| 5e-5 | 35.38 | 32.01 | 29.68 | 27.78 | 26.13 | 24.64 | 23.28 |
| 1e-4 | 35.89 | 32.38 | 29.98 | 28.03 | 26.34 | 24.81 | 23.41 |
| 5e-4 | 36.09 | 32.53 | 30.12 | 28.16 | 26.46 | 24.93 | 23.53 |
| 2e-4 | 36.18 | 32.59 | 30.15 | 28.17 | 26.44 | 24.89 | 23.47 |

### Batch Size Comparison (LR=2e-4, Res scale=0.1, Iter=10000)

| Batch | Epoch | σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|------:|------:|------|------|------|------|------|------|------|
| 12 | 149 | 36.14 | 32.55 | 30.12 | 28.14 | 26.42 | 24.87 | 23.45 |
| 16 | 200 | 36.16 | 32.56 | 30.12 | 28.14 | 26.43 | 24.88 | 23.46 |
| 20 | 250 | 36.22 | 32.59 | 30.14 | 28.16 | 26.43 | 24.86 | 23.44 |
| 24 | 303 | 36.19 | 32.59 | 30.14 | 28.16 | 26.44 | 24.88 | 23.46 |

### 50K (LR=2e-4, Batch=8, Res scale=0.1, Epoch=500)

| σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|------|------|------|------|------|------|------|
| 37.11 | 33.80 | 31.90 | 30.56 | 29.54 | 28.70 | 28.00 |

## Pretrained DnCNN vs Ours

| Model | σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|-------|------|------|------|------|------|------|------|
| Pretrained (Colorblind.pth) | 36.34 | 32.94 | 30.57 | 28.64 | 26.95 | 25.43 | 24.02 |
| Ours (50K) | 37.11 | 33.80 | 31.90 | 30.56 | 29.54 | 28.70 | 28.00 |

## Noise Map Ablation (with/without)

| Setting | σ=10 | σ=20 | σ=30 | σ=40 | σ=50 | σ=60 | σ=70 |
|---------|------|------|------|------|------|------|------|
| With noise map | 37.11 | 33.80 | 31.90 | 30.56 | 29.54 | 28.70 | 28.00 |
| Without noise map | 37.01 | 33.73 | 31.84 | 30.51 | 29.48 | 28.64 | 27.92 |

## File Structure

```
├── common.py          # ResBlock implementation
├── model.py           # DenoisingNet model
├── dataloader.py      # DIV2K dataset loader
├── train.py           # Training script
├── test.py            # Testing script
├── test_results/      # Test results (images/figures/logs)
│   └── patch_figures/ # Patch comparison figures
```

## Usage

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

## Requirements

```
torch
torchvision
numpy
Pillow
matplotlib
```

## Noise Map Effect

Add the noise map effect figure here:
- test_results/noisemap_effect.png

## Patch Comparison

Add patch comparison figures here:
- test_results/patch_comparison_sigma25.png
- test_results/patch_comparison_sigma50.png
- test_results/patch_figures/ (multiple patches)

## Perceptual Comparison

Replace with your comparison images (noisy / denoised / ground truth).

## visualize_patches.py Usage

```bash
python visualize_patches.py --input_dir test_results/patch_figures --out_dir test_results
```

> Adjust arguments to match your actual script options.

## Author

Developed as part of computer vision research internship at Gwangju Institute of Science and Technology (Nov 2025 - Feb 2026).
