# DnCNN Image Denoising

PyTorch implementation of image denoising using DnCNN architecture with EDSR-style residual blocks.

## Project Overview

Built a deep learning-based denoising system achieving **11.4dB PSNR improvement** (15.17→26.57dB) on high-resolution images with Gaussian noise (σ=50).

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
- **Noise range**: Gaussian σ=5-60 (training), σ=50 (testing)
- **Loss**: L1 Loss
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 50
- **Batch size**: 64

## Results

| Metric | Value |
|--------|-------|
| Average Noisy PSNR | 15.17 dB |
| Average Output PSNR | 26.57 dB |
| Average Improvement | **11.40 dB** |

## File Structure

```
├── common.py          # ResBlock implementation
├── model.py           # DenoisingNet model
├── dataloader.py      # DIV2K dataset loader
├── train.py           # Training script
├── test.py            # Testing script
├── test_results/      # Test results (noise σ=25)
└── test_resultsss_noise50/  # Test results (noise σ=50)
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

## Sample Results
Noisy Image 
<img width="772" height="459" alt="image" src="https://github.com/user-attachments/assets/ffad7d73-c162-47aa-9c88-bfbd1a147ff9" /> 
denoised 
<img width="771" height="457" alt="image" src="https://github.com/user-attachments/assets/a667e039-9169-4b78-8586-a03e0bae7285" /> 
Ground Truth 
<img width="772" height="460" alt="image" src="https://github.com/user-attachments/assets/90a7b4d2-7049-4097-8dab-6c55d8755556" /> 



See `test_resultsss_noise50/` for full test results on 50 images.

## Author

Developed as part of computer vision research internship at Gwangju Institute of Science and Technology (Nov 2025 - Feb 2026).
