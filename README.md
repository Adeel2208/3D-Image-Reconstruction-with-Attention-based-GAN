# 3D Reconstruction GAN with Hybrid Vision Transformer

This project implements a Generative Adversarial Network (GAN) for 3D image reconstruction, featuring a Hybrid Vision Transformer U-Net generator with window-based self-attention, frequency enhancement, and multi-scale feature aggregation. The model is trained on paired RGB image-mask datasets, using advanced loss functions (perceptual, edge-aware, multi-scale SSIM, and frequency-based) and a PatchGAN discriminator. It includes data augmentation, checkpointing, and comprehensive evaluation with metrics like F1 score, precision, recall, accuracy, SSIM, and PSNR.

## Features
- **Generator**: Hybrid Vision Transformer U-Net with multi-scale feature aggregation and frequency enhancement.
- **Discriminator**: PatchGAN with spectral normalization for stable training.
- **Loss Functions**: Combines adversarial, L1, perceptual, edge-aware, multi-scale SSIM, and frequency losses.
- **Data Augmentation**: Random flips, rotations, and color jitter for robust training.
- **Evaluation**: Tracks and visualizes F1, precision, recall, accuracy, SSIM, and PSNR, with metrics saved to CSV.
- **Checkpointing**: Saves model states periodically and for best SSIM score.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/3d_reconstruction_gan.git
   cd 3d_reconstruction_gan
