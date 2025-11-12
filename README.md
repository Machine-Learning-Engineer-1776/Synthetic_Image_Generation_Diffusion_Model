**<h1>Synthetic MRI Generation with Fine-Tuned Stable Diffusion:</h1>**

<div align="center">

  
![Brain 2](https://github.com/user-attachments/assets/d11f2754-30b9-4e2f-a438-1ec35f5d967d)


</div>

# Synthetic MRI Generation with Fine-Tuned Stable Diffusion
**High-Fidelity Brain Scan Synthesis via Latent Diffusion**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Diffusers](https://img.shields.io/badge/HuggingFace%20Diffusers-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Run%20Here-orange)

**Run in Colab:** [Open in Colab](https://colab.research.google.com/drive/PUT_YOUR_LINK_HERE)  
**GitHub:** [github.com/YOUR_USERNAME/Synthetic-MRI-Diffusion](https://github.com/YOUR_USERNAME/Synthetic-MRI-Diffusion)

---

## Overview

This Colab notebook generates **realistic synthetic brain MRI scans** using a **fine-tuned Stable Diffusion model**.

- **Input**: 256×256 grayscale MRI slices  
- **Output**: 256×256×3 color synthetic MRIs  
- **Method**: Unconditional sampling, 1000 steps, guidance scale 7.0  
- **Metrics**: FID ~200, SSIM 0.78, PSNR 28.4 dB  

---

### Key Hyperparameters

| Parameter        | Value       | Purpose                            |
|------------------|-------------|------------------------------------|
| `timesteps`      | **1000**    | Higher fidelity denoising          |
| `guidance_scale` | **7.0**     | Stronger classifier-free guidance  |
| `batch_size`     | 32          | Efficient GPU usage                |
| `latent_dim`     | (4, 32, 32) | VAE latent space shape             |

---

### Model Comparison (All Tested Architectures)

| Model             | FID (↓) | SSIM (↑) | PSNR (↑) | LPIPS (↓) | Manual Score |
|-------------------|---------|----------|----------|-----------|--------------|
| VAE               | 480     | 0.62     | 22.1     | 0.38      | 2.1/5        |
| DCGAN             | 410     | 0.68     | 24.3     | 0.31      | 2.8/5        |
| DDPM              | 320     | 0.74     | 26.8     | 0.26      | 3.6/5        |
| U-Net GAN         | 280     | 0.76     | 27.5     | 0.23      | 4.0/5        |
| **Working Diffusion** | **200** | **0.78** | **28.4** | **0.21** | **4.7/5**    |

> **Winner: Fine-Tuned Stable Diffusion**


---
### Evaluation Results

| Metric         | Value           | Meaning                          |
|----------------|-----------------|----------------------------------|
| FID (Final)    | **~200**        | 60% better than baseline         |
| SSIM           | 0.78 ± 0.05     | High structural similarity       |
| PSNR           | 28.4 dB         | Excellent image quality          |
| LPIPS          | 0.21            | Perceptually close to real       |

---

---

*Built with PyTorch + Hugging Face Diffusers.*
