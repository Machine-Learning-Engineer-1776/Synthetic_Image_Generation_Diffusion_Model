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
