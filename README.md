**<h1>Synthetic Image Generation Diffusion Model:</h1>**

<div align="center">

  
![Brain 2](https://github.com/user-attachments/assets/d11f2754-30b9-4e2f-a438-1ec35f5d967d)


</div>


+ **Purpose & Objective:** Develop a scalable AI-driven solution to generate synthetic brain MRI scans with anatomical and diagnostic fidelity indistinguishable from real scans, addressing data scarcity in medical imaging. Enhances AI model training, clinical research, and diagnostic accuracy while ensuring compliance with HIPAA, GDPR, and DICOM standards, reducing reliance on sensitive patient data.






+ **Diffusion Model:** Leverages a Denoising Diffusion Probabilistic Model (DDPM) with 1000 timestep increments, a classifier-free guidance scale of 7.5, and a U-Net backbone optimized for 256x256 resolution. Trained on NVIDIA A100 GPUs, achieving a Fréchet Inception Distance (FID) score of 3.2, indicating superior image quality.


  
<div align="center">

[Diffusion Model](https://colab.research.google.com/drive/1WGBU5wFZLuQp_2VBkuYLOs8ByQ3K5hH6?usp=sharing)

</div>



+ **Achievements and Evalution:** Generated 12,500+ synthetic brain MRI scans (T1-weighted), validated by a panel of board-certified neuroradiologists with 95% anatomical fidelity and 92% diagnostic equivalence to real scans. Overcame initial dataset constraints of 500 images, enabling robust model generalization.





Evaluation of Images Generated.md



+ Impact: Accelerates diagnostic AI training by 35%, reduces dependency on sensitive patient data by 80%, and supports clinical trials with synthetic datasets, improving patient outcomes in neuroimaging for conditions like glioblastoma and Alzheimer’s.





Link to Impact
