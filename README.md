# Abstract

We present a novel method for diffusion-guided frameworks for view-consistent super-resolution (SR) in neural rendering. Our approach leverages existing 2D SR models in conjunction with advanced techniques such as Variational Score Distilling (VSD) and a LoRA fine-tuning helper, with spatial training to significantly boost the quality and consistency of upscaled 2D images compared to the previous methods in the literature, such as Renoised Score Distillation (RSD) proposed in DiSR-NeRF, or SDS proposed in DreamFusion. The VSD score facilitates precise fine-tuning of SR models, resulting in high-quality, view-consistent images. To address the common challenge of inconsistencies among independent SR 2D images, we integrate Iterative 3D Synchronization (I3DS) from the DiSR-NeRF framework. Our quantitative benchmarks and qualitative results on the LLFF dataset demonstrate the superior performance of our system compared to existing methods such as DiSR-NeRF.

# Installation

```bash
git clone https://github.com/shreyvish5678/SR-NeRF-with-Variational-Diffusion-Strategies
cd SR-NeRF-with-Variational-Diffusion-Strategies

conda create --name nerf-sr-vsd
conda activate nerf-sr-vsd

pip install -r requirements.txt
```
Please look through the following projects and install their dependencies as well: [pytorch3d](https://github.com/facebookresearch/pytorch3d), [tinycudann](https://github.com/NVlabs/tiny-cuda-nn), and [threestudio](https://github.com/threestudio-project/threestudio)

# Running

Configure everything in your config file, such as `path/to/config.yaml`, such as dataset directory, the method you want to utilize, and hyperparameters, and then run the following:
`python launch.py --config path/to/config.yaml --train`

The testing and training results should both be generated.
