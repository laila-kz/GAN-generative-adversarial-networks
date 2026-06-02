# Technical Report

## Overview
This technical report summarizes the theory, algorithms, mathematical intuition, and implementation choices for the GAN experiments in this repository.

## Theory and Algorithm
Generative Adversarial Networks (GANs) [Goodfellow et al., 2014] are a class of generative models where two neural networks — a generator G and a discriminator D — are trained in opposition. The generator learns to map random noise z to data space, trying to produce samples that the discriminator cannot distinguish from real data.

Objective (minimax game):

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x\sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1 - D(G(z)))]
$$

Key training insights:
- Optimize D to maximize V, then optimize G to minimize V (or equivalently maximize \(\mathbb{E}[\log D(G(z))]\) in practice with non-saturating loss).
- Use batch normalization and careful learning rates for stability.
- Use convolutional architectures (DCGAN) for image generation.

## Mathematical Intuition
Under ideal conditions and sufficient model capacity, the generator distribution \(p_g\) converges to the data distribution \(p_{data}\) when the minimax game reaches its global optimum. The training is related to minimizing the Jensen-Shannon divergence between distributions.

## Implementation Choices
- Framework: PyTorch (flexible, widely used for research and production).
- DCGAN architecture: follows canonical choices (stride-2 convolutions, transposed convolutions, batchnorm, LeakyReLU for D, ReLU for G final tanh for image outputs).
- Optimizers: Adam with recommended betas (e.g., 0.5, 0.999) for stability.
- Data: images folder contains small datasets used in notebooks. Users can swap in their own dataset.

## Reproducibility and Running
1. Create virtual environment and install `requirements.txt`.
2. Run notebooks top-to-bottom or scripts from command line.
3. Save generated samples, training loss curves, and model checkpoints to `results/` and `models/`.

## Suggested Visualizations (to add under `results/`)
- Generator loss vs Discriminator loss over training epochs.
- Sample grids of generated images at fixed epoch checkpoints.
- FID or IS metrics comparison (if implemented).

## Notes on Limitations
- This repository is educational and focuses on clarity and reproducibility. It does not include large-scale training scripts or automated benchmarking.
- For production or research-grade experiments, add seed control, deterministic CUDA settings, and experiment logging (TensorBoard or Weights & Biases).

## References
- Goodfellow et al., Generative Adversarial Nets (2014)
- Radford et al., Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015)
