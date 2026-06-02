# Generative Adversarial Networks (GNA) Collection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)


## Project Overview
This repository collects practical experiments and example implementations of Generative Adversarial Networks (GANs), including a DCGAN example and exploratory notebooks. It is intended as a portfolio project demonstrating understanding of GAN theory, implementation, and experimentation.

## Quick Links
- Getting started: `requirements.txt` and `README.md` (this file)
- Notebooks: `DCGAN/deep_convolutional_GAN.ipynb`, `GAN/GAN.ipynb`
- Scripts: `DCGAN/DCGAN_script.py`, `GAN/understand_gna.py`, `GPU_vs_CPU/gpu_cpu_script.py`
- Documentation: `docs/project_summary.md`, `docs/technical_report.md`, `docs/resume_pitch.md`

## Quick Install
Recommended: create a Python virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Simple usage (run notebook or script):

```powershell
python DCGAN/DCGAN_script.py
# or open and run the notebooks in JupyterLab
jupyter lab
```

## Folder Structure
- `DCGAN/` — DCGAN scripts and notebook
- `GAN/` — vanilla GAN notebook and supporting scripts
- `GPU_vs_CPU/` — small GPU vs CPU experiment
- `docs/` — high-level project summary and technical report
- `results/` — placeholder for charts and trained models
- `models/` — recommended place to store exported model weights

## Reproducibility
- See `requirements.txt` for exact dependencies
- Notebooks contain runnable cells; run them top-to-bottom after installing deps

## Results and Visuals
Place training curves, generated samples, and model checkpoints under `results/` and `models/` respectively. Suggested plots and charts are described in `docs/technical_report.md`.

## License
This project is released under the MIT License. See `LICENSE`.

## Contact
Project prepared as an engineering portfolio piece. For questions or suggested improvements, open an issue or contact the author via GitHub.
