# 📦 DMS2FHAD

> **DMS2FHAD** is a Dual Mamba-based Spectral and Spatial Fusion Network for detecting anomalies in hyperspectral images.

---

## 📑 Table of Contents

- [About](#about)
- [Features](#features)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## About

**DMS2FHAD** (Dual Mamba Spectral-Spatial Fusion for Hyperspectral Anomaly Detection) is a deep learning framework designed to identify anomalies in hyperspectral images by leveraging both spatial and spectral information. It employs a dual-branch architecture powered by Mamba blocks, enabling adaptive feature fusion and robust anomaly detection across complex hyperspectral datasets.

---

## Features

- 📊 Patch-wise preprocessing of hyperspectral images.
- 🧠 Dual-branch anomaly detection using spatial and spectral information with gated fusion.
- 🎭 Random binary masking for self-supervised learning.
- 📦 Modular and extendable PyTorch-based architecture.
- 📈 Built-in evaluation and visualization tools for detection results.

---

## Software Requirements

Ensure the following dependencies are installed:

- **Operating System**: Linux (Ubuntu 20+), macOS, or Windows 10+
- **Python Version**: 3.12.9 (recommended to use [Anaconda](https://www.anaconda.com/download/))
- **PyTorch**: 2.5.1+ with CUDA 12.1 support
- **CUDA Toolkit**: Recommended for GPU acceleration
- **NVIDIA GPU**: Required for training on GPU

Python packages listed in `requirements.txt`:

- `numpy`
- `torch`
- `matplotlib`
- `scikit-learn`
- `h5py`
- `tqdm`

---

## Installation (Linux + Conda)

### 1. Clone and Set Up Conda Environment

```bash
# Clone the repository
Download the repo
cd DMS2FHAD

# Create a new conda environment with Python 3.12
conda create -n dms2fhad python=3.12 -y

# Activate the environment
conda activate dms2fhad

# Install required packages
pip install -r requirements.txt
```

##  Prepare Dataset:

Datasets are available in `DMS2F-HAD/Data`.
```shell
-- cri.mat
-- aviris1.mat
-- gulfport.mat
-- aviris2.mat
-- cat-island.mat
-- SanDiego.mat
```
##  Project Structure:
```
📁 DMS2F-HAD/
├── Data/               
├── models/             
├── src/                
│   ├── block_utils.py
│   ├── mask.py
│   ├── train.py 
│   ├── data_loader.py.py    
│   ├── mamba_simple.py 
│   ├── model.py     
│   └── evaluate.py 
├── main.py
├── requirements.txt
└── README.md

```

## Usage
### 1. Training the Model

To train the model on a specific dataset:

```bash
python main.py --mode train --data_path data/los-angeles-1.mat --save_path outputs/ --epochs 100

