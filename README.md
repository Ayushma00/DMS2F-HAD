# DMS2F-HAD: Dual Mamba Spectral-Spatial Fusion for Hyperspectral Anomaly Detection

**DMS2F-HAD** is a deep learning framework for hyperspectral anomaly detection using a dual-branch Mamba architecture that adaptively fuses spectral and spatial features with gated  blocks.


## Features

- Dual-branch Mamba architecture (spectral + spatial)
- Gated fusion for enhanced anomaly detection
- Patch-wise image preprocessing with random masking
- Built-in evaluation and ROC visualization tools
- PyTorch-based, modular, and easily extendable

## Installation

```bash
# 1. Create environment
conda create -n mamba_env python=3.12 -y
conda activate mamba_env

# 2. Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore Ninja mmdet==2.25.3
pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-gpu
pip3 install adeval fastprogress geomloss FrEIA mamba_ssm triton causal_conv1d numpy-hilbert-curve pyzorder
conda install -c conda-forge accimage
pip install -r requirements.txt

## 3. Download folder

Download the code folder
cd DMS2FHAD
```

##  Project Structure:
```
📁 DMS2F-HAD/
├── data/                          
├── artifacts/
│   ├── models/
│   └── results/
├── experiments/
│   ├── ablation/
├── docs/
├── src/                
│   ├── block_utils.py
│   ├── mask.py
│   ├── train.py 
│   ├── data_loader.py.py    
│   ├── mamba_simple.py 
│   ├── model.py     
│   └── analyze_residual.py 
├── main.py
├── requirements.txt
└── README.md

```
##  Prepare Dataset:

Datasets are available in `DMS2F-HAD/data`.
```shell
-- cri.mat
-- aviris1.mat
-- aviris2.mat
-- SanDiego.mat
```

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 
python main.py
python src/analyze_residual.py

```
## Demo

```bash
python src/analyze_residual.py
```
Results are saved in **artifacts/results/current**.
## Software Requirements

- **OS**: Ubuntu 20+, macOS, or Windows 10+
- **Python**: 3.12.9
- **PyTorch**: ≥ 2.5.1 with CUDA 12.1
- **GPU**: Recommended (NVIDIA,greater than A100 with CUDA installed)
- **Package Manager**: [Anaconda](https://www.anaconda.com/download/) (recommended)
