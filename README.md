# Few-Shot Fine-Grained Radiolarian Fossil Identification Using DINOv3

<div align="center">

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/DINOv3-Meta_AI-blue.svg)](https://github.com/facebookresearch/dinov3)

**A Task-Adaptive Pre-training Framework for Extreme Few-Shot Learning in Palaeontology**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
- [Training & Evaluation](#-training--evaluation)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This repository provides the official implementation for the paper:

> **Few-Shot Fine-Grained Radiolarian Fossil Identification Using DINOv3: Insights into Permian-Triassic Mass Extinction Dynamics**

Radiolarian microfossils present unique challenges for automated identification due to:
- **Extreme data scarcity**: ~15 images per taxon
- **Fine-grained discrimination**: Subtle morphological differences
- **Domain shift**: Microscopy imagery differs significantly from natural images
- **Variable preservation**: Fragmentation and sedimentary occlusion

We propose a **Task-Adaptive Pre-training (TAPT)** framework that leverages self-supervised learning to adapt DINOv3 representations to the radiolarian domain before supervised fine-tuning, achieving state-of-the-art performance under extreme few-shot settings.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **DINOv3 Integration** | Fully integrated DINOv3 library with discriminative self-distillation and masked image modeling |
| 🎯 **TAPT Pipeline** | One-click self-supervised domain adaptation using unlabeled target-domain data |
| 📉 **Layer-wise LR Decay** | Optimized fine-tuning strategy preserving early-layer representations while adapting higher layers |
| 📊 **Unified Evaluation** | Support for k-NN, Linear Probing, Prototype Classification, and Full Fine-tuning |
| 🛡️ **Robust Data Loading** | Fault-tolerant image loading with automatic fallback for corrupted files |
| 📐 **Multi-scale Crops** | Global (224×224) and local (96×96) views for morphology-aware representation learning |
| 🔗 **KoLeo Regularization** | Feature space uniformity enforcement to prevent cluster collapse |
| 📐 **Gram Anchoring** | Structural correlation preservation for high-frequency detail reconstruction |

---

## 🛠️ Installation

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.8 (recommended for faster training)
- 8GB+ GPU memory (for batch_size=32)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# 2. Create conda environment
conda create -n dinov3-radiolaria python=3.9
conda activate dinov3-radiolaria

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Image transformations and datasets |
| numpy | ≥1.21.0 | Numerical computations |
| pandas | ≥1.3.0 | Data manipulation and CSV handling |
| scikit-learn | ≥1.0.0 | Machine learning utilities |
| pyyaml | ≥5.3.0 | Configuration file parsing |
| omegaconf | ≥2.0.0 | Config management |
| termcolor | ≥1.1.0 | Colored terminal output |
| tqdm | ≥4.62.0 | Progress bars |
| scipy | ≥1.7.0 | Scientific computing |

---

## 📂 Data Preparation

### 1. Organize Raw Images

Place all radiolarian fossil images (`.jpg`, `.png`, `.tif`) into a source directory:

```
/path/to/raw_images/
├── radiolaria_001.jpg
├── radiolaria_002.jpg
├── ...
```

### 2. Prepare CSV Metadata (Optional but Recommended)

Create `train.csv` with filepath and label columns:

```csv
filepath,label
radiolaria_001.jpg,Albaillellaria_spp
radiolaria_002.jpg,Entactinaria_spp
radiolaria_003.jpg,Nassellaria_spp
```

### 3. Format Dataset for Self-Supervised Learning

Run the preparation script to organize data into DINOv3-compatible format:

```bash
# Using symbolic links (saves disk space)
python utils/prepare_data.py \
  --source /path/to/raw_images \
  --dest ./data/images \
  --symlink
```

**Expected Directory Structure:**

```
data/
├── images/
│   └── train/
│       └── 0/              # All images for SSL (DINOv3 ImageFolder format)
│           ├── 0000001_radiolaria_001.jpg
│           ├── 0000002_radiolaria_002.jpg
│           └── ...
└── splits_fixed/           # Auto-generated stratified splits
    ├── train_fixed.csv
    ├── val_fixed.csv
    └── test_fixed.csv
```

### 4. Create Data Proportions for Ablation Study (Optional)

For reproducing Section 3.1 experiments (0%, 20%, 50%, 80%, 100% data proportions):

```bash
python utils/sample_data_proportions.py \
  --csv-path ./data/train.csv \
  --output-dir ./data/splits_proportions
```

This creates stratified splits for each data proportion:

```
data/splits_proportions/
├── 0%/          # Empty training set (baseline with ImageNet weights)
├── 20%/
├── 50%/
├── 80%/
└── 100%/        # Full dataset
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## ⚡ Quick Start

### Phase A: Baseline Evaluation

Evaluate off-the-shelf DINOv3 performance without any training:

```bash
python core/baseline.py \
  --data_root ./data \
  --weights /path/to/dinov3_vitl16_pretrain.pth \
  --model_size vitl16 \
  --output_dir ./runs/baseline_vitl16
```

**Outputs:**
- k-NN accuracy (k=1, 5)
- Prototype classifier accuracy
- Linear probing accuracy (Logistic Regression)
- Confusion matrix

---

## 📊 Training & Evaluation

### Phase B: Task-Adaptive Pre-training (TAPT)

Perform self-supervised domain adaptation using unlabeled target-domain data:

```bash
# ViT-Base configuration (recommended for most users)
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain.pth

# ViT-Large configuration (higher capacity, more GPU memory)
python core/train_ssl.py \
  --config-file configs/pretrain/vitl16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitl16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitl16_pretrain.pth
```

**Training Configuration (from Section 3.1):**
- Batch size: 32 per GPU
- Total iterations: 40,000
- Learning rate: 2×10⁻⁴ (warmup: 1,000 iterations)
- Weight decay: 0.04
- Teacher momentum: 0.996
- Centering: Sinkhorn-Knopp

**Auto-resume:** Training automatically resumes from the latest checkpoint if interrupted.

### Phase C: Supervised Fine-tuning

Fine-tune the domain-adapted model using labeled data:

```bash
# Two-stage training: Linear probing → Full fine-tuning with LLRD
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage \
  --lr 5e-4 \
  --epochs 100
```

**Training Modes:**
- `linear`: Freeze backbone, train only classification head
- `full_ft`: End-to-end fine-tuning with uniform learning rate
- `two_stage`: Linear probing (20 epochs) → Full fine-tuning with LLRD

**Layer-wise Learning Rate Decay (LLRD):**
- Decay factor: γ = 0.75
- Learning rate for layer l: η_l = η_base × γ^(L-l)
- Preserves early-layer texture features while adapting higher layers

### Phase D: Few-Shot Evaluation

Assess performance under N-way K-shot settings:

```bash
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn \
  --n-way 5 \
  --k-shot 1 \
  --episodes 600 \
  --output-dir ./runs/fewshot_vitb16
```

**Evaluation Metrics:**
- k-NN accuracy with confidence intervals (95% CI)
- N-way K-shot classification accuracy
- Macro F1-score

---

## 🔄 Reproducing Paper Results

### Step 1: Data Preparation

```bash
# Prepare data for SSL
python utils/prepare_data.py --source /path/to/images --dest ./data/images

# Create stratified splits
python utils/sample_data_proportions.py --csv-path ./data/train.csv
```

### Step 2: TAPT with Different Data Proportions

For each proportion p ∈ {0%, 20%, 50%, 80%, 100%}:

```bash
# Use the appropriate split directory
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16_${p} \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain.pth
```

### Step 3: Few-Shot Evaluation

```bash
# Evaluate on the test set
python core/fewshot.py \
  --data-path ./data/splits_proportions/${p} \
  --weights ./runs/ssl_vitb16_${p}/checkpoint.pth \
  --model-size vitb16 \
  --run-knn \
  --n-way 5 \
  --k-shot 1 \
  --episodes 600
```

### Step 4: Training Configuration (from Table 1)

| Parameter | Setting |
|-----------|---------|
| Batch size | 32 per GPU |
| Total iterations | 40,000 |
| Learning rate | 2×10⁻⁴ |
| Warmup iterations | 1,000 |
| Weight decay | 0.04 |
| Centering | Sinkhorn-Knopp |
| Teacher momentum | 0.996 |

---

## 📁 Project Structure

```
Radiolaria-DINOv3/
├── core/                      # Core training and evaluation scripts
│   ├── baseline.py           # Phase A: Baseline evaluation
│   ├── train_ssl.py          # Phase B: TAPT (self-supervised)
│   ├── train_cls.py          # Phase C: Supervised fine-tuning
│   └── fewshot.py            # Phase D: Few-shot evaluation
├── configs/                   # Configuration files
│   └── pretrain/
│       ├── vitb16_domain_adapt.yaml   # ViT-Base configuration
│       ├── vitl16_domain_adapt.yaml   # ViT-Large configuration
│       ├── vits16_domain_adapt.yaml   # ViT-Small configuration
│       └── vits16plus_domain_adapt.yaml # ViT-Small+ configuration
├── dinov3/                    # DINOv3 library (embedded)
│   ├── models/               # Vision Transformer architectures
│   ├── train/                # Training loops
│   ├── loss/                 # Loss functions (DINO, iBOT, KoLeo, Gram)
│   ├── data/                 # Data augmentation and transforms
│   └── utils/                # Utility functions
├── utils/                     # Project-specific utilities
│   ├── data_utils.py         # Data loading and splitting
│   ├── prepare_data.py       # Data preparation script
│   └── sample_data_proportions.py  # Data proportion sampling
├── runs/                      # Training outputs (auto-created)
│   ├── baseline_*/           # Baseline evaluation results
│   ├── ssl_*/                # TAPT checkpoints and logs
│   └── sft_*/                # Supervised fine-tuning results
├── data/                      # Dataset directory (user-provided)
│   ├── images/               # SSL training images
│   └── splits_fixed/         # Auto-generated splits
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📝 Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{radiolaria2025dinov3,
  title={Few-Shot Fine-Grained Radiolarian Fossil Identification Using DINOv3: 
         Insights into Permian-Triassic Mass Extinction Dynamics},
  author={Scott, Yang and others},
  journal={Nature Communications},
  year={2025}
}
```

---

## 🤝 Acknowledgements

- **DINOv3**: We thank Meta AI for open-sourcing the DINOv3 library.
- **Palaeontological Insights**: This work builds upon decades of radiolarian research.
- **Computational Resources**: Supported by [Your Institution].

---

## 📄 License

This project is licensed under the **CC BY-NC 4.0** license. See the [LICENSE](LICENSE) file for details.

> **Note:** The DINOv3 library included in this repository is licensed under CC-BY-NC 4.0.

---

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact:

**Yang Scott**  
Email: syxscott@example.com  
GitHub: [@syxscott](https://github.com/syxscott)

---

<div align="center">

**Happy Researching! 🧪🔬**

</div>
