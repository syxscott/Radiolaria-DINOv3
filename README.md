<div align="center">

# Few-Shot Fine-Grained Radiolarian Fossil Identification Using DINOv3
# 基于 DINOv3 的放射虫化石小样本细粒度识别

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/DINOv3-Meta_AI-blue.svg)](https://github.com/facebookresearch/dinov3)

**A Task-Adaptive Pre-training Framework for Extreme Few-Shot Learning in Palaeontology**
<br>
**古生物学极端小样本学习任务自适应预训练框架**

[English](#english) | [中文](#chinese)

</div>

---

<a name="english"></a>
## 🇬🇧 English

### 📋 Table of Contents
- [Overview](#overview-en)
- [Key Features](#features-en)
- [Installation](#installation-en)
- [Quick Start](#quickstart-en)
- [Citation](#citation-en)

<a name="overview-en"></a>
### 🎯 Overview

This repository provides the official implementation for the paper:

> **Few-Shot Fine-Grained Radiolarian Fossil Identification Using DINOv3: Insights into Permian-Triassic Mass Extinction Dynamics**

Radiolarian microfossils present unique challenges for automated identification due to:
- **Extreme data scarcity**: ~15 images per taxon
- **Fine-grained discrimination**: Subtle morphological differences
- **Domain shift**: Microscopy imagery differs significantly from natural images
- **Variable preservation**: Fragmentation and sedimentary occlusion

We propose a **Task-Adaptive Pre-training (TAPT)** framework that leverages self-supervised learning to adapt DINOv3 representations to the radiolarian domain before supervised fine-tuning.

<a name="features-en"></a>
### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **DINOv3 Integration** | Discriminative self-distillation + masked image modeling |
| 🎯 **TAPT Pipeline** | Self-supervised domain adaptation using unlabeled data |
| 📉 **Layer-wise LR Decay** | Optimized fine-tuning preserving early-layer features |
| 📊 **Unified Evaluation** | k-NN, Linear Probing, Prototype, Full Fine-tuning |

<a name="installation-en"></a>
### 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# Create environment
conda create -n dinov3-radiolaria python=3.9
conda activate dinov3-radiolaria

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

<a name="quickstart-en"></a>
### ⚡ Quick Start

```bash
# Phase A: Baseline Evaluation
python core/baseline.py \
  --data_root ./data \
  --weights /path/to/dinov3_vitl16_pretrain.pth \
  --model_size vitl16

# Phase B: Task-Adaptive Pre-training (TAPT)
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain.pth

# Phase C: Supervised Fine-tuning
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage

# Phase D: Few-Shot Evaluation
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn --n-way 5 --k-shot 1 --episodes 600
```

---

<a name="chinese"></a>
## 🇨🇳 中文

### 📋 目录
- [概述](#overview-cn)
- [核心特性](#features-cn)
- [安装](#installation-cn)
- [快速开始](#quickstart-cn)
- [引用](#citation-cn)

<a name="overview-cn"></a>
### 🎯 概述

本仓库提供了论文的官方实现：

> **基于 DINOv3 的放射虫化石小样本细粒度识别：二叠纪-三叠纪大灭绝动态的启示**

放射虫微体化石的自动识别面临独特挑战：
- **极端数据稀缺**：每个分类单元约 15 张图像
- **细粒度判别**：形态差异细微
- **域偏移**：显微图像与自然图像差异显著
- **保存状况不一**：碎片化和沉积物遮挡

我们提出了**任务自适应预训练（TAPT）**框架，利用自监督学习在监督微调前将 DINOv3 表征适应到放射虫领域。

<a name="features-cn"></a>
### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🧬 **DINOv3 集成** | 判别式自蒸馏 + 掩码图像建模 |
| 🎯 **TAPT 流程** | 使用无标签数据进行自监督域适应 |
| 📉 **分层学习率衰减** | 优化微调，保留浅层特征 |
| 📊 **统一评估** | k-NN、线性探测、原型分类、完全微调 |

<a name="installation-cn"></a>
### 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# 创建环境
conda create -n dinov3-radiolaria python=3.9
conda activate dinov3-radiolaria

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

<a name="quickstart-cn"></a>
### ⚡ 快速开始

```bash
# 阶段 A：基线评估
python core/baseline.py \
  --data_root ./data \
  --weights /path/to/dinov3_vitl16_pretrain.pth \
  --model_size vitl16

# 阶段 B：任务自适应预训练（TAPT）
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain.pth

# 阶段 C：监督微调
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage

# 阶段 D：小样本评估
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn --n-way 5 --k-shot 1 --episodes 600
```

---

## 📊 Training Configuration | 训练配置

### Table 1. Hyperparameters used for TAPT | TAPT 超参数表

| Parameter | Setting | 参数 | 设置 |
|-----------|---------|------|------|
| Batch size | 32 per GPU | 批次大小 | 每 GPU 32 |
| Total iterations | 40,000 | 总迭代次数 | 40,000 |
| Learning rate | 2×10⁻⁴ | 学习率 | 2×10⁻⁴ |
| Warmup | 1,000 iterations | 预热 | 1,000 次迭代 |
| Weight decay | 0.04 | 权重衰减 | 0.04 |
| Centering | Sinkhorn–Knopp | 中心化 | Sinkhorn–Knopp |
| Teacher momentum | 0.996 | 教师动量 | 0.996 |

---

## 📁 Project Structure | 项目结构

```
Radiolaria-DINOv3/
├── core/                      # Core scripts | 核心脚本
│   ├── baseline.py           # Phase A: Baseline | 阶段A：基线
│   ├── train_ssl.py          # Phase B: TAPT | 阶段B：自监督预训练
│   ├── train_cls.py          # Phase C: Fine-tuning | 阶段C：微调
│   └── fewshot.py            # Phase D: Evaluation | 阶段D：评估
├── configs/                   # Configurations | 配置文件
│   └── pretrain/
│       ├── vitb16_domain_adapt.yaml
│       ├── vitl16_domain_adapt.yaml
│       └── vits16_domain_adapt.yaml
├── dinov3/                    # DINOv3 library | DINOv3 库
├── utils/                     # Utilities | 工具函数
│   ├── data_utils.py
│   ├── prepare_data.py
│   └── sample_data_proportions.py
├── scripts/                   # Helper scripts | 辅助脚本
│   ├── reproduce_paper.py
│   └── test_project.py
├── requirements.txt           # Dependencies | 依赖项
└── README.md                  # This file | 本文件
```

---

<a name="citation-en"></a>
<a name="citation-cn"></a>
## 📝 Citation | 引用

If you find this code useful in your research, please cite our paper:
<br>
如果您在研究中使用了本代码，请引用我们的论文：

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

## 🤝 Acknowledgements | 致谢

- **DINOv3**: We thank Meta AI for open-sourcing the DINOv3 library.
- **Palaeontological Insights**: This work builds upon decades of radiolarian research.

---

## 📄 License | 许可证

This project is licensed under the **CC BY-NC 4.0** license.
<br>
本项目采用 **CC BY-NC 4.0** 许可证。

> **Note**: The DINOv3 library is licensed under CC-BY-NC 4.0.
> <br>**注意**：DINOv3 库采用 CC-BY-NC 4.0 许可证。

---

<div align="center">

**Happy Researching! 🧪🔬 | 研究愉快！**

</div>
