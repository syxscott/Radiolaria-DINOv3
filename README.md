<div align="center">

# Radiolaria-DINOv3

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/DINOv3-Meta_AI-blue.svg)](https://github.com/facebookresearch/dinov3)

**Task-Adaptive Pre-training Framework for Few-Shot Fine-Grained Classification**

[🇬🇧 English](#english) | [🇨🇳 中文](#chinese) | [🇯🇵 日本語](#japanese)

</div>

---

<a name="english"></a>
## 🇬🇧 English

### 📋 Table of Contents
- [Overview](#overview-en)
- [Key Features](#features-en)
- [Installation](#installation-en)
- [Data Preparation](#data-en)
- [Quick Start](#quickstart-en)
- [Project Structure](#structure-en)

<a name="overview-en"></a>
### 🎯 Overview

This repository provides a DINOv3-based framework for fine-grained few-shot classification under extreme data scarcity. It implements a Task-Adaptive Pre-training (TAPT) pipeline that leverages self-supervised learning to adapt vision representations to domain-specific data before supervised fine-tuning.

**Key Challenges Addressed:**
- Extreme data scarcity (~15 images per class)
- Fine-grained inter-class discrimination
- Domain shift from natural images
- Variable data quality and preservation

<a name="features-en"></a>
### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **DINOv3 Core** | Discriminative self-distillation + masked image modeling |
| 🎯 **TAPT Pipeline** | Self-supervised domain adaptation using unlabeled target data |
| 📉 **LLRD** | Layer-wise Learning Rate Decay for ViT fine-tuning |
| 📊 **Unified Evaluation** | k-NN, Linear Probing, Prototype, Full Fine-tuning |
| 🛡️ **Robust Loading** | Fault-tolerant image loading with automatic fallback |
| 📐 **Multi-scale Crops** | Global (224²) + Local (96²) views for morphology-aware learning |

<a name="installation-en"></a>
### 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# Create environment
conda create -n dinov3 python=3.9
conda activate dinov3

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

<a name="data-en"></a>
### 📂 Data Preparation

```bash
# Prepare data for self-supervised learning
python utils/prepare_data.py \
  --source /path/to/raw_images \
  --dest ./data/images \
  --symlink

# Create stratified splits for ablation study
python utils/sample_data_proportions.py \
  --csv-path ./data/train.csv \
  --output-dir ./data/splits_proportions
```

**Expected Directory Structure:**
```
data/
├── images/
│   └── train/
│       └── 0/          # All images for SSL
└── splits_proportions/
    ├── 0%/
    ├── 20%/
    ├── 50%/
    ├── 80%/
    └── 100%/
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

<a name="structure-en"></a>
### 📁 Project Structure

```
Radiolaria-DINOv3/
├── core/                      # Core training scripts
│   ├── baseline.py           # Baseline evaluation
│   ├── train_ssl.py          # Self-supervised pre-training
│   ├── train_cls.py          # Supervised fine-tuning
│   └── fewshot.py            # Few-shot evaluation
├── configs/                   # Configuration files
│   └── pretrain/
│       ├── vitb16_domain_adapt.yaml
│       ├── vitl16_domain_adapt.yaml
│       └── vits16_domain_adapt.yaml
├── dinov3/                    # DINOv3 library (embedded)
├── utils/                     # Utility functions
│   ├── data_utils.py
│   ├── prepare_data.py
│   └── sample_data_proportions.py
├── scripts/                   # Helper scripts
│   ├── reproduce_paper.py
│   └── test_project.py
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

<a name="chinese"></a>
## 🇨🇳 中文

### 📋 目录
- [概述](#overview-cn)
- [核心特性](#features-cn)
- [安装](#installation-cn)
- [数据准备](#data-cn)
- [快速开始](#quickstart-cn)
- [项目结构](#structure-cn)

<a name="overview-cn"></a>
### 🎯 概述

本仓库提供基于 DINOv3 的细粒度小样本分类框架，适用于极端数据稀缺场景。实现了任务自适应预训练（TAPT）流程，利用自监督学习在监督微调前将视觉表征适应到特定领域数据。

**解决的关键挑战：**
- 极端数据稀缺（每类约 15 张图像）
- 细粒度类间判别
- 与自然图像的域偏移
- 数据质量和保存状况不一

<a name="features-cn"></a>
### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🧬 **DINOv3 核心** | 判别式自蒸馏 + 掩码图像建模 |
| 🎯 **TAPT 流程** | 使用无标签目标数据进行自监督域适应 |
| 📉 **LLRD** | ViT 微调的分层学习率衰减 |
| 📊 **统一评估** | k-NN、线性探测、原型分类、完全微调 |
| 🛡️ **鲁棒加载** | 容错图像加载，自动降级处理 |
| 📐 **多尺度裁剪** | 全局 (224²) + 局部 (96²) 视图，感知形态学习 |

<a name="installation-cn"></a>
### 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# 创建环境
conda create -n dinov3 python=3.9
conda activate dinov3

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

<a name="data-cn"></a>
### 📂 数据准备

```bash
# 为自监督学习准备数据
python utils/prepare_data.py \
  --source /path/to/raw_images \
  --dest ./data/images \
  --symlink

# 为消融实验创建分层划分
python utils/sample_data_proportions.py \
  --csv-path ./data/train.csv \
  --output-dir ./data/splits_proportions
```

**预期目录结构：**
```
data/
├── images/
│   └── train/
│       └── 0/          # 自监督学习用图像
└── splits_proportions/
    ├── 0%/
    ├── 20%/
    ├── 50%/
    ├── 80%/
    └── 100%/
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

<a name="structure-cn"></a>
### 📁 项目结构

```
Radiolaria-DINOv3/
├── core/                      # 核心训练脚本
│   ├── baseline.py           # 基线评估
│   ├── train_ssl.py          # 自监督预训练
│   ├── train_cls.py          # 监督微调
│   └── fewshot.py            # 小样本评估
├── configs/                   # 配置文件
│   └── pretrain/
│       ├── vitb16_domain_adapt.yaml
│       ├── vitl16_domain_adapt.yaml
│       └── vits16_domain_adapt.yaml
├── dinov3/                    # DINOv3 库（嵌入式）
├── utils/                     # 工具函数
│   ├── data_utils.py
│   ├── prepare_data.py
│   └── sample_data_proportions.py
├── scripts/                   # 辅助脚本
│   ├── reproduce_paper.py
│   └── test_project.py
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

---

<a name="japanese"></a>
## 🇯🇵 日本語

### 📋 目次
- [概要](#overview-jp)
- [主な機能](#features-jp)
- [インストール](#installation-jp)
- [データ準備](#data-jp)
- [クイックスタート](#quickstart-jp)
- [プロジェクト構成](#structure-jp)

<a name="overview-jp"></a>
### 🎯 概要

このリポジトリは、極端なデータ不足状況下での細粒度少数ショット分類のための DINOv3 ベースフレームワークを提供します。タスク適応型事前学習（TAPT）パイプラインを実装し、教師ありファインチューニング前に自己教師あり学習を用いて視覚表現をドメイン固有データに適応させます。

**解決する主要な課題：**
- 極端なデータ不足（クラスあたり約 15 画像）
- 細粒度のクラス間識別
- 自然画像からのドメインシフト
- データ品質と保存状態のばらつき

<a name="features-jp"></a>
### ✨ 主な機能

| 機能 | 説明 |
|------|------|
| 🧬 **DINOv3 コア** | 判別的自己蒸留 + マスク画像モデリング |
| 🎯 **TAPT パイプライン** | ラベルなしターゲットデータを用いた自己教師ありドメイン適応 |
| 📉 **LLRD** | ViT ファインチューニングのための層別学習率減衰 |
| 📊 **統一評価** | k-NN、線形探査、プロトタイプ、完全ファインチューニング |
| 🛡️ **堅牢なローディング** | 自動フォールバックを備えたフォールトトレラント画像ローディング |
| 📐 **マルチスケールクロップ** | 形態認識学習のためのグローバル (224²) + ローカル (96²) ビュー |

<a name="installation-jp"></a>
### 🛠️ インストール

```bash
# リポジトリのクローン
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3

# 環境の作成
conda create -n dinov3 python=3.9
conda activate dinov3

# PyTorch のインストール（CUDA バージョンに合わせて調整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 依存関係のインストール
pip install -r requirements.txt
```

<a name="data-jp"></a>
### 📂 データ準備

```bash
# 自己教師あり学習用のデータ準備
python utils/prepare_data.py \
  --source /path/to/raw_images \
  --dest ./data/images \
  --symlink

# アブレーション研究のための層別分割の作成
python utils/sample_data_proportions.py \
  --csv-path ./data/train.csv \
  --output-dir ./data/splits_proportions
```

**予想されるディレクトリ構造：**
```
data/
├── images/
│   └── train/
│       └── 0/          # 自己教師あり学習用画像
└── splits_proportions/
    ├── 0%/
    ├── 20%/
    ├── 50%/
    ├── 80%/
    └── 100%/
```

<a name="quickstart-jp"></a>
### ⚡ クイックスタート

```bash
# フェーズ A：ベースライン評価
python core/baseline.py \
  --data_root ./data \
  --weights /path/to/dinov3_vitl16_pretrain.pth \
  --model_size vitl16

# フェーズ B：タスク適応型事前学習（TAPT）
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain.pth

# フェーズ C：教師ありファインチューニング
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage

# フェーズ D：少数ショット評価
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn --n-way 5 --k-shot 1 --episodes 600
```

<a name="structure-jp"></a>
### 📁 プロジェクト構成

```
Radiolaria-DINOv3/
├── core/                      # コアトレーニングスクリプト
│   ├── baseline.py           # ベースライン評価
│   ├── train_ssl.py          # 自己教師あり事前学習
│   ├── train_cls.py          # 教師ありファインチューニング
│   └── fewshot.py            # 少数ショット評価
├── configs/                   # 設定ファイル
│   └── pretrain/
│       ├── vitb16_domain_adapt.yaml
│       ├── vitl16_domain_adapt.yaml
│       └── vits16_domain_adapt.yaml
├── dinov3/                    # DINOv3 ライブラリ（埋め込み）
├── utils/                     # ユーティリティ関数
│   ├── data_utils.py
│   ├── prepare_data.py
│   └── sample_data_proportions.py
├── scripts/                   # ヘルパースクリプト
│   ├── reproduce_paper.py
│   └── test_project.py
├── requirements.txt           # Python 依存関係
└── README.md                  # このファイル
```

---

## 📊 Training Configuration | 訓練設定 | 訓練設定

### Table 1. Hyperparameters for TAPT | TAPT 超参数 | TAPT ハイパーパラメータ

| Parameter | 参数 | パラメータ | Setting | 設定 | 設定 |
|-----------|------|-----------|---------|------|------|
| Batch size | 批次大小 | バッチサイズ | 32 per GPU | 每 GPU 32 | GPU あたり 32 |
| Total iterations | 总迭代次数 | 総イテレーション数 | 40,000 | 40,000 | 40,000 |
| Learning rate | 学习率 | 学習率 | 2×10⁻⁴ | 2×10⁻⁴ | 2×10⁻⁴ |
| Warmup | 预热 | ウォームアップ | 1,000 iters | 1,000 次迭代 | 1,000 イテレーション |
| Weight decay | 权重衰减 | 重み減衰 | 0.04 | 0.04 | 0.04 |
| Centering | 中心化 | 中心化 | Sinkhorn-Knopp | Sinkhorn-Knopp | Sinkhorn-Knopp |
| Teacher momentum | 教师动量 | 教師運動量 | 0.996 | 0.996 | 0.996 |

---

## 🤝 Acknowledgements | 致谢 | 謝辞

- **DINOv3**: We thank Meta AI for open-sourcing the DINOv3 library.
- **DINOv3**: 感谢 Meta AI 开源 DINOv3 库。
- **DINOv3**: Meta AI の DINOv3 ライブラリオープンソースに感謝します。

---

## 📄 License | 许可证 | ライセンス

This project is licensed under the **CC BY-NC 4.0** license.
<br>
本项目采用 **CC BY-NC 4.0** 许可证。
<br>
このプロジェクトは **CC BY-NC 4.0** ライセンスの下で提供されています。

> **Note**: The DINOv3 library is licensed under CC-BY-NC 4.0.
> <br>**注意**：DINOv3 库采用 CC-BY-NC 4.0 许可证。
> <br>**注**：DINOv3 ライブラリは CC-BY-NC 4.0 ライセンスです。

---

<div align="center">

**Happy Researching! 🧪🔬**

**研究愉快！ 🧪🔬**

**研究を楽しみましょう！ 🧪🔬**

</div>
