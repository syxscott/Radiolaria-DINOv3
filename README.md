Fine-Grained Few-Shot Classification with DINOv3This repository provides an official implementation for fine-grained few-shot classification using DINOv3 (Distillation with no labels v3).Targeting challenges such as extreme class imbalance, scarcity of samples (e.g., ~15 images per class), and high inter-class similarity, we propose a Task-Adaptive Pre-training (TAPT) framework. This approach leverages unlabeled domain-specific data to continuously pre-train the vision backbone before supervised fine-tuning, significantly enhancing feature robustness in data-scarce scenarios.🚀 Key FeaturesEmbedded DINOv3 Core: Fully integrated DINOv3 library, no external submodules required.Automated TAPT Pipeline: One-click scripts for self-supervised domain adaptation using your own data.Layer-wise Learning Rate Decay (LLRD): Optimized fine-tuning strategy for Vision Transformers.Unified Evaluation Suite: Integrated support for k-NN, Linear Probing, ProtoNet, and Full Fine-tuning.Robust Data Loading: Built-in fault tolerance for corrupted images during training.🛠️ InstallationWe recommend using Anaconda to manage the environment:conda create -n dinov3-project python=3.9
conda activate dinov3-project

# Install PyTorch (Adjust CUDA version as needed)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install dependencies
pip install -r requirements.txt
📂 Data Preparation1. Organize Raw ImagesPlace all your images (e.g., .jpg, .png, .tif) into a source directory. The structure can be flat or hierarchical.2. (Optional) Prepare CSV MetadataIf you have label information, prepare train.csv and test.csv in the ./data directory.Format:filepath,label
image_001.jpg,Class_A
image_002.jpg,Class_B
...
3. Format Dataset for Self-Supervised LearningRun the preparation script to organize data into the DINOv3-compatible ImageFolder format:# --symlink: Use symbolic links to save disk space
python utils/prepare_data.py --source /path/to/raw_images --dest ./data/images --symlink
Expected Directory Structure:data/
├── images/
│   └── train/
│       └── 0/          # All images linked here for SSL
│           ├── 00001_img.jpg
│           └── ...
└── splits_fixed/       # Stratified splits generated automatically
⚡ Quick StartPhase A: Baseline EvaluationEvaluate the off-the-shelf performance of official DINOv3 weights on your dataset without any training.python core/baseline.py \
  --data_root ./data \
  --weights /path/to/official/dinov3_vitl16_pretrain.pth \
  --model_size vitl16
Phase B: Unsupervised Domain Adaptation (TAPT)Perform self-supervised pre-training on your dataset.Supported architectures: vits16, vits16plus, vitb16, vitl16.# Example: Adaptation with ViT-Base
# Ensure you have downloaded the official pretrained weights first
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=/path/to/official/dinov3_vitb16_pretrain.pth
Note: Logs and checkpoints are saved to --output-dir. Auto-resume is supported.Phase C: Supervised Fine-tuning (SFT)Fine-tune the adapted model using label information.python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage \
  --lr 5e-4
Phase D: Few-Shot EvaluationAssess performance under N-way K-shot settings (e.g., 5-way 1-shot).python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn \
  --n-way 5 \
  --episodes 600
📊 Example ResultsModel BackbonePre-training StrategySFT Accuracy (Top-1)5-Way 1-Shot AccViT-BaseImageNet-21k32.1%29.5%DINOv3-BaseOfficial Weights45.3%58.4%DINOv3-BaseTAPT (Ours)52.8%65.3%(Results are for demonstration purposes only)🤝 AcknowledgementsThis project is built upon the open-source implementation of Meta AI DINOv3. The core library is included under the CC-BY-NC 4.0 license.