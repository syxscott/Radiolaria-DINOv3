# SSH 算力平台训练指南

<div align="center">

**从零开始在 Linux 算力平台上训练 DINOv3 模型**

</div>

---

## 📋 目录

1. [准备工作](#step0)
2. [SSH 连接](#step1)
3. [环境配置](#step2)
4. [数据准备与上传](#step3)
5. [预训练权重管理](#step4)
6. [开始训练](#step5)
7. [监控进度](#step6)
8. [常见问题](#faq)

---

<a name="step0"></a>
## 1. 准备工作

| 物品 | 说明 | 位置 |
|------|------|------|
| 🔑 SSH 密钥 | 用于连接服务器 | `~/.ssh/id_rsa` |
| 💻 终端软件 | MobaXterm / PowerShell / Terminal | 本地电脑 |
| 📁 数据集 | 放射虫化石图像 | `D:\数据集\dataR` |
| 🎯 预训练权重 | DINOv3 官方权重 | 自动下载到 `model/` |

### 生成 SSH 密钥（如果没有）
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 一路回车，密钥保存在 ~/.ssh/
```

---

<a name="step1"></a>
## 2. SSH 连接

```bash
# 基本连接
ssh -p 端口号 用户名@服务器 IP

# 示例
ssh -p 22 user@192.168.1.100

# 使用密钥
ssh -i ~/.ssh/id_rsa -p 22 user@192.168.1.100
```

---

<a name="step2"></a>
## 3. 环境配置

### 3.1 检查 GPU
```bash
nvidia-smi
```

### 3.2 创建 Conda 环境
```bash
conda create -n dinov3 python=3.9 -y
conda activate dinov3
```

### 3.3 安装 PyTorch（根据 CUDA 版本）
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3.4 安装项目依赖
```bash
git clone https://github.com/syxscott/Radiolaria-DINOv3.git
cd Radiolaria-DINOv3
pip install -r requirements.txt
```

### 3.5 验证安装
```bash
python scripts/test_project.py
```

---

<a name="step3"></a>
## 4. 数据准备与上传

### 📁 数据集格式要求

项目支持两种数据格式：

#### 格式 A：ImageFolder 格式（推荐用于自监督学习）
```
data/
└── images/
    └── train/
        └── 0/                    # 所有图像放在这里
            ├── image_001.jpg
            ├── image_002.png
            └── ...
```

#### 格式 B：CSV + 图像（推荐用于监督学习）
```
data/
├── images/                       # 图像文件夹
│   ├── class_A/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── class_B/
│       ├── img_003.jpg
│       └── img_004.jpg
└── train.csv                     # 标签文件
```

**CSV 格式示例：**
```csv
filepath,label
class_A/img_001.jpg,class_A
class_A/img_002.jpg,class_A
class_B/img_003.jpg,class_B
class_B/img_004.jpg,class_B
```

### 🔧 数据预处理

如果你的数据是 `D:数据集dataR` 格式，需要先转换：

```bash
# 假设你的数据在服务器上解压后为 ./raw_data/
# 转换为 ImageFolder 格式
python utils/prepare_data.py \
  --source ./raw_data \
  --dest ./data/images \
  --symlink

# 创建分层划分（用于消融实验）
python utils/sample_data_proportions.py \
  --csv-path ./data/train.csv \
  --output-dir ./data/splits_proportions
```

### 📤 上传数据到服务器

**方法 A：scp 命令（本地执行）**
```bash
# 上传整个数据目录
scp -r -P 端口号 ./data 用户名@服务器 IP:/path/to/Radiolaria-DINOv3/

# 示例
scp -r -P 22 ./data user@192.168.1.100:/home/user/Radiolaria-DINOv3/
```

**方法 B：rsync（支持断点续传）**
```bash
rsync -avz -e "ssh -p 端口号" ./data 用户名@服务器 IP:/path/to/Radiolaria-DINOv3/
```

**方法 C：MobaXterm 拖拽**
- 连接服务器后，直接拖拽本地文件夹到右侧文件浏览器

### 🔄 本地数据整理（Windows 用户）

如果你的数据在 Windows 上，先在本地整理好再上传：

```bash
# 在 Windows PowerShell 中运行
# 整理 D:\数据集\dataR 到当前目录的 data/ 文件夹

python scripts/organize_dataset.py \
  --data-root "D:\数据集\dataR" \
  --output-dir "./data"

# 使用符号链接（节省空间，不推荐 Windows）
python scripts/organize_dataset.py \
  --data-root "D:\数据集\dataR" \
  --output-dir "./data" \
  --symlink
```

整理完成后，会生成：
```
data/
├── images/train/0/       # 所有训练图像（自监督用）
├── splits_fixed/         # 标准化后的 CSV
│   ├── train_fixed.csv
│   ├── val_fixed.csv
│   └── test_fixed.csv
└── class_mapping.csv     # 类别映射
```

然后上传整理好的 data/ 目录到服务器。

---

<a name="step4"></a>
## 5. 预训练权重管理

### 📂 权重存放位置

所有预训练权重应放在项目根目录的 `model/` 文件夹中：

```
Radiolaria-DINOv3/
├── model/                      # 预训练权重目录
│   ├── dinov3_vits16_pretrain.pth
│   ├── dinov3_vitb16_pretrain.pth
│   └── dinov3_vitl16_pretrain.pth
├── core/
├── configs/
└── ...
```

### ⬇️ 自动下载脚本

项目提供了自动下载脚本，如果 `model/` 文件夹中没有权重，会自动下载：

```bash
# 创建 model 目录
mkdir -p model

# 运行下载脚本（自动检测并下载缺失的权重）
python scripts/download_weights.py --model-size vitb16

# 下载所有可用权重
python scripts/download_weights.py --download-all
```

**手动下载（如果自动下载失败）：**
```bash
# ViT-Small
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16_pretrain.pth -O model/dinov3_vits16_pretrain.pth

# ViT-Base（推荐，约 350MB）
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth -O model/dinov3_vitb16_pretrain.pth

# ViT-Large（约 1.2GB）
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_pretrain.pth -O model/dinov3_vitl16_pretrain.pth
```

### 🔍 查看可用权重

```bash
# 列出 model 目录中的权重
ls -lh model/

# 输出示例：
# -rw-r--r-- 1 user user 350M Mar 10 10:00 dinov3_vitb16_pretrain.pth
# -rw-r--r-- 1 user user 1.2G Mar 10 10:05 dinov3_vitl16_pretrain.pth
```

---

<a name="step5"></a>
## 6. 开始训练

### 6.1 阶段 A：基线评估

```bash
# 使用 model/ 目录中的权重
python core/baseline.py \
  --data_root ./data \
  --weights ./model/dinov3_vitb16_pretrain.pth \
  --model_size vitb16 \
  --output_dir ./runs/baseline_vitb16
```

### 6.2 阶段 B：TAPT（自监督预训练）⭐

**单 GPU：**
```bash
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./model/dinov3_vitb16_pretrain.pth
```

**多 GPU（推荐）：**
```bash
torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./model/dinov3_vitb16_pretrain.pth
```

**后台运行（防止 SSH 断开）：**
```bash
# 使用 nohup
nohup torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./model/dinov3_vitb16_pretrain.pth \
  > runs/ssl_train.log 2>&1 &

# 或使用 screen
screen -S dinov3_train
torchrun --nproc_per_node=4 core/train_ssl.py ...
# Ctrl+A, D 分离会话
# screen -r dinov3_train 重新连接
```

### 6.3 阶段 C：监督微调

```bash
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage \
  --lr 5e-4
```

### 6.4 阶段 D：小样本评估

```bash
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn --n-way 5 --k-shot 1 --episodes 600
```

---

<a name="step6"></a>
## 7. 监控进度

```bash
# 查看实时日志
tail -f runs/ssl_vitb16/log.txt

# 监控 GPU 使用率
watch -n 1 nvidia-smi

# 查看检查点
ls -lh runs/ssl_vitb16/ckpt/

# 查看训练损失
grep "Loss" runs/ssl_vitb16/log.txt | tail -20
```

---

<a name="faq"></a>
## 8. 常见问题

### Q1: 显存不足 (CUDA out of memory)
```bash
# 编辑配置文件，减小 batch size
vim configs/pretrain/vitb16_domain_adapt.yaml
# 修改：batch_size_per_gpu: 32 → 16 或 8
```

### Q2: 找不到预训练权重
```bash
# 检查 model/ 目录是否存在
ls -la model/

# 自动下载
python scripts/download_weights.py --model-size vitb16
```

### Q3: 数据格式不匹配
```bash
# 检查数据格式
ls -R data/images/

# 重新准备数据
python utils/prepare_data.py --source ./raw_data --dest ./data/images
```

### Q4: SSH 断开导致训练中断
```bash
# 使用 screen 保持会话
screen -S dinov3_train
# 运行训练命令
# Ctrl+A 然后 D 分离
screen -r dinov3_train  # 重新连接
```

### Q5: 下载权重速度慢
```bash
# 使用国内镜像（如果可用）
wget -c https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth -O model/dinov3_vitb16_pretrain.pth

# 或使用 axel 多线程下载
axel -n 10 https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth -o model/dinov3_vitb16_pretrain.pth
```

---

## 📝 快速命令参考

```bash
# ========== 连接 ==========
ssh -p 22 user@server_ip

# ========== 环境 ==========
conda activate dinov3

# ========== 权重 ==========
python scripts/download_weights.py --model-size vitb16

# ========== 数据 ==========
python utils/prepare_data.py --source ./raw_data --dest ./data/images

# ========== 训练 ==========
# 基线
python core/baseline.py --data_root ./data --weights ./model/dinov3_vitb16_pretrain.pth --model_size vitb16

# TAPT（多 GPU）
torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./model/dinov3_vitb16_pretrain.pth

# 微调
python core/train_cls.py --data_root ./data --weights ./runs/ssl_vitb16/checkpoint.pth --model_size vitb16 --mode two_stage

# 评估
python core/fewshot.py --data-path ./data --weights ./runs/ssl_vitb16/checkpoint.pth --model-size vitb16 --run-knn --n-way 5 --k-shot 1

# ========== 监控 ==========
nvidia-smi
tail -f runs/ssl_vitb16/log.txt
```

---

<div align="center">

**祝训练顺利！🚀**

遇到问题请查看 [GitHub Issues](https://github.com/syxscott/Radiolaria-DINOv3/issues)

</div>
