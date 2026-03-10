# SSH 算力平台训练指南

<div align="center">

**从零开始在 Linux 算力平台上训练 DINOv3 模型**

</div>

---

## 📋 目录

1. [准备工作](#step0)
2. [SSH 连接](#step1)
3. [环境配置](#step2)
4. [数据上传](#step3)
5. [开始训练](#step4)
6. [监控进度](#step5)
7. [常见问题](#faq)

---

<a name="step0"></a>
## 1. 准备工作

| 物品 | 说明 |
|------|------|
| SSH 密钥 | `~/.ssh/id_rsa` |
| 终端软件 | MobaXterm / PowerShell / Terminal |
| 数据集 | 放射虫化石图像 |
| 预训练权重 | DINOv3 官方权重 |

生成 SSH 密钥：
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

---

<a name="step1"></a>
## 2. SSH 连接

```bash
# 基本连接
ssh -p 端口号 用户名@服务器 IP

# 示例
ssh -p 22 user@192.168.1.100
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
```

### 3.4 安装依赖
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
## 4. 数据上传

### 方法 A：scp 命令
```bash
# 本地执行（非 SSH）
scp -r -P 端口号 ./data/images 用户名@服务器 IP:/path/to/Radiolaria-DINOv3/data/
```

### 方法 B：rsync（支持断点续传）
```bash
rsync -avz -e "ssh -p 端口号" ./data/images 用户名@服务器 IP:/path/to/Radiolaria-DINOv3/data/
```

---

<a name="step4"></a>
## 5. 开始训练

### 5.1 下载预训练权重
```bash
mkdir -p weights
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth -O weights/dinov3_vitb16_pretrain.pth
```

### 5.2 阶段 A：基线评估
```bash
python core/baseline.py \
  --data_root ./data \
  --weights ./weights/dinov3_vitb16_pretrain.pth \
  --model_size vitb16 \
  --output_dir ./runs/baseline_vitb16
```

### 5.3 阶段 B：TAPT（自监督预训练）⭐

**单 GPU：**
```bash
python core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./weights/dinov3_vitb16_pretrain.pth
```

**多 GPU（推荐）：**
```bash
torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./weights/dinov3_vitb16_pretrain.pth
```

**后台运行（防止断开）：**
```bash
nohup torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16 \
  train.dataset_path=ImageFolder:root=./data/images:split=TRAIN \
  student.pretrained_weights=./weights/dinov3_vitb16_pretrain.pth \
  > runs/ssl_train.log 2>&1 &
```

### 5.4 阶段 C：监督微调
```bash
python core/train_cls.py \
  --data_root ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model_size vitb16 \
  --output_dir ./runs/sft_vitb16 \
  --mode two_stage \
  --lr 5e-4
```

### 5.5 阶段 D：小样本评估
```bash
python core/fewshot.py \
  --data-path ./data \
  --weights ./runs/ssl_vitb16/checkpoint.pth \
  --model-size vitb16 \
  --run-knn --n-way 5 --k-shot 1 --episodes 600
```

---

<a name="step5"></a>
## 6. 监控进度

```bash
# 查看日志
tail -f runs/ssl_vitb16/log.txt

# 监控 GPU
watch -n 1 nvidia-smi

# 查看检查点
ls -lh runs/ssl_vitb16/ckpt/
```

---

<a name="faq"></a>
## 7. 常见问题

### Q1: 显存不足
```bash
# 减小 batch size（编辑配置文件）
# batch_size_per_gpu: 32 → 16 或 8
```

### Q2: SSH 断开导致中断
```bash
# 使用 screen
screen -S dinov3_train
# 运行训练命令
# Ctrl+A 然后 D 分离
# 重新连接：screen -r dinov3_train
```

### Q3: 找不到 CUDA
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

---

## 📝 快速命令参考

```bash
# 连接
ssh -p 22 user@server_ip

# 环境
conda activate dinov3

# 训练（多 GPU）
torchrun --nproc_per_node=4 core/train_ssl.py \
  --config-file configs/pretrain/vitb16_domain_adapt.yaml \
  --output-dir ./runs/ssl_vitb16

# 监控
nvidia-smi
tail -f runs/ssl_vitb16/log.txt
```

---

<div align="center">

**祝训练顺利！🚀**

</div>
