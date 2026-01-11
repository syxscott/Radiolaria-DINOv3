import os
import sys
import logging
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.datasets import ImageFolder

# 确保项目根目录在 path 中，以便能 import dinov3
# 假设脚本是从项目根目录运行的: python core/train_ssl.py ...
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === 1. 动态修复日志报错 (Monkey Patching) ===
# DINOv3 原生日志在处理 DictConfig 时可能会报错，这里进行补丁修复
original_info = logging.Logger.info

def patched_info(self, msg, *args, **kwargs):
    if isinstance(msg, (DictConfig, dict, list)):
        try:
            if isinstance(msg, DictConfig):
                msg = OmegaConf.to_yaml(msg)
            else:
                msg = str(msg)
        except:
            msg = str(msg)
    original_info(self, msg, *args, **kwargs)

logging.Logger.info = patched_info

# === 2. 准备自定义加载函数 (Path Patching) ===
# 拦截 DINOv3 的数据集加载逻辑，使其能直接加载本地 ImageFolder
import dinov3.data.loaders as loaders

if hasattr(loaders.make_dataset, "_is_patched"):
    original_make_dataset = loaders.make_dataset._original
else:
    original_make_dataset = loaders.make_dataset

def patched_make_dataset(dataset_str, transform, target_transform=None, **kwargs):
    # 只要是存在的目录，就强制拦截，防止 DINOv3 内部解析逻辑(如ImageNet解析)报错
    # 这允许我们直接传入 "./data/unlabeled_train" 这样的路径
    if os.path.exists(dataset_str) and os.path.isdir(dataset_str):
        # 仅在第一次调用时打印，防止日志刷屏
        if not hasattr(patched_make_dataset, "has_printed"):
            print(f"[Patched Loader] -> Intercepted Directory: {dataset_str}")
            patched_make_dataset.has_printed = True
        return ImageFolder(root=dataset_str, transform=transform, target_transform=target_transform)

    return original_make_dataset(dataset_str, transform, target_transform, **kwargs)

patched_make_dataset._is_patched = True
patched_make_dataset._original = original_make_dataset
loaders.make_dataset = patched_make_dataset

import dinov3.train.train as train_module
if hasattr(train_module, "make_dataset"):
    train_module.make_dataset = patched_make_dataset

# === 3. 启动训练 ===
from dinov3.train.train import main

if __name__ == "__main__":
    # === [新增] 自动断点续训逻辑 (Auto-Resume) ===
    # 目的: 检测 output_dir 下是否存在 checkpoint，如果存在，自动注入 resume 参数
    try:
        output_dir = None
        # 1. 从命令行参数中寻找 train.output_dir
        for arg in sys.argv:
            if arg.startswith("train.output_dir="):
                output_dir = arg.split("=", 1)[1]
                break

        # 2. 如果找到了输出目录，且目录存在
        if output_dir and os.path.exists(output_dir):
            ckpt_path = os.path.join(output_dir, "checkpoint.pth")
            # 3. 检查是否存在 checkpoint.pth (这是 DINOv3 默认的保存文件名)
            if os.path.exists(ckpt_path):
                print("=" * 60)
                print(f"[Auto-Resume] 检测到现有 Checkpoint: {ckpt_path}")
                print(f"[Auto-Resume] 正在自动注入 'train.resume=true' 以继续训练...")
                print("=" * 60)

                # 4. 注入参数 (避免重复注入)
                has_resume = any(arg.startswith("train.resume=") for arg in sys.argv)
                if not has_resume:
                    sys.argv.append("train.resume=true")
            else:
                print(f"[Auto-Resume] 目录存在但无 checkpoint.pth，将作为新实验运行: {output_dir}")

    except Exception as e:
        print(f"[Auto-Resume] 自动检测逻辑出错 (不影响正常训练): {e}")

    main()