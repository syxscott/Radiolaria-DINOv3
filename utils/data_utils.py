import os
import time
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
import torch

# 防止因截断图像导致的报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_col(df, options, name):
    """
    在DataFrame中查找符合options列表的列名
    """
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(f"无法在CSV中找到{name}列。现有列名: {df.columns}")


def get_stratified_datasets(data_root, img_root, transform_train=None, transform_val=None, save_dir=None):
    """
    加载或创建分层划分的数据集 (Train/Val/Test)。
    包含防数据泄露和多进程文件锁逻辑。

    Args:
        data_root (str): 包含 train.csv, val.csv 等原始文件的根目录
        img_root (str): 图片存放的根目录
        transform_train: 训练集预处理
        transform_val: 验证/测试集预处理
        save_dir (str): 保存固定划分CSV的目录
    """
    if save_dir is None:
        save_dir = os.path.join(data_root, 'splits_fixed')

    os.makedirs(save_dir, exist_ok=True)

    fixed_train_path = os.path.join(save_dir, 'train_fixed.csv')
    fixed_val_path = os.path.join(save_dir, 'val_fixed.csv')
    fixed_test_path = os.path.join(save_dir, 'test_fixed.csv')

    # --- 1. 尝试加载已固化的数据 ---
    # 简单的文件锁逻辑：等待文件完全写入
    wait_attempts = 0
    while wait_attempts < 10:
        if os.path.exists(fixed_train_path) and os.path.exists(fixed_test_path):
            try:
                # 尝试读取，确认文件没损坏
                pd.read_csv(fixed_train_path)
                break
            except:
                pass
        time.sleep(1)
        wait_attempts += 1

    if os.path.exists(fixed_train_path) and os.path.exists(fixed_test_path):
        print(f"[DataUtils] 发现已固化的数据划分: {save_dir}，直接加载...")
        train_df = pd.read_csv(fixed_train_path)
        val_df = pd.read_csv(fixed_val_path)
        test_df = pd.read_csv(fixed_test_path)
    else:
        # --- 2. 重新划分逻辑 (仅主进程执行，外部通过 barrier 控制) ---
        print("[DataUtils] 未发现固化划分，正在合并原始 CSV 并重新进行分层划分...")
        dfs = []
        # 尝试读取原始的 CSV 文件
        found_files = False
        for fname in ['train.csv', 'val.csv', 'test.csv']:
            path = os.path.join(data_root, fname)
            if os.path.exists(path):
                found_files = True
                df = pd.read_csv(path)
                img_col = find_col(df, ['image', 'filepath', 'path', 'file', 'filename', 'id'], "图片路径")
                lbl_col = find_col(df, ['label', 'class', 'category', 'y', 'target'], "标签")

                df = df.rename(columns={img_col: 'filepath', lbl_col: 'label'})
                df = df[['filepath', 'label']]
                dfs.append(df)

        if not found_files:
            # 如果没有csv，尝试从文件夹结构读取 (ImageFolder style)
            print(f"[DataUtils] 未找到CSV文件，尝试扫描文件夹结构: {img_root}")
            data = []
            if os.path.exists(img_root):
                for root, _, files in os.walk(img_root):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                            # 假设父文件夹名是标签
                            label = os.path.basename(root)
                            # 如果父文件夹是 img_root 本身，说明是扁平结构，无法推断标签
                            if os.path.abspath(root) == os.path.abspath(img_root):
                                continue
                            data.append({'filepath': os.path.join(root, file), 'label': label})

                if len(data) > 0:
                    dfs = [pd.DataFrame(data)]
                else:
                    raise ValueError(f"在 {data_root} 未找到CSV且在 {img_root} 未扫描到结构化图片")
            else:
                raise ValueError(f"图片目录不存在: {img_root}")

        full_df = pd.concat(dfs, ignore_index=True)
        # 去重
        full_df.drop_duplicates(subset=['filepath'], inplace=True)
        labels = full_df['label'].tolist()

        # 第一次划分: 拆分出 Test (15%)
        # 注意：如果类别极少样本，stratify可能会报错，这里假设数据量足够
        try:
            train_val_df, test_df = train_test_split(
                full_df, test_size=0.15, stratify=labels, random_state=42
            )
        except ValueError:
            print("[Warning] 样本数过少，无法分层划分，切换为随机划分")
            train_val_df, test_df = train_test_split(
                full_df, test_size=0.15, random_state=42
            )

        # 第二次划分: 拆分出 Val (从剩下的里面拿，最终约 15% total)
        train_val_labels = train_val_df['label'].tolist()
        try:
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.176, stratify=train_val_labels, random_state=42
            )
        except ValueError:
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.176, random_state=42
            )

        print(f"[DataUtils] 划分完成 -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        train_df.to_csv(fixed_train_path, index=False)
        val_df.to_csv(fixed_val_path, index=False)
        test_df.to_csv(fixed_test_path, index=False)

    # --- 3. 构建映射与Dataset ---
    all_labels = sorted(
        list(set(train_df['label'].unique()) | set(val_df['label'].unique()) | set(test_df['label'].unique())))
    class_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    print(f"总类别数: {len(class_to_idx)}")

    def df_to_samples(df):
        samples = []
        for _, row in df.iterrows():
            p = str(row['filepath'])
            # 兼容绝对路径和相对路径
            if not os.path.isabs(p):
                p = os.path.join(img_root, p)
            samples.append((p, class_to_idx[row['label']]))
        return samples

    train_ds = RadiolariaDataset(df_to_samples(train_df), transform=transform_train, name="Train")
    val_ds = RadiolariaDataset(df_to_samples(val_df), transform=transform_val, name="Val")
    test_ds = RadiolariaDataset(df_to_samples(test_df), transform=transform_val, name="Test")

    return train_ds, val_ds, test_ds, class_to_idx


class RadiolariaDataset(Dataset):
    def __init__(self, samples, transform=None, name="Dataset"):
        self.samples = samples
        self.transform = transform
        self.name = name
        self.error_count = 0
        self.max_errors = 100  # 熔断阈值
        self._check_integrity()

    def _check_integrity(self):
        """随机检查前几张图，确保路径基本正确"""
        if len(self.samples) == 0: return
        indices = np.random.choice(len(self.samples), min(5, len(self.samples)), replace=False)
        for idx in indices:
            path, _ = self.samples[idx]
            if not os.path.exists(path):
                print(f"⚠️ [警告] {self.name} 数据集完整性检查失败: 找不到文件 {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, path
        except Exception as e:
            self.error_count += 1
            if self.error_count < 5:
                print(f"❌ [错误] {self.name} 无法加载图片 ({self.error_count}次): {path}, {e}")
            elif self.error_count == self.max_errors:
                raise RuntimeError(
                    f"{self.name} 数据集加载失败次数过多 (> {self.max_errors})，请检查数据路径或文件损坏情况！")

            # 返回一张全黑图作为 fallback，避免 crash (虽然有风险，但在训练中通常权重会被 loss 压低)
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, label, path


def get_transforms(img_size=224, is_train=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # 使用 BICUBIC 插值对抗低分辨率
    interpolation = InterpolationMode.BICUBIC

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            # 验证集也使用高质量放大
            transforms.Resize(int(img_size * 256 / 224), interpolation=interpolation),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])