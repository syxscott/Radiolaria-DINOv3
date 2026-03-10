#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集整理脚本 - Organize Dataset Script"""

import os
import pandas as pd
import shutil
from pathlib import Path
import argparse


def organize_dataset(data_root, output_dir='./data', symlink=False):
    """整理数据集为项目所需格式"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    images_dir = data_root / 'images'

    print('=' * 60)
    print('数据集整理工具 - Organize Dataset Tool')
    print('=' * 60)
    print(f'输入目录: {data_root}')
    print(f'输出目录: {output_dir}')
    print(f'使用符号链接: {symlink}')
    print('=' * 60)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建所有必要的目录
    # 对于自监督训练 (TAPT)，所有图像都放在 images/train/0/
    (output_dir / 'images' / 'train' / '0').mkdir(parents=True, exist_ok=True)

    # 对于监督训练，创建按类别分组的目录结构
    (output_dir / 'images_supervised' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images_supervised' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images_supervised' / 'test').mkdir(parents=True, exist_ok=True)

    (output_dir / 'splits_fixed').mkdir(parents=True, exist_ok=True)

    # 读取所有 CSV
    all_classes = set()
    splits = {}

    for split in ['train', 'val', 'test']:
        csv_path = data_root / f'{split}.csv'
        if not csv_path.exists():
            print(f'跳过 {split}.csv (不存在)')
            continue

        df = pd.read_csv(csv_path)

        # 自动检测列名
        filepath_col = None
        label_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'path' in col_lower or 'file' in col_lower or 'image' in col_lower:
                filepath_col = col
            if 'label' in col_lower or 'class' in col_lower or 'category' in col_lower:
                label_col = col

        if not filepath_col:
            filepath_col = df.columns[0]
        if not label_col:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        df = df.rename(columns={filepath_col: 'filepath', label_col: 'label'})
        all_classes.update(df['label'].unique())
        splits[split] = df

        # 保存标准化的 CSV
        df.to_csv(output_dir / 'splits_fixed' / f'{split}_fixed.csv', index=False)
        print(f'{split}: {len(df)} samples, {df["label"].nunique()} classes')

    print(f'\n总类别数: {len(all_classes)}')

    # 创建类别到索引的映射
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}

    # 保存类别映射
    mapping_df = pd.DataFrame([
        {'class_name': cls, 'class_idx': idx}
        for cls, idx in class_to_idx.items()
    ])
    mapping_df.to_csv(output_dir / 'class_mapping.csv', index=False)

    # 复制所有图像到相应目录
    for split, df in splits.items():
        print(f'\n正在复制 {split} 集图像...')
        
        # 复制到监督训练目录 (按类别分组)
        for idx, row in df.iterrows():
            src = images_dir / row['filepath']
            class_name = row['label']

            # 如果 filepath 包含子目录，只取文件名
            dst_filename = Path(row['filepath']).name
            dst = output_dir / 'images_supervised' / split / class_name / dst_filename

            # 创建类别目录
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.exists():
                if symlink:
                    try:
                        os.symlink(str(src.resolve()), str(dst))
                    except:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                # 如果原始路径不存在，尝试直接在 images_dir 中查找文件名
                src_by_name = images_dir / dst_filename
                if src_by_name.exists():
                    if symlink:
                        try:
                            os.symlink(str(src_by_name.resolve()), str(dst))
                        except:
                            shutil.copy2(src_by_name, dst)
                    else:
                        shutil.copy2(src_by_name, dst)
                else:
                    print(f"  ⚠️  找不到图像: {src} 或 {src_by_name}")
        
        print(f'  ✓ {split} 集: {len(df)} 张图像')
    
    # 复制训练集图像到自监督训练目录 (所有图像放在一个文件夹)
    if 'train' in splits:
        train_df = splits['train']
        print(f'\n正在复制训练集图像到自监督目录...')
        for idx, row in train_df.iterrows():
            src = images_dir / row['filepath']

            # 如果 filepath 包含子目录，只取文件名
            dst_filename = Path(row['filepath']).name
            dst = output_dir / 'images' / 'train' / '0' / dst_filename

            if src.exists():
                if symlink:
                    try:
                        os.symlink(str(src.resolve()), str(dst))
                    except:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                # 如果原始路径不存在，尝试直接在 images_dir 中查找文件名
                src_by_name = images_dir / dst_filename
                if src_by_name.exists():
                    if symlink:
                        try:
                            os.symlink(str(src_by_name.resolve()), str(dst))
                        except:
                            shutil.copy2(src_by_name, dst)
                    else:
                        shutil.copy2(src_by_name, dst)
                else:
                    print(f"  ⚠️  找不到图像: {src} 或 {src_by_name}")

            if (idx + 1) % 100 == 0:
                print(f'  自监督训练集进度: {idx + 1}/{len(train_df)}')
        
        print(f'  ✓ 自监督训练集: {len(train_df)} 张图像')

    print('\n' + '=' * 60)
    print('整理完成!')
    print('=' * 60)
    print(f'输出目录: {output_dir}')
    print(f'  ├── images/                    # 自监督训练用')
    print(f'  │   └── train/0/              # 训练图像 ({len(splits.get("train", []))} 张)')
    print(f'  ├── images_supervised/         # 监督训练用')
    print(f'  │   ├── train/                # 按类别分组')
    print(f'  │   ├── val/                  # 按类别分组')
    print(f'  │   └── test/                 # 按类别分组')
    print(f'  ├── splits_fixed/             # 标准化 CSV')
    print(f'  │   ├── train_fixed.csv')
    print(f'  │   ├── val_fixed.csv')
    print(f'  │   └── test_fixed.csv')
    print(f'  └── class_mapping.csv')
    print('\n下一步: 上传 data/ 目录到服务器')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='整理数据集为项目所需格式')
    parser.add_argument('--data-root', required=True, help='原始数据目录')
    parser.add_argument('--output-dir', default='./data', help='输出目录')
    parser.add_argument('--symlink', action='store_true', help='使用符号链接')

    args = parser.parse_args()
    organize_dataset(args.data_root, args.output_dir, args.symlink)
