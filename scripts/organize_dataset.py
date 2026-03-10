#!/usr/bin/env python3  
\"\"\"数据集整理脚本\"\"\" 
import os  
import pandas as pd  
import shutil  
from pathlib import Path  
import argparse 
  
def organize_dataset(data_root, output_dir='./data', symlink=False):  
\"    data_root = Path(data_root)\"  
\"    output_dir = Path(output_dir)\"  
\"    images_dir = data_root / 'images'\" 
\"    print('数据集整理工具')\"  
\"    print(f'输入: {data_root}')\"  
\"    print(f'输出: {output_dir}')\" 
\"    output_dir.mkdir(parents=True, exist_ok=True)\"  
\"    (output_dir / 'images' / 'train' / '0').mkdir(parents=True, exist_ok=True)\"  
\"    (output_dir / 'splits_fixed').mkdir(parents=True, exist_ok=True)\" 
\"    all_classes = set()\"  
\"    for split in ['train', 'val', 'test']:\"  
\"        csv_path = data_root / f'{split}.csv'\"  
\"        if not csv_path.exists(): continue\"  
\"        df = pd.read_csv(csv_path)\" 
\"        filepath_col = [c for c in df.columns if 'path' in c.lower() or 'file' in c.lower()][0]\"  
\"        label_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()][0]\"  
\"        df = df.rename(columns={filepath_col: 'filepath', label_col: 'label'})\"  
\"        all_classes.update(df['label'].unique())\"  
\"        df.to_csv(output_dir / 'splits_fixed' / f'{split}_fixed.csv', index=False)\"  
\"        print(f'{split}: {len(df)} samples')\" 
\"    print(f'Classes: {len(all_classes)}')\"  
\"    train_csv = data_root / 'train.csv'\"  
\"    if train_csv.exists():\"  
\"        train_df = pd.read_csv(train_csv)\"  
\"        filepath_col = [c for c in train_df.columns if 'path' in c.lower() or 'file' in c.lower()][0]\"  
\"        print(f'Copying {len(train_df)} train images...')\" 
\"        for idx, row in train_df.iterrows():\"  
\"            src = images_dir / row[filepath_col]\"  
\"            dst = output_dir / 'images' / 'train' / '0' / Path(row[filepath_col]).name\"  
\"            if src.exists():\"  
\"                if symlink: os.symlink(src, dst)\"  
\"                else: shutil.copy2(src, dst)\"  
\"            if (idx + 1) % 100 == 0: print(f'  {idx + 1}/{len(train_df)}')\" 
\"    print('Done!')\"  
\"    print(f'Output: {output_dir}')\"  
  
if __name__ == '__main__':  
\"    parser = argparse.ArgumentParser()\"  
\"    parser.add_argument('--data-root', required=True)\"  
\"    parser.add_argument('--output-dir', default='./data')\"  
\"    parser.add_argument('--symlink', action='store_true')\"  
\"    args = parser.parse_args()\"  
\"    organize_dataset(args.data_root, args.output_dir, args.symlink)\" 
