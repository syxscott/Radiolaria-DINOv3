import os
import pandas as pd

print("=" * 60)
print("数据集结构检查")
print("=" * 60)

# 检查 splits_fixed
splits_dir = 'data/splits_fixed'
if os.path.exists(splits_dir):
    print(f"\n✓ {splits_dir} 存在")
    files = os.listdir(splits_dir)
    print(f"  文件: {files}")
    for f in files:
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(splits_dir, f))
            print(f"  {f}: {len(df)} 条记录, 列: {list(df.columns)}")
            if len(df) > 0:
                print(f"    示例: {df.iloc[0].to_dict()}")
else:
    print(f"\n✗ {splits_dir} 不存在")

# 检查 images/train/0
train_dir = 'data/images/train/0'
print(f"\n检查目录: {train_dir}")
if os.path.exists(train_dir):
    files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
    print(f"✓ 目录存在，图片数量: {len(files)}")
    if files:
        print(f"  示例文件: {files[:5]}")
else:
    print(f"✗ 目录不存在")
    # 检查 train 目录
    train_parent = 'data/images/train'
    if os.path.exists(train_parent):
        subdirs = os.listdir(train_parent)
        print(f"  但 {train_parent} 存在，子目录: {subdirs}")

# 检查 images_supervised
supervised_dir = 'data/images_supervised'
if os.path.exists(supervised_dir):
    print(f"\n✓ {supervised_dir} 存在")
    subdirs = os.listdir(supervised_dir)
    print(f"  子目录: {subdirs}")
    
    # 统计每个子目录的类别数
    for subdir in ['train', 'val', 'test']:
        path = os.path.join(supervised_dir, subdir)
        if os.path.exists(path):
            classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            print(f"  {subdir}: {len(classes)} 个类别")
else:
    print(f"\n✗ {supervised_dir} 不存在")

print("\n" + "=" * 60)
