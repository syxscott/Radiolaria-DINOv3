import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

# 支持的图片扩展名
EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'}

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for DINOv3 (ImageFolder format)")
    parser.add_argument('--source', type=str, required=True, help="Path to the flat folder containing images")
    parser.add_argument('--dest', type=str, required=True, help="Output path (e.g., ./data/unlabeled_train)")
    parser.add_argument('--symlink', action='store_true', help="Use symlinks instead of copying (saves space)")
    return parser.parse_args()

def prepare_data(source_dir, target_dir, use_symlink=False):
    """
    针对扁平目录（无子文件夹）的无标数据进行预处理。
    将数据整理为: target_dir/train/0/*.jpg 结构，以适配 DINOv3 的 ImageFolder 读取逻辑。
    """
    print(f"正在扫描扁平目录: {source_dir} ...")

    # 目标路径: target_dir/train/0/*.jpg
    dest_folder = os.path.join(target_dir, "train", "0")

    if os.path.exists(target_dir):
        print(f"警告: 目标目录 {target_dir} 已存在，将被清理...")
        shutil.rmtree(target_dir)

    os.makedirs(dest_folder, exist_ok=True)

    files = []
    # 扫描当前目录下的文件
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(source_dir, f"*.{ext}")))
        files.extend(glob.glob(os.path.join(source_dir, f"*.{ext.upper()}")))

    files = sorted(list(set(files)))
    print(f"共找到 {len(files)} 张图片。正在处理...")

    for i, file_path in enumerate(tqdm(files)):
        file_path = Path(file_path).resolve()

        # 直接使用原始文件名 + 序号前缀 (防止重名)
        file_name = file_path.name
        new_name = f"{i:07d}_{file_name}"

        dest_path = os.path.join(dest_folder, new_name)

        if use_symlink:
            try:
                os.symlink(file_path, dest_path)
            except OSError as e:
                print(f"链接创建失败 {file_path}: {e}")
        else:
            try:
                shutil.copy2(file_path, dest_path)
            except OSError as e:
                print(f"复制失败 {file_path}: {e}")

    print(f"完成！数据已准备在: {target_dir}")
    print(f"目录结构: {target_dir}/train/0/*.images (已兼容 DINOv3 ImageFolder)")


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.source, args.dest, args.symlink)