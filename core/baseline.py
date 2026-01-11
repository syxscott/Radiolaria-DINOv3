import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import warnings
import sys

# 过滤警告
warnings.filterwarnings("ignore")

# 路径修复: 确保能找到根目录下的 dinov3 和 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 引入 get_stratified_datasets
from utils.data_utils import get_stratified_datasets, get_transforms
from dinov3.models.vision_transformer import vit_base, vit_large, vit_small, vit_giant2


def load_backbone(model_size, weights_path, device):
    print(f"正在初始化 {model_size} 并加载权重: {weights_path}")

    if model_size == 'vitl16':
        model = vit_large(patch_size=16, num_classes=0)
    elif model_size == 'vitb16':
        model = vit_base(patch_size=16, num_classes=0)
    elif model_size == 'vits16':
        model = vit_small(patch_size=16, num_classes=0)
    elif model_size == 'vitg14':
        model = vit_giant2(patch_size=14, num_classes=0)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # 清理 key
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"权重加载状态: {msg}")
    else:
        print(f"警告: 未找到权重文件 {weights_path}，使用随机初始化 (仅用于调试)")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    features = []
    labels = []
    for img, label, _ in tqdm(loader, desc="特征提取中"):
        img = img.to(device)
        out = model(img)
        features.append(out.cpu().numpy())
        labels.append(label.numpy() if isinstance(label, torch.Tensor) else label)
    if len(features) == 0:
        return np.array([]), np.array([])
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    # L2 Normalize
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    return features, labels


def evaluate_prototypes(train_feats, train_labels, test_feats, test_labels, idx_to_class):
    print("正在运行原型分类器 (Prototype Classifier)...")
    unique_classes = np.unique(train_labels)
    prototypes = []
    proto_labels = []

    for c in unique_classes:
        indices = np.where(train_labels == c)[0]
        if len(indices) == 0: continue
        class_mean = np.mean(train_feats[indices], axis=0)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-6)
        prototypes.append(class_mean)
        proto_labels.append(c)

    prototypes = np.stack(prototypes)
    sims = np.dot(test_feats, prototypes.T)
    preds = np.argmax(sims, axis=1)
    pred_labels = [proto_labels[p] for p in preds]

    acc = np.mean(np.array(pred_labels) == test_labels)
    f1 = f1_score(test_labels, pred_labels, average='macro')
    print(f"原型分类结果 -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    return pred_labels


def evaluate_knn(train_feats, train_labels, test_feats, test_labels, k=5):
    print(f"正在运行 k-NN (k={k})...")
    sims = np.dot(test_feats, train_feats.T)
    topk_indices = np.argsort(-sims, axis=1)[:, :k]
    preds = []
    for i in range(len(test_feats)):
        neighbor_labels = train_labels[topk_indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        preds.append(unique[np.argmax(counts)])
    acc = np.mean(np.array(preds) == test_labels)
    print(f"k-NN (k={k}) 结果 -> Accuracy: {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='数据集根目录 (包含CSV)')
    parser.add_argument('--img_root', default=None, help='图片目录 (默认在 data_root/images)')
    parser.add_argument('--weights', required=True, help='预训练权重路径')
    parser.add_argument('--model_size', default='vitl16', choices=['vits16', 'vitb16', 'vitl16'])
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', default='runs/baseline_evaluation', help='结果输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_root = args.img_root if args.img_root else os.path.join(args.data_root, 'images')

    # === 使用重分层划分的函数读取数据 ===
    tfm_val = get_transforms(args.img_size, is_train=False)

    train_ds, val_ds, test_ds, class_to_idx = get_stratified_datasets(
        args.data_root, img_root, transform_train=tfm_val, transform_val=tfm_val
    )

    # 保存映射供参考
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pd.DataFrame(list(class_to_idx.items()), columns=['class', 'idx']).to_csv(
        os.path.join(args.output_dir, 'class_mapping.csv'), index=False
    )

    # 2. 提取特征
    model = load_backbone(args.model_size, args.weights, device)

    print("正在提取训练集特征...")
    train_feats, train_labels = extract_features(model, DataLoader(train_ds, batch_size=args.batch_size, num_workers=4),
                                                 device)
    print("正在提取测试集特征...")
    test_feats, test_labels = extract_features(model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=4),
                                               device)

    np.save(os.path.join(args.output_dir, 'train_feats.npy'), train_feats)
    np.save(os.path.join(args.output_dir, 'test_feats.npy'), test_feats)

    # 3. 评估
    evaluate_prototypes(train_feats, train_labels, test_feats, test_labels, idx_to_class)
    evaluate_knn(train_feats, train_labels, test_feats, test_labels, k=1)
    evaluate_knn(train_feats, train_labels, test_feats, test_labels, k=5)

    print("正在运行线性分类器 (sklearn)...")
    clf = LogisticRegression(random_state=0, C=1.0, solver='lbfgs', max_iter=2000, multi_class='multinomial')
    clf.fit(train_feats, train_labels)
    lr_preds = clf.predict(test_feats)
    lr_acc = np.mean(lr_preds == test_labels)
    lr_f1 = f1_score(test_labels, lr_preds, average='macro')
    print(f"线性探测结果 -> Accuracy: {lr_acc:.4f}, Macro-F1: {lr_f1:.4f}")

    cm = confusion_matrix(test_labels, lr_preds)
    pd.DataFrame(cm).to_csv(os.path.join(args.output_dir, 'confusion_matrix_linear.csv'))


if __name__ == '__main__':
    main()