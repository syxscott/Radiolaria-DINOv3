import os
import sys
import argparse
import random
import logging
import warnings
import gc
import shutil
import time
import json
import math
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 抑制干扰日志
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.distributed")
warnings.filterwarnings("ignore", category=FutureWarning)

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import DINOv3 models directly
from dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# Import Utils
try:
    from utils.data_utils import get_transforms as get_advanced_transforms
except ImportError:
    get_advanced_transforms = None


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_logger(output_dir, rank):
    logger = logging.getLogger("dinov3_fewshot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode='a')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return gpu, rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def save_checkpoint_atomic(state, is_best, filename, output_dir):
    if not is_main_process(): return
    filepath = os.path.join(output_dir, filename)
    tmp_path = filepath + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, filepath)
        if is_best:
            best_path = os.path.join(output_dir, filename.replace('latest', 'best'))
            shutil.copyfile(filepath, best_path)
    except Exception as e:
        print(f"[Error] Save failed: {e}")


class AverageMeter(object):
    def __init__(self): self.reset()

    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def accuracy_count(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        correct = pred.eq(target.view(-1, 1)).sum().item()
        return correct


def gather_all_tensors(tensor):
    """Robust gather that handles uneven batch sizes across ranks."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return tensor

    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max([x.item() for x in size_list])

    if local_size.item() < max_size:
        pad_size = (max_size - local_size.item(),) + tensor.shape[1:]
        padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat((tensor, padding), dim=0)
    else:
        tensor_padded = tensor

    gathered_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered_list, tensor_padded)

    output_tensors = []
    for i, t in enumerate(gathered_list):
        output_tensors.append(t[:size_list[i].item()])

    return torch.cat(output_tensors, dim=0)


def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    if n <= 1: return 0.0, 0.0
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


# -----------------------------------------------------------------------------
# Dataset & Sampler
# -----------------------------------------------------------------------------

class MiniImageNetDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        try:
            df = pd.read_csv(csv_path)
            # 兼容列名
            if 'filepath' in df.columns:
                img_col = 'filepath'
            elif 'filename' in df.columns:
                img_col = 'filename'
            else:
                img_col = df.columns[0]

            if 'label' in df.columns:
                lbl_col = 'label'
            else:
                lbl_col = df.columns[1]

            self.img_names = df[img_col].values
            self.label_names = df[lbl_col].values

        except Exception as e:
            raise RuntimeError(f"CSV read failed: {csv_path} - {e}")

        self.classes = sorted(list(set(self.label_names)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[l] for l in self.label_names]

        if is_main_process():
            print(f"Loaded {os.path.basename(csv_path)}: {len(self.img_names)} imgs, {len(self.classes)} cls.")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = str(self.img_names[idx])
        if not os.path.isabs(path):
            path = os.path.join(self.img_root, path)

        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]


class EpisodicBatchSampler(Sampler):
    def __init__(self, labels, n_way, k_shot, q_query, episodes):
        self.episodes = episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.indices_per_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.epoch = 0

        min_s = min([len(self.indices_per_class[c]) for c in self.classes])
        if min_s < k_shot + q_query:
            if is_main_process(): print(f"[Warning] Min samples {min_s} < req {k_shot + q_query}. Using replacement.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        seed = int(self.epoch * 1000)
        rng = np.random.default_rng(seed)

        for _ in range(self.episodes):
            batch = []
            selected_cls = rng.choice(self.classes, self.n_way, replace=False)
            support, query = [], []
            for c in selected_cls:
                indices = self.indices_per_class[c]
                replace = len(indices) < (self.k_shot + self.q_query)
                selected = rng.choice(indices, self.k_shot + self.q_query, replace=replace)
                support.extend(selected[:self.k_shot])
                query.extend(selected[self.k_shot:])
            yield np.concatenate([support, query])

    def __len__(self):
        return self.episodes


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class DINOv3Wrapper(nn.Module):
    def __init__(self, model_size='vitl16', weights_path=None, feature_mode='cls'):
        super().__init__()
        self.backbone = self._load_backbone_dynamic(model_size, weights_path)
        if hasattr(self.backbone, 'embed_dim'):
            self.embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            self.embed_dim = self.backbone.num_features
        else:
            self.embed_dim = 1024  # default for Large
        self.feature_mode = feature_mode

    def _load_backbone_dynamic(self, model_size, weights_path):
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
            if is_main_process(): print(f"Loading weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu')
            state = ckpt.get('model', ckpt.get('teacher', ckpt))
            # Clean keys
            state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

            # Simple pos embed interpolation if needed
            if 'pos_embed' in state and state['pos_embed'].shape != model.pos_embed.shape:
                pass  # For now, let strict=False handle or use the interpolate func from train_cls.py

            msg = model.load_state_dict(state, strict=False)
            if is_main_process(): print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")
        else:
            if is_main_process(): print("Using random initialization (or ImageNet if not provided)")

        return model

    def forward(self, x):
        outs = self.backbone.forward_features(x)
        if self.feature_mode == 'cls':
            return outs['x_norm_clstoken']
        elif self.feature_mode == 'mean':
            return outs['x_norm_patchtokens'].mean(dim=1)
        elif self.feature_mode == 'concat':
            return torch.cat([outs['x_norm_clstoken'], outs['x_norm_patchtokens'].mean(dim=1)], dim=1)
        return outs['x_norm_clstoken']


class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        input_dim = backbone.embed_dim * (2 if backbone.feature_mode == 'concat' else 1)
        self.head = nn.Linear(input_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def extract_features(self, x):
        return self.backbone(x)


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, n_way, k_shot, q_query):
        features = self.backbone(x)
        features = F.normalize(features, dim=1)
        features = features.view(n_way, k_shot + q_query, -1)
        support = features[:, :k_shot, :]
        query = features[:, k_shot:, :]
        prototypes = support.mean(dim=1)
        query_flat = query.contiguous().view(n_way * q_query, -1)
        dists = torch.cdist(query_flat, prototypes)
        return -dists

    def extract_features(self, x):
        return self.backbone(x)


# -----------------------------------------------------------------------------
# Engines
# -----------------------------------------------------------------------------

def apply_tta(model, imgs, use_tta=False):
    if not use_tta:
        return F.normalize(model(imgs), dim=1)
    f1 = F.normalize(model(imgs), dim=1)
    f2 = F.normalize(model(torch.flip(imgs, [3])), dim=1)
    return F.normalize((f1 + f2) / 2, dim=1)


def extract_features_distributed(model, loader, device, use_tta=False):
    real_model = unwrap_model(model)
    extract_fn = getattr(real_model, 'extract_features', real_model)
    model.eval()
    local_feats, local_labels = [], []
    iterator = loader
    if is_main_process(): iterator = tqdm(loader, desc="Feat Extract", unit="bt")

    with torch.no_grad():
        for imgs, lbls in iterator:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                feats = apply_tta(extract_fn, imgs, use_tta)
            local_feats.append(feats.float())
            local_labels.append(lbls)

    local_feats = torch.cat(local_feats)
    local_labels = torch.cat(local_labels)
    all_feats = gather_all_tensors(local_feats)
    all_labels = gather_all_tensors(local_labels)
    return all_feats, all_labels


def evaluate_knn_distributed(model, test_loader, device, n_way=5, n_shot=1, episodes=600, use_tta=False):
    all_feats, all_labels = extract_features_distributed(model, test_loader, device, use_tta)
    dist.barrier()
    all_feats = all_feats.to(device)
    all_labels = all_labels.to(device)

    world_size = get_world_size()
    rank = get_rank()
    eps_per_rank = math.ceil(episodes / world_size)
    rng = random.Random(42 + rank)

    accs = []
    classes = torch.unique(all_labels)
    cls_idx = {c.item(): torch.nonzero(all_labels == c, as_tuple=True)[0] for c in classes}
    valid_classes = [c for c, idx in cls_idx.items() if len(idx) >= n_shot + 15]

    if len(valid_classes) < n_way: return 0.0, 0.0

    pbar = range(eps_per_rank)
    if is_main_process(): pbar = tqdm(pbar, desc=f"KNN {n_way}w{n_shot}s", leave=False)

    for _ in pbar:
        c_sampled = torch.tensor(rng.sample(valid_classes, n_way), device=device)
        S_list, Q_list = [], []
        valid_ep = True
        for c in c_sampled:
            indices = cls_idx[c.item()]
            if len(indices) < n_shot + 15: valid_ep = False; break
            perm = torch.randperm(len(indices))
            S_list.append(indices[perm[:n_shot]])
            Q_list.append(indices[perm[n_shot:n_shot + 15]])

        if not valid_ep: continue
        S_idx = torch.cat(S_list)
        Q_idx = torch.cat(Q_list)
        Sx = all_feats[S_idx]
        Qx = all_feats[Q_idx]

        dists = torch.cdist(Qx, Sx)
        _, min_indices = dists.min(dim=1)
        pred_labels = torch.div(min_indices, n_shot, rounding_mode='floor')
        true_labels = torch.arange(n_way, device=device).repeat_interleave(15)
        acc = (pred_labels == true_labels).float().mean().item() * 100
        accs.append(acc)

    local_accs = torch.tensor(accs, device=device)
    all_accs_t = gather_all_tensors(local_accs)

    if is_main_process():
        final_accs = all_accs_t.cpu().numpy()[:episodes]
        return mean_confidence_interval(final_accs)
    return 0.0, 0.0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help="Data root containing CSVs")
    parser.add_argument('--img-root', type=str, default=None, help="Image root")
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--weights', type=str, default=None, help="Path to checkpoint")
    parser.add_argument('--model-size', type=str, default='vitl16', choices=['vits16', 'vitb16', 'vitl16'])
    parser.add_argument('--feature-mode', type=str, default='cls')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--run-knn', action='store_true')
    parser.add_argument('--n-way', type=int, default=5)
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-tta', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    gpu, rank, world_size = setup_distributed()
    logger = setup_logger(args.output_dir, rank)
    set_seed(args.seed)
    device = torch.device(gpu)

    if rank == 0: logger.info(f"Args: {args}")

    splits_dir = os.path.join(args.data_path, 'splits_fixed')
    if os.path.exists(splits_dir):
        test_csv = os.path.join(splits_dir, 'test_fixed.csv')
    else:
        test_csv = os.path.join(args.data_path, 'test.csv')

    img_root = args.img_root if args.img_root else os.path.join(args.data_path, 'images')

    if get_advanced_transforms:
        test_tf = get_advanced_transforms(224, False)
    else:
        test_tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_set = MiniImageNetDataset(test_csv, img_root, test_tf)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers,
                             pin_memory=True)

    if args.run_knn:
        if rank == 0: logging.info("=== Phase: KNN Evaluation ===")
        backbone = DINOv3Wrapper(args.model_size, args.weights, args.feature_mode).to(device)
        backbone = DDP(backbone, device_ids=[gpu])

        acc1, c1 = evaluate_knn_distributed(backbone, test_loader, device, n_way=args.n_way, n_shot=1,
                                            episodes=args.episodes, use_tta=args.use_tta)
        acc5, c5 = evaluate_knn_distributed(backbone, test_loader, device, n_way=args.n_way, n_shot=5,
                                            episodes=args.episodes, use_tta=args.use_tta)

        if rank == 0:
            logging.info(f"[KNN] 1-shot: {acc1:.2f} +/- {c1:.2f}")
            logging.info(f"[KNN] 5-shot: {acc5:.2f} +/- {c5:.2f}")

    cleanup_distributed()


if __name__ == '__main__':
    main()