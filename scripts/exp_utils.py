#!/usr/bin/env python3
"""Common utilities for Radiolaria DINOv3 experiments.

This module is intentionally standalone and does not modify existing project code.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境: 强制非交互后端, 否则 Qt/xcb 会 core dump
# PDF/SVG 中文字保持可编辑文本 (而非曲线轮廓)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt


def save_figure(fig, out_prefix):
    """同一张图输出 png/pdf/svg 三种格式。"""
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=300, bbox_inches="tight")
    return f"{out_prefix}.png", f"{out_prefix}.pdf", f"{out_prefix}.svg"
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SEEDS_DEFAULT = [42, 123, 456, 789, 999]


@dataclass
class SampleRecord:
    path: str
    class_name: str
    label: int


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 多 seed 实验不依赖位级确定性, 放开 cudnn autotune / TF32 换取速度
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_train_transform(
    img_size: int = 224,
    randaugment_magnitude: int = 2,
    rotation_deg: int = 0,
) -> transforms.Compose:
    aug = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    if rotation_deg and rotation_deg > 0:
        # 域动机: 放射虫标本在薄片中任意取向投影, 180° 旋转是语义保持增强
        aug.append(transforms.RandomRotation(rotation_deg))
    if randaugment_magnitude > 0:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude))

    aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(aug)


def discover_dataset_roots(data_root: str) -> List[str]:
    """Support both expected and existing layouts.

    Priority:
    1) data/radiolaria_281_classes/{class_name}/*.jpg
    2) data/images_supervised/{train,val,test}/{class_name}/*.jpg
    """
    p = Path(data_root)
    direct = p / "radiolaria_281_classes"
    if direct.exists() and any(x.is_dir() for x in direct.iterdir()):
        return [str(direct)]

    sup = p / "images_supervised"
    splits = [sup / "train", sup / "val", sup / "test"]
    if all(s.exists() for s in splits):
        return [str(s) for s in splits]

    # fallback: data_root itself as class-folder root
    return [str(p)]


def _iter_class_folders(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def build_radiolaria_index(data_root: str) -> Tuple[List[SampleRecord], Dict[str, int], Dict[int, str]]:
    roots = discover_dataset_roots(data_root)

    class_names = set()
    for r in roots:
        root = Path(r)
        for cdir in _iter_class_folders(root):
            class_names.add(cdir.name)

    class_names = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    records: List[SampleRecord] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for r in roots:
        root = Path(r)
        for cdir in _iter_class_folders(root):
            label = class_to_idx[cdir.name]
            for f in sorted(cdir.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    records.append(SampleRecord(path=str(f), class_name=cdir.name, label=label))

    return records, class_to_idx, idx_to_class


class RadiolariaDataset(Dataset):
    def __init__(self, records: Sequence[SampleRecord], transform: Optional[transforms.Compose] = None):
        self.records = list(records)
        self.transform = transform
        self.error_count = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            img = Image.open(rec.path).convert("RGB")
        except Exception as exc:
            # 坏图不能静默混入训练：记录前 5 次 + 追加到错误日志，然后回退黑图
            self.error_count += 1
            if self.error_count <= 5:
                print(f"❌ [RadiolariaDataset] 第 {self.error_count} 次加载失败: {rec.path} ({exc})")
            try:
                with open("image_load_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"{rec.path}\t{exc}\n")
            except OSError:
                pass
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, rec.label, rec.path, rec.class_name


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


class MixupCutmixCollator:
    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0, prob: float = 1.0, num_classes: int = 1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    @staticmethod
    def _rand_bbox(size: Sequence[int], lam: float):
        _, _, h, w = size
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        return x1, y1, x2, y2

    def __call__(self, batch):
        imgs, labels, paths, names = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        targets = F.one_hot(labels, num_classes=self.num_classes).float()

        if np.random.rand() > self.prob:
            return imgs, targets, paths, names

        use_cutmix = np.random.rand() > 0.5 and self.cutmix_alpha > 0
        index = torch.randperm(imgs.size(0))

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = self._rand_bbox(imgs.size(), lam)
            imgs[:, :, y1:y2, x1:x2] = imgs[index, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (imgs.size(-1) * imgs.size(-2)))
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            imgs = lam * imgs + (1.0 - lam) * imgs[index]

        targets = lam * targets + (1.0 - lam) * targets[index]
        return imgs, targets, paths, names


def build_optimizer_llrd(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float = 0.75):
    real_model = model.module if hasattr(model, "module") else model
    backbone = real_model.backbone
    head = real_model.head

    num_layers = len(backbone.blocks) if hasattr(backbone, "blocks") else 12
    param_groups = {}

    def get_layer_id(name: str) -> int:
        if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
            return 0
        if "blocks" in name:
            try:
                return int(name.split(".")[1]) + 1
            except Exception:
                return 0
        return num_layers + 1

    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        lid = get_layer_id(name)
        scale = layer_decay ** (num_layers - lid + 1)
        key = f"layer_{lid}"
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": weight_decay}
        param_groups[key]["params"].append(p)

    head_params = [p for p in head.parameters() if p.requires_grad]
    final_groups = list(param_groups.values())
    if head_params:
        final_groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(final_groups)


def load_dinov3_backbone(
    repo_root: str,
    model_size: str,
    checkpoint_path: str,
    prefer_torch_hub: bool = True,
):
    """Load DINOv3 via the official local dinov3.hub API.

    Loads the official architecture and attaches weights from
    `checkpoint_path` (a local .pth file or a DINOv3 training checkpoint).
    `prefer_torch_hub` is kept for backward compatibility and ignored:
    torch.hub would need a network clone of the dinov3 repo, which the
    local API covers offline.
    """
    model_size = model_size.lower().strip()
    hub_name = {
        "small": "dinov3_vits16",
        "base": "dinov3_vitb16",
        "large": "dinov3_vitl16",
        "vits16": "dinov3_vits16",
        "vitb16": "dinov3_vitb16",
        "vitl16": "dinov3_vitl16",
        "smallplus": "dinov3_vits16plus",
        "vits16plus": "dinov3_vits16plus",
        "splus": "dinov3_vits16plus",
    }.get(model_size)
    if hub_name is None:
        raise ValueError(f"Unsupported model size: {model_size}")

    # Ensure local repo is importable when running from arbitrary working dir.
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from dinov3.hub.backbones import dinov3_vitb16, dinov3_vits16, dinov3_vits16plus, dinov3_vitl16

    # 1) Try official pretrained path first.
    try:
        if hub_name == "dinov3_vits16":
            return dinov3_vits16(pretrained=True, weights=checkpoint_path)
        if hub_name == "dinov3_vitb16":
            return dinov3_vitb16(pretrained=True, weights=checkpoint_path)
        if hub_name == "dinov3_vitl16":
            return dinov3_vitl16(pretrained=True, weights=checkpoint_path)
        if hub_name == "dinov3_vits16plus":
            return dinov3_vits16plus(pretrained=True, weights=checkpoint_path)
    except Exception:
        pass

    # 2) Robust fallback: initialize official architecture, then load local checkpoint.
    if hub_name == "dinov3_vits16":
        model = dinov3_vits16(pretrained=False)
    elif hub_name == "dinov3_vitb16":
        model = dinov3_vitb16(pretrained=False)
    elif hub_name == "dinov3_vitl16":
        model = dinov3_vitl16(pretrained=False)
    elif hub_name == "dinov3_vits16plus":
        model = dinov3_vits16plus(pretrained=False)
    else:
        raise ValueError(hub_name)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "teacher" in ckpt:
        state_dict = ckpt["teacher"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    return model


class DinoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim", 768)
        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.head(feat)


def set_train_mode(model: DinoClassifier, mode: str):
    if mode == "linear":
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif mode == "full_ft":
        model.backbone.train()
        for p in model.backbone.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    paths: List[str] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for imgs, y, p, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
            all_logits.append(logits.float().cpu().numpy())
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(y.numpy().tolist())
            paths.extend(list(p))

    scores = np.concatenate(all_logits, axis=0)
    num_classes = scores.shape[1]
    acc = float(np.mean(np.array(preds) == np.array(labels)) * 100.0)
    topk = topk_accuracy_from_scores(scores, list(range(num_classes)), labels)
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return acc, macro_f1, topk[3], topk[5], labels, preds, paths


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device):
    backbone.eval()
    feats, labels, paths = [], [], []
    for imgs, y, p, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = backbone(imgs)
        out = F.normalize(out.float(), dim=-1)
        feats.append(out.cpu().numpy())
        labels.append(y.numpy())
        paths.extend(list(p))
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels, paths


def topk_accuracy_from_scores(
    scores: np.ndarray,
    candidate_labels: Sequence[int],
    true_labels: Sequence[int],
    ks: Sequence[int] = (3, 5),
) -> Dict[int, float]:
    """Top-k accuracy from a [N, C] score matrix (similarities or logits)."""
    cand = np.asarray(candidate_labels)
    truth = np.asarray(true_labels)
    out: Dict[int, float] = {}
    for k in ks:
        k_eff = min(int(k), scores.shape[1])
        idx = np.argsort(-scores, axis=1)[:, :k_eff]
        hits = [truth[i] in cand[idx[i]] for i in range(len(truth))]
        out[int(k)] = float(np.mean(hits) * 100.0)
    return out


def run_prototype_classifier(
    support_feats: np.ndarray,
    support_labels: np.ndarray,
    query_feats: np.ndarray,
    query_labels: np.ndarray,
):
    classes = np.unique(support_labels)
    protos = []
    proto_labels = []
    for c in classes:
        idx = np.where(support_labels == c)[0]
        proto = support_feats[idx].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-6)
        protos.append(proto)
        proto_labels.append(c)

    protos = np.stack(protos, axis=0)
    sims = query_feats @ protos.T
    pred_idx = np.argmax(sims, axis=1)
    preds = np.array([proto_labels[i] for i in pred_idx])

    acc = float(np.mean(preds == query_labels) * 100.0)
    topk = topk_accuracy_from_scores(sims, np.array(proto_labels), query_labels)
    macro_f1 = float(f1_score(query_labels, preds, average="macro"))
    return acc, macro_f1, topk[3], topk[5], preds.tolist()


def run_knn_classifier(
    support_feats: np.ndarray,
    support_labels: np.ndarray,
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    k: int = 1,
):
    """Cosine kNN on L2-normalised features; majority vote among k neighbours."""
    k_eff = max(1, min(int(k), len(support_labels)))
    sims = query_feats @ support_feats.T
    neighbour_idx = np.argsort(-sims, axis=1)[:, :k_eff]
    preds = []
    for row in neighbour_idx:
        neighbour_labels = support_labels[row]
        vals, counts = np.unique(neighbour_labels, return_counts=True)
        preds.append(vals[np.argmax(counts)])
    preds = np.array(preds)
    acc = float(np.mean(preds == query_labels) * 100.0)
    macro_f1 = float(f1_score(query_labels, preds, average="macro"))
    return acc, macro_f1, preds.tolist()


def save_result_tables(df: pd.DataFrame, out_dir: str, base_name: str):
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    md_path = os.path.join(out_dir, f"{base_name}.md")
    xlsx_path = os.path.join(out_dir, f"{base_name}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    return csv_path, md_path, xlsx_path


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
    out_prefix: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, cmap="mako", cbar=True, ax=ax)
    ax.set_title("Confusion Matrix (normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    png_path, pdf_path, svg_path = save_figure(fig, out_prefix)
    plt.close(fig)
    return png_path, pdf_path


def _to_pil_from_tensor(img_t: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(3, 1, 1)
    x = (img_t * std + mean).clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


@torch.no_grad()
def build_rollout_like_attention_map(backbone: nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """Build a rollout-like map from CLS/patch token affinity.

    This is a lightweight proxy when explicit attention matrices are unavailable.
    """
    outs = backbone.forward_features(img_tensor.unsqueeze(0))
    cls = outs["x_norm_clstoken"]  # [1, C]
    patch = outs["x_norm_patchtokens"]  # [1, N, C]

    cls = F.normalize(cls, dim=-1)
    patch = F.normalize(patch, dim=-1)
    score = (patch * cls.unsqueeze(1)).sum(dim=-1).squeeze(0)  # [N]

    n = score.numel()
    side = int(math.sqrt(n))
    if side * side != n:
        side = int(math.sqrt(n // 2))
    heat = score[: side * side].reshape(side, side)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat.cpu().numpy()


def save_attention_visualization(
    backbone: nn.Module,
    image_tensor: torch.Tensor,
    image_path: str,
    out_prefix: str,
):
    heat = build_rollout_like_attention_map(backbone, image_tensor)
    pil = _to_pil_from_tensor(image_tensor)
    img_np = np.array(pil)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np)
    ax.imshow(heat, cmap="jet", alpha=0.45, extent=(0, img_np.shape[1], img_np.shape[0], 0))
    ax.set_title(Path(image_path).name)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out_prefix)
    plt.close(fig)


def select_support_query(
    records: Sequence[SampleRecord],
    num_classes: int,
    k_shot: int,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[SampleRecord]] = {i: [] for i in range(num_classes)}
    for r in records:
        by_class[r.label].append(r)

    support, query = [], []
    for c in range(num_classes):
        cls_records = by_class[c]
        if len(cls_records) <= k_shot:
            raise ValueError(f"Class {c} has {len(cls_records)} samples, cannot do k_shot={k_shot}")
        idx = np.arange(len(cls_records))
        rng.shuffle(idx)
        sidx = idx[:k_shot]
        qidx = idx[k_shot:]
        support.extend([cls_records[i] for i in sidx])
        query.extend([cls_records[i] for i in qidx])
    return support, query


def summarize_metric(values: Sequence[float]) -> str:
    arr = np.array(values, dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():.4f}"


def maybe_wrap_dataparallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def get_backbone_module(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def parse_model_size_alias(x: str) -> str:
    x = x.lower().strip()
    if x in {"small", "vits16"}:
        return "small"
    if x in {"smallplus", "vits16plus", "splus", "s+"}:
        return "smallplus"
    if x in {"base", "vitb16"}:
        return "base"
    if x in {"large", "vitl16"}:
        return "large"
    raise ValueError(f"Unsupported model_size: {x}")


def parse_fraction_list(value: str) -> List[str]:
    return [v.strip().replace("%", "") for v in value.split(",") if v.strip()]
