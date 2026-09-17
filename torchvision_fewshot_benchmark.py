#!/usr/bin/env python3
"""Benchmark 10 official torchvision backbones under the existing few-shot protocol.

This script reuses the project's shared-class episodic few-shot evaluation setting:
for each class, K support images are sampled and the remaining images serve as query
samples. It reports Prototype / Frozen Linear Probe / Two-stage Fine-tuning results,
and optionally compares the best torchvision runs with an existing DINOv3 summary file.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_utils import (
    MixupCutmixCollator,
    RadiolariaDataset,
    SEEDS_DEFAULT,
    SoftTargetCrossEntropy,
    build_optimizer_llrd,
    build_radiolaria_index,
    ensure_dir,
    evaluate_classifier,
    extract_features,
    get_backbone_module,
    get_device,
    get_eval_transform,
    get_train_transform,
    maybe_wrap_dataparallel,
    plot_confusion_matrix,
    run_prototype_classifier,
    save_result_tables,
    select_support_query,
    set_seed,
    set_train_mode,
    setup_logger,
)


DEFAULT_MODELS = [
    "resnet18",
    "resnet50",
    "resnet101",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
]


def _save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _infer_backbone_embed_dim(model: nn.Module, image_size: int = 224) -> int:
    """Infer output feature dimension after replacing classification head."""
    training_state = model.training
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size)
        out = model(x)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        dim = int(out.shape[1])
    model.train(training_state)
    return dim


def _maybe_wrap_dataparallel_compat(model: nn.Module, device: torch.device) -> nn.Module:
    """Compatibility wrapper for older/newer exp_utils.maybe_wrap_dataparallel signatures."""
    try:
        sig = inspect.signature(maybe_wrap_dataparallel)
        if "preferred_device" in sig.parameters:
            return maybe_wrap_dataparallel(model, preferred_device=device)
        return maybe_wrap_dataparallel(model)
    except TypeError:
        return maybe_wrap_dataparallel(model)


def _records_to_df(records) -> pd.DataFrame:
    return pd.DataFrame(
        [{"path": r.path, "class_name": r.class_name, "label": int(r.label)} for r in records]
    )


def _save_prediction_table(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    paths: Sequence[str],
    idx_to_class: Dict[int, str],
    output_csv: str,
):
    rows = []
    for t, p, path in zip(y_true, y_pred, paths):
        rows.append(
            {
                "path": path,
                "true_label": int(t),
                "pred_label": int(p),
                "true_class": idx_to_class.get(int(t), str(int(t))),
                "pred_class": idx_to_class.get(int(p), str(int(p))),
                "correct": int(t) == int(p),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Torchvision few-shot backbone benchmark")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_root", type=str, default="./results")
    parser.add_argument("--experiment_name", type=str, default="torchvision_fewshot_benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Torchvision model names")
    parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs_linear", type=int, default=10)
    parser.add_argument("--epochs_full_ft", type=int, default=20)
    parser.add_argument("--lr_linear", type=float, default=5e-4)
    parser.add_argument("--lr_full_ft", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mix_prob", type=float, default=1.0)
    parser.add_argument("--randaugment_magnitude", type=int, default=2)
    parser.add_argument("--lp_c", type=float, default=1.0)
    parser.add_argument("--lp_max_iter", type=int, default=2000)
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--no_skip_completed", action="store_false", dest="skip_completed")
    parser.add_argument(
        "--dinov3_summary",
        type=str,
        default=str(PROJECT_ROOT / "results/few_shot_classifier/few_shot_summary.xlsx"),
        help="Existing DINOv3 summary xlsx for best-vs-best comparison",
    )
    parser.add_argument("--save_confusion", action="store_true", default=True, help="Save confusion matrices")
    parser.add_argument("--no_save_confusion", action="store_false", dest="save_confusion")
    parser.add_argument("--save_predictions", action="store_true", default=True, help="Save LP/FT prediction tables")
    parser.add_argument("--no_save_predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--save_features", action="store_true", default=True, help="Save support/query features")
    parser.add_argument("--no_save_features", action="store_false", dest="save_features")
    parser.add_argument(
        "--save_episode_metadata",
        action="store_true",
        default=True,
        help="Save per-seed support/query metadata",
    )
    parser.add_argument("--no_save_episode_metadata", action="store_false", dest="save_episode_metadata")
    parser.add_argument(
        "--resume_training",
        action="store_true",
        default=True,
        help="Resume interrupted two-stage training from per-seed checkpoint",
    )
    parser.add_argument("--no_resume_training", action="store_false", dest="resume_training")
    parser.add_argument("--save_checkpoints", action="store_true", default=True, help="Save per-epoch checkpoints")
    parser.add_argument("--no_save_checkpoints", action="store_false", dest="save_checkpoints")
    parser.add_argument("--checkpoint_every_epoch", type=int, default=10,
                        help="Save resume checkpoint every N epochs (I/O heavy; 10 keeps writes ~3x per run)")
    return parser.parse_args()


def _load_torchvision_model(model_name: str) -> Tuple[nn.Module, int]:
    name = model_name.lower().strip()

    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif name == "resnet101":
        model = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif name == "densenet121":
        model = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif name == "efficientnet_b3":
        model = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif name == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif name == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[-1] = nn.Identity()
    elif name == "swin_t":
        model = tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT)
        model.head = nn.Identity()
    elif name == "vit_b_16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unsupported torchvision backbone: {model_name}")

    # Infer dim after head replacement to avoid model-specific mismatches.
    embed_dim = _infer_backbone_embed_dim(model)
    return model, int(embed_dim)


class TorchvisionBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model, self.embed_dim = _load_torchvision_model(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        return out


class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim", None)
        if self.embed_dim is None:
            raise ValueError("Backbone is missing embed_dim")
        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.head(feat)


def _is_transformer_like(backbone: nn.Module) -> bool:
    return any(hasattr(backbone, attr) for attr in ["encoder", "heads", "blocks", "pos_embed"])


def _build_optimizer_for_model(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float):
    raw = get_backbone_module(model)
    backbone = raw.backbone
    if _is_transformer_like(backbone) and hasattr(backbone, "blocks"):
        return build_optimizer_llrd(model, base_lr, weight_decay, layer_decay)
    return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


def _format_metric(values: Sequence[float]) -> str:
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return f"{arr.mean():.4f}"
    return f"{arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"


def train_two_stage(
    backbone: nn.Module,
    num_classes: int,
    support_records,
    query_records,
    args,
    device: torch.device,
    run_dir: str,
    logger,
):
    train_ds = RadiolariaDataset(
        support_records,
        transform=get_train_transform(args.img_size, args.randaugment_magnitude),
    )
    eval_ds = RadiolariaDataset(
        query_records,
        transform=get_eval_transform(args.img_size),
    )

    mix_collator = MixupCutmixCollator(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mix_prob,
        num_classes=num_classes,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=mix_collator,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=min(args.eval_batch_size, len(eval_ds)),
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = BackboneClassifier(backbone=copy.deepcopy(backbone), num_classes=num_classes).to(device)
    model = _maybe_wrap_dataparallel_compat(model, device)
    criterion = SoftTargetCrossEntropy()
    model_raw = get_backbone_module(model)

    ckpt_latest = os.path.join(run_dir, "train_state_latest.pt")
    history_path = os.path.join(run_dir, "train_history.json")
    history = []

    resume_state = None
    if args.resume_training and os.path.exists(ckpt_latest):
        resume_state = torch.load(ckpt_latest, map_location=device)
        model_raw.load_state_dict(resume_state["model"])
        history = resume_state.get("history", [])
        logger.info(
            "Resuming seed training from %s (stage=%s, next_epoch=%s)",
            ckpt_latest,
            resume_state.get("stage", "linear"),
            resume_state.get("epoch", 0),
        )

    set_train_mode(get_backbone_module(model), "linear")
    optimizer = torch.optim.AdamW(
        get_backbone_module(model).head.parameters(),
        lr=args.lr_linear,
        weight_decay=args.weight_decay,
    )

    linear_start_epoch = 0
    if resume_state is not None and resume_state.get("stage") == "linear":
        linear_start_epoch = int(resume_state.get("epoch", 0))
        optimizer.load_state_dict(resume_state["optimizer"])
    elif resume_state is not None and resume_state.get("stage") == "full_ft":
        # Checkpoint was taken during full fine-tuning, so the linear stage is done.
        linear_start_epoch = args.epochs_linear

    for epoch in range(linear_start_epoch, args.epochs_linear):
        model.train()
        get_backbone_module(model).backbone.eval()  # 冻结 backbone 防止 BN 统计量漂移
        epoch_losses = []
        for imgs, targets, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))

        history.append(
            {
                "stage": "linear",
                "epoch": int(epoch + 1),
                "loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        _save_json(history_path, {"history": history})
        if args.save_checkpoints and (epoch + 1) % max(1, args.checkpoint_every_epoch) == 0:
            torch.save(
                {
                    "stage": "linear",
                    "epoch": int(epoch + 1),
                    "model": model_raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_latest,
            )

    set_train_mode(get_backbone_module(model), "full_ft")
    optimizer = _build_optimizer_for_model(model, args.lr_full_ft, args.weight_decay, args.layer_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_full_ft))

    full_start_epoch = 0
    if resume_state is not None and resume_state.get("stage") == "full_ft":
        full_start_epoch = int(resume_state.get("epoch", 0))
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    for epoch in range(full_start_epoch, args.epochs_full_ft):
        model.train()
        epoch_losses = []
        for imgs, targets, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
        scheduler.step()

        history.append(
            {
                "stage": "full_ft",
                "epoch": int(epoch + 1),
                "loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        _save_json(history_path, {"history": history})
        if args.save_checkpoints and (epoch + 1) % max(1, args.checkpoint_every_epoch) == 0:
            torch.save(
                {
                    "stage": "full_ft",
                    "epoch": int(epoch + 1),
                    "model": model_raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                ckpt_latest,
            )

    acc, macro_f1, top3, top5, y_true, y_pred, paths = evaluate_classifier(model, eval_loader, device)
    if args.save_checkpoints:
        torch.save({"model": model_raw.state_dict(), "history": history}, os.path.join(run_dir, "final_model.pt"))
    _save_json(history_path, {"history": history, "done": True})
    return acc, macro_f1, top3, top5, y_true, y_pred, paths


def _parse_first_float(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    if not m:
        raise ValueError(f"Could not parse metric from: {value}")
    return float(m.group(1))


def _parse_fraction_to_int(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    m = re.search(r"([0-9]+)", str(value))
    if m:
        return int(m.group(1))
    return -1


def build_dinov3_comparison(torchvision_summary: pd.DataFrame, dinov3_summary_path: str) -> pd.DataFrame:
    if not dinov3_summary_path or not os.path.exists(dinov3_summary_path):
        return pd.DataFrame()

    dinov3_df = pd.read_excel(dinov3_summary_path)
    tv_df = torchvision_summary.copy()

    metric_cols = [
        "proto_acc",
        "proto_macro_f1",
        "lp_acc",
        "lp_macro_f1",
        "ft_acc",
        "ft_macro_f1",
    ]
    for metric in metric_cols:
        dinov3_df[metric + "_mean"] = dinov3_df[metric].apply(_parse_first_float)
        tv_df[metric + "_mean"] = tv_df[metric].apply(_parse_first_float)

    rows = []
    for k_shot in sorted(tv_df["k_shot"].unique()):
        row = {"k_shot": int(k_shot)}
        tv_k = tv_df[tv_df["k_shot"] == k_shot]
        dino_k = dinov3_df[dinov3_df["k_shot"] == k_shot]
        if tv_k.empty or dino_k.empty:
            continue

        for metric in metric_cols:
            best_tv = tv_k.loc[tv_k[metric + "_mean"].idxmax()]
            best_dino = dino_k.loc[dino_k[metric + "_mean"].idxmax()]

            row[f"torchvision_best_{metric}"] = best_tv[metric]
            row[f"torchvision_best_{metric}_model"] = best_tv["model_name"]
            row[f"dinov3_best_{metric}"] = best_dino[metric]
            row[f"dinov3_best_{metric}_model"] = best_dino["model_size"]
            row[f"dinov3_best_{metric}_init"] = (
                "Baseline" if _parse_fraction_to_int(best_dino["fraction"]) == 0 else "TAPT"
            )
            row[f"delta_{metric}_tv_minus_dinov3"] = (
                best_tv[metric + "_mean"] - best_dino[metric + "_mean"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_single_setting(
    records,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    model_name: str,
    k_shot: int,
    args,
    logger,
    out_dir: str,
    completed_keys: set,
):
    device = get_device()
    num_classes = len(class_to_idx)
    seed_results = []

    logger.info(f"Loading torchvision backbone={model_name} (official ImageNet weights)")
    raw_backbone = TorchvisionBackbone(model_name).to(device)
    feature_backbone = _maybe_wrap_dataparallel_compat(copy.deepcopy(raw_backbone), device)
    feature_backbone.eval()

    seed_iter: Iterable[int] = tqdm(
        args.seeds,
        desc=f"{model_name}-k{k_shot}",
        leave=False,
    )

    for seed in seed_iter:
        run_key = (model_name, int(k_shot), int(seed))
        if args.skip_completed and run_key in completed_keys:
            logger.info(f"[Skip Completed] model={model_name}, k={k_shot}, seed={seed}")
            continue

        set_seed(seed)
        support_records, query_records = select_support_query(records, num_classes, k_shot, seed)
        seed_dir = ensure_dir(os.path.join(out_dir, model_name, f"kshot_{k_shot}", f"seed_{seed}"))
        start_ts = time.time()

        if args.save_episode_metadata:
            _records_to_df(support_records).to_csv(
                os.path.join(seed_dir, "support_records.csv"), index=False, encoding="utf-8-sig"
            )
            _records_to_df(query_records).to_csv(
                os.path.join(seed_dir, "query_records.csv"), index=False, encoding="utf-8-sig"
            )

        support_ds = RadiolariaDataset(support_records, transform=get_eval_transform(args.img_size))
        query_ds = RadiolariaDataset(query_records, transform=get_eval_transform(args.img_size))

        support_loader = torch.utils.data.DataLoader(
            support_ds,
            batch_size=min(args.eval_batch_size, len(support_ds)),
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        query_loader = torch.utils.data.DataLoader(
            query_ds,
            batch_size=min(args.eval_batch_size, len(query_ds)),
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        support_feats, support_labels, _ = extract_features(feature_backbone, support_loader, device)
        query_feats, query_labels, _ = extract_features(feature_backbone, query_loader, device)
        if args.save_features:
            np.savez_compressed(
                os.path.join(seed_dir, "features_eval.npz"),
                support_feats=support_feats,
                support_labels=support_labels,
                query_feats=query_feats,
                query_labels=query_labels,
            )

        proto_acc, proto_f1, proto_top3, proto_top5, _ = run_prototype_classifier(
            support_feats=support_feats,
            support_labels=support_labels,
            query_feats=query_feats,
            query_labels=query_labels,
        )

        lp = LogisticRegression(
            random_state=seed,
            C=args.lp_c,
            solver="lbfgs",
            max_iter=args.lp_max_iter,
            n_jobs=1,
        )
        lp.fit(support_feats, support_labels)
        lp_pred = lp.predict(query_feats)
        lp_acc = float(np.mean(lp_pred == query_labels) * 100.0)
        lp_f1 = float(f1_score(query_labels, lp_pred, average="macro"))
        if args.save_predictions:
            _save_prediction_table(
                query_labels,
                lp_pred,
                [r.path for r in query_records],
                idx_to_class,
                os.path.join(seed_dir, "lp_predictions.csv"),
            )

        ft_acc, ft_f1, ft_top3, ft_top5, ft_true, ft_pred, _ = train_two_stage(
            backbone=raw_backbone,
            num_classes=num_classes,
            support_records=support_records,
            query_records=query_records,
            args=args,
            device=device,
            run_dir=seed_dir,
            logger=logger,
        )

        if args.save_confusion:
            labels_all = list(range(num_classes))
            class_names = [idx_to_class[i] for i in labels_all]
            plot_confusion_matrix(ft_true, ft_pred, labels_all, class_names, os.path.join(seed_dir, "confusion_matrix"))

        if args.save_predictions:
            _save_prediction_table(
                ft_true,
                ft_pred,
                [r.path for r in query_records],
                idx_to_class,
                os.path.join(seed_dir, "ft_predictions.csv"),
            )

        elapsed = time.time() - start_ts
        _save_json(
            os.path.join(seed_dir, "metrics_detail.json"),
            {
                "model_name": model_name,
                "k_shot": int(k_shot),
                "seed": int(seed),
                "proto_acc": proto_acc,
                "proto_top3": proto_top3,
                "proto_top5": proto_top5,
                "proto_macro_f1": proto_f1,
                "lp_acc": lp_acc,
                "lp_macro_f1": lp_f1,
                "ft_acc": ft_acc,
                "ft_top3": ft_top3,
                "ft_top5": ft_top5,
                "ft_macro_f1": ft_f1,
                "elapsed_seconds": elapsed,
            },
        )

        seed_results.append(
            {
                "model_name": model_name,
                "k_shot": int(k_shot),
                "seed": int(seed),
                "proto_acc": proto_acc,
                "proto_top3": proto_top3,
                "proto_top5": proto_top5,
                "proto_macro_f1": proto_f1,
                "lp_acc": lp_acc,
                "lp_macro_f1": lp_f1,
                "ft_acc": ft_acc,
                "ft_top3": ft_top3,
                "ft_top5": ft_top5,
                "ft_macro_f1": ft_f1,
            }
        )

    return seed_results


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "proto_acc",
        "proto_top3",
        "proto_top5",
        "proto_macro_f1",
        "lp_acc",
        "lp_macro_f1",
        "ft_acc",
        "ft_top3",
        "ft_top5",
        "ft_macro_f1",
    ]
    if df.empty:
        return pd.DataFrame(columns=["model_name", "k_shot"] + metric_cols)

    rows = []
    for (model_name, k_shot), group in df.groupby(["model_name", "k_shot"]):
        row = {"model_name": model_name, "k_shot": int(k_shot)}
        for col in metric_cols:
            if col in group.columns:
                row[col] = _format_metric(group[col].values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_name", "k_shot"]).reset_index(drop=True)


def main():
    args = parse_args()

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    out_dir = ensure_dir(os.path.join(args.results_root, args.experiment_name))
    logger = setup_logger(os.path.join(out_dir, "run.log"))
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Detected GPUs: {available_gpus}")
    logger.info(f"Models: {args.models}")
    logger.info(f"K-shots: {args.k_shots}")
    logger.info(f"Seeds: {args.seeds}")
    _save_json(
        os.path.join(out_dir, "run_config.json"),
        {
            "data_root": args.data_root,
            "results_root": args.results_root,
            "experiment_name": args.experiment_name,
            "models": args.models,
            "k_shots": args.k_shots,
            "seeds": args.seeds,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs_linear": args.epochs_linear,
            "epochs_full_ft": args.epochs_full_ft,
            "lr_linear": args.lr_linear,
            "lr_full_ft": args.lr_full_ft,
            "weight_decay": args.weight_decay,
            "layer_decay": args.layer_decay,
            "save_confusion": args.save_confusion,
            "save_predictions": args.save_predictions,
            "save_features": args.save_features,
            "save_episode_metadata": args.save_episode_metadata,
            "resume_training": args.resume_training,
            "save_checkpoints": args.save_checkpoints,
            "checkpoint_every_epoch": args.checkpoint_every_epoch,
        },
    )

    existing_seed_path = os.path.join(out_dir, "torchvision_seed_results.csv")
    existing_seed_df = pd.DataFrame()
    completed_keys = set()
    if args.skip_completed and os.path.exists(existing_seed_path):
        try:
            existing_seed_df = pd.read_csv(existing_seed_path)
            need_cols = {"model_name", "k_shot", "seed"}
            if need_cols.issubset(set(existing_seed_df.columns)):
                for _, row in existing_seed_df.iterrows():
                    completed_keys.add((str(row["model_name"]), int(row["k_shot"]), int(row["seed"])))
                logger.info(f"Loaded completed runs: {len(completed_keys)}")
        except Exception as exc:
            logger.warning(f"Failed to read existing seed results: {exc}")

    records, class_to_idx, idx_to_class = build_radiolaria_index(args.data_root)
    logger.info(f"Loaded dataset records={len(records)}, classes={len(class_to_idx)}")

    all_rows = []
    total_jobs = len(args.models) * len(args.k_shots)
    pbar = tqdm(total=total_jobs, desc="Torchvision few-shot benchmark")

    for model_name in args.models:
        for k_shot in args.k_shots:
            rows = run_single_setting(
                records=records,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                model_name=model_name,
                k_shot=int(k_shot),
                args=args,
                logger=logger,
                out_dir=out_dir,
                completed_keys=completed_keys,
            )
            all_rows.extend(rows)
            pbar.update(1)

            current_seed_df = pd.DataFrame(all_rows)
            merged_seed_df = (
                pd.concat([existing_seed_df, current_seed_df], ignore_index=True)
                if not existing_seed_df.empty
                else current_seed_df
            )
            if not merged_seed_df.empty:
                merged_seed_df = merged_seed_df.drop_duplicates(
                    subset=["model_name", "k_shot", "seed"],
                    keep="last",
                ).reset_index(drop=True)
                merged_summary_df = aggregate_results(merged_seed_df)
                save_result_tables(merged_seed_df, out_dir, "torchvision_seed_results")
                save_result_tables(merged_summary_df, out_dir, "torchvision_summary")

    pbar.close()

    new_seed_df = pd.DataFrame(all_rows)
    if not existing_seed_df.empty:
        seed_df = pd.concat([existing_seed_df, new_seed_df], ignore_index=True)
        if not seed_df.empty:
            seed_df = seed_df.drop_duplicates(subset=["model_name", "k_shot", "seed"], keep="last").reset_index(drop=True)
    else:
        seed_df = new_seed_df

    summary_df = aggregate_results(seed_df)
    save_result_tables(seed_df, out_dir, "torchvision_seed_results")
    save_result_tables(summary_df, out_dir, "torchvision_summary")

    comparison_df = build_dinov3_comparison(summary_df, args.dinov3_summary)
    if not comparison_df.empty:
        save_result_tables(comparison_df, out_dir, "best_torchvision_vs_best_dinov3")
        logger.info(f"DINOv3 comparison saved using summary: {args.dinov3_summary}")
    else:
        logger.info("Skipped DINOv3 comparison because the summary file was not found or was empty.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
