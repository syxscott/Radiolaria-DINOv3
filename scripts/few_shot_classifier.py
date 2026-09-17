#!/usr/bin/env python3
"""核心 few-shot 实验脚本 (K-shot = 1/3/5/10, 5 seeds)。

每类抽 K 张 support、其余 query; 方法: Prototype / 冻结 LP / 两阶段 FT
(LLRD + Mixup/CutMix + SoftCE); 输出 CSV/Markdown/Excel + 混淆矩阵 + attention 图。

示例:
  python scripts/few_shot_classifier.py --config configs/experiments/few_shot_classifier.yaml --run_full_grid
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (
    DinoClassifier, MixupCutmixCollator, RadiolariaDataset, SEEDS_DEFAULT,
    SoftTargetCrossEntropy, build_optimizer_llrd, build_radiolaria_index,
    ensure_dir, evaluate_classifier, extract_features, get_backbone_module,
    get_device, get_eval_transform, get_train_transform, load_dinov3_backbone,
    load_yaml_config, maybe_wrap_dataparallel, parse_fraction_list,
    parse_model_size_alias, plot_confusion_matrix, run_knn_classifier,
    run_prototype_classifier, save_attention_visualization, save_result_tables,
    select_support_query, set_seed, set_train_mode, setup_logger,
    summarize_metric, topk_accuracy_from_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Radiolaria few-shot classifier")
    parser.add_argument("--config", type=str, default="configs/experiments/few_shot_classifier.yaml")

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)

    parser.add_argument("--tapt_fraction", type=str, default=None, help="single fraction, e.g. 100")
    parser.add_argument("--model_size", type=str, default=None, help="small/base/large")
    parser.add_argument("--k_shot", type=int, default=None, help="single k-shot override")

    parser.add_argument("--run_full_grid", action="store_true", help="ignore single overrides and run full grid")
    return parser.parse_args()


def train_two_stage(
    backbone: nn.Module,
    num_classes: int,
    support_records,
    query_records,
    device: torch.device,
    cfg: dict,
):
    train_ds = RadiolariaDataset(
        support_records,
        transform=get_train_transform(
            cfg["img_size"], cfg.get("randaugment_magnitude", 2), cfg.get("rotation_deg", 0)
        ),
    )
    eval_ds = RadiolariaDataset(query_records, transform=get_eval_transform(cfg["img_size"]))

    mix_collator = MixupCutmixCollator(
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],
        prob=cfg["mix_prob"],
        num_classes=num_classes,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(cfg["batch_size"], len(train_ds)),
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=mix_collator,
        drop_last=False,
        persistent_workers=cfg["num_workers"] > 0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=min(cfg["eval_batch_size"], len(eval_ds)),
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = DinoClassifier(backbone=copy.deepcopy(backbone), num_classes=num_classes).to(device)
    model = maybe_wrap_dataparallel(model)

    criterion = SoftTargetCrossEntropy()

    # Stage-1: Linear probing
    set_train_mode(get_backbone_module(model), "linear")
    optimizer = torch.optim.AdamW(get_backbone_module(model).head.parameters(), lr=cfg["lr_linear"], weight_decay=cfg["weight_decay"])

    linear_epoch_iter = range(cfg["epochs_linear"])
    if cfg.get("progress_tqdm", True):
        linear_epoch_iter = tqdm(
            linear_epoch_iter,
            desc="Linear-Probing Epochs",
            leave=False,
        )

    for _ in linear_epoch_iter:
        model.train()
        get_backbone_module(model).backbone.eval()  # 冻结 backbone 防止 BN 统计量漂移
        for imgs, targets, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    # Stage-2: Full FT + LLRD
    set_train_mode(get_backbone_module(model), "full_ft")
    if cfg["use_llrd"]:
        optimizer = build_optimizer_llrd(model, cfg["lr_full_ft"], cfg["weight_decay"], cfg["layer_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr_full_ft"], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs_full_ft"]))

    ft_epoch_iter = range(cfg["epochs_full_ft"])
    if cfg.get("progress_tqdm", True):
        ft_epoch_iter = tqdm(
            ft_epoch_iter,
            desc="Full-FT Epochs",
            leave=False,
        )

    for _ in ft_epoch_iter:
        model.train()
        for imgs, targets, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    acc, macro_f1, top3, top5, y_true, y_pred, paths = evaluate_classifier(model, eval_loader, device)
    return acc, macro_f1, top3, top5, y_true, y_pred, paths, get_backbone_module(model).backbone


def run_single_setting(
    records,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    fraction: str,
    model_size: str,
    k_shot: int,
    cfg: dict,
    logger,
):
    num_classes = len(class_to_idx)
    device = get_device()

    ckpt_map = cfg["checkpoints"].get(model_size, {})
    ckpt_path = ckpt_map.get(str(fraction))
    if not ckpt_path:
        logger.warning(f"[Skip] No checkpoint configured for model={model_size}, fraction={fraction}")
        return []
    if not os.path.exists(ckpt_path):
        logger.warning(f"[Skip] Checkpoint not found: {ckpt_path}")
        return []

    logger.info(f"Loading model={model_size}, fraction={fraction}, checkpoint={ckpt_path}")
    backbone = load_dinov3_backbone(cfg["repo_root"], model_size, ckpt_path, prefer_torch_hub=True)
    backbone.to(device)
    backbone.eval()

    seed_results: List[dict] = []

    seed_iter = cfg["seeds"]
    if cfg.get("progress_tqdm", True):
        seed_iter = tqdm(
            seed_iter,
            desc=f"Seeds f{fraction}-{model_size}-k{k_shot}",
            leave=False,
        )

    for seed in seed_iter:
        set_seed(seed)
        support_records, query_records = select_support_query(records, num_classes, k_shot, seed)

        # -------- Prototype --------
        support_ds_eval = RadiolariaDataset(support_records, transform=get_eval_transform(cfg["img_size"]))
        query_ds_eval = RadiolariaDataset(query_records, transform=get_eval_transform(cfg["img_size"]))
        support_loader = torch.utils.data.DataLoader(
            support_ds_eval,
            batch_size=min(cfg["eval_batch_size"], len(support_ds_eval)),
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
        query_loader = torch.utils.data.DataLoader(
            query_ds_eval,
            batch_size=min(cfg["eval_batch_size"], len(query_ds_eval)),
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        support_feats, support_labels, _ = extract_features(backbone, support_loader, device)
        query_feats, query_labels, query_paths = extract_features(backbone, query_loader, device)
        proto_acc, proto_f1, proto_top3, proto_top5, proto_pred = run_prototype_classifier(
            support_feats=support_feats,
            support_labels=support_labels,
            query_feats=query_feats,
            query_labels=query_labels,
        )

        # -------- kNN on frozen features (1-NN cosine) --------
        knn_acc, knn_f1, _ = run_knn_classifier(
            support_feats, support_labels, query_feats, query_labels, k=1
        )

        # -------- Linear Probing on frozen features --------
        lp = LogisticRegression(
            random_state=seed,
            C=cfg["lp_c"],
            solver="lbfgs",
            max_iter=cfg["lp_max_iter"],
            n_jobs=1,
        )
        lp.fit(support_feats, support_labels)
        lp_pred = lp.predict(query_feats)
        lp_proba = lp.predict_proba(query_feats)
        lp_acc = float(np.mean(lp_pred == query_labels) * 100.0)
        lp_topk = topk_accuracy_from_scores(lp_proba, lp.classes_, query_labels)
        lp_f1 = float(f1_score(query_labels, lp_pred, average="macro"))

        # -------- Progressive supervised fine-tuning --------
        ft_acc, ft_f1, ft_top3, ft_top5, ft_true, ft_pred, ft_paths, ft_backbone = train_two_stage(
            backbone=backbone,
            num_classes=num_classes,
            support_records=support_records,
            query_records=query_records,
            device=device,
            cfg=cfg,
        )

        # Save confusion matrix + attention map for this seed
        setting_dir = ensure_dir(
            os.path.join(
                cfg["results_root"],
                cfg["experiment_name"],
                f"fraction_{fraction}",
                f"model_{model_size}",
                f"kshot_{k_shot}",
                f"seed_{seed}",
            )
        )

        labels_all = list(range(num_classes))
        class_names = [idx_to_class[i] for i in labels_all]
        plot_confusion_matrix(ft_true, ft_pred, labels_all, class_names, os.path.join(setting_dir, "confusion_matrix"))

        # attention rollout-like map on first query sample
        if len(query_ds_eval) > 0:
            img_t, _, path, _ = query_ds_eval[0]
            save_attention_visualization(ft_backbone, img_t.to(device), path, os.path.join(setting_dir, "attention_rollout"))

        seed_results.append(
            {
                "fraction": str(fraction),
                "model_size": model_size,
                "k_shot": int(k_shot),
                "seed": int(seed),
                "proto_acc": proto_acc,
                "proto_top3": proto_top3,
                "proto_top5": proto_top5,
                "proto_macro_f1": proto_f1,
                "knn_acc": knn_acc,
                "knn_macro_f1": knn_f1,
                "lp_acc": lp_acc,
                "lp_top3": lp_topk[3],
                "lp_top5": lp_topk[5],
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
        "knn_acc",
        "knn_macro_f1",
        "lp_acc",
        "lp_top3",
        "lp_top5",
        "lp_macro_f1",
        "ft_acc",
        "ft_top3",
        "ft_top5",
        "ft_macro_f1",
    ]
    if df.empty:
        return pd.DataFrame(columns=["fraction", "model_size", "k_shot"] + metric_cols)

    grouped = []
    for (fraction, model_size, k_shot), g in df.groupby(["fraction", "model_size", "k_shot"]):
        row = {"fraction": fraction, "model_size": model_size, "k_shot": k_shot}
        for col in metric_cols:
            if col in g.columns:
                row[col] = summarize_metric(g[col].values)
        grouped.append(row)
    out = pd.DataFrame(grouped).sort_values(["fraction", "model_size", "k_shot"]).reset_index(drop=True)
    return out


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.results_root is not None:
        cfg["results_root"] = args.results_root

    cfg["repo_root"] = str(Path(__file__).resolve().parents[1])
    cfg["seeds"] = cfg.get("seeds", SEEDS_DEFAULT)

    if args.k_shot is not None:
        k_shots = [int(args.k_shot)]
    else:
        k_shots = [int(x) for x in cfg["k_shots"]]

    if args.run_full_grid:
        fractions = [str(x) for x in cfg["fractions"]]
        model_sizes = [parse_model_size_alias(x) for x in cfg["model_sizes"]]
    else:
        fractions = [args.tapt_fraction] if args.tapt_fraction is not None else [str(x) for x in cfg["fractions"]]
        if args.model_size is not None:
            model_sizes = [parse_model_size_alias(args.model_size)]
        else:
            model_sizes = [parse_model_size_alias(x) for x in cfg["model_sizes"]]

    out_dir = ensure_dir(os.path.join(cfg["results_root"], cfg["experiment_name"]))
    logger = setup_logger(os.path.join(out_dir, "run.log"))

    records, class_to_idx, idx_to_class = build_radiolaria_index(cfg["data_root"])
    logger.info(f"Loaded dataset records={len(records)}, classes={len(class_to_idx)}")

    # Basic integrity report
    count_df = pd.DataFrame([{"class_name": r.class_name, "path": r.path} for r in records])
    count_summary = count_df.groupby("class_name").size().describe().to_dict()
    logger.info(f"Per-class count summary: {count_summary}")

    all_seed_rows = []

    total_jobs = len(fractions) * len(model_sizes) * len(k_shots)
    job_bar = tqdm(total=total_jobs, desc="Few-shot grid")

    for fraction in fractions:
        for model_size in model_sizes:
            for k in k_shots:
                rows = run_single_setting(
                    records=records,
                    class_to_idx=class_to_idx,
                    idx_to_class=idx_to_class,
                    fraction=str(fraction),
                    model_size=model_size,
                    k_shot=int(k),
                    cfg=cfg,
                    logger=logger,
                )
                all_seed_rows.extend(rows)
                job_bar.update(1)

    job_bar.close()

    seed_df = pd.DataFrame(all_seed_rows)
    agg_df = aggregate_results(seed_df)

    save_result_tables(seed_df, out_dir, "few_shot_seed_results")
    save_result_tables(agg_df, out_dir, "few_shot_summary")

    logger.info("Done. Results saved:")
    logger.info(f"- {out_dir}")


if __name__ == "__main__":
    main()
