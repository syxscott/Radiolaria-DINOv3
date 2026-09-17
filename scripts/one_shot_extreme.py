#!/usr/bin/env python3
"""1-shot 极限测试: 对比 TAPT 0%/100%, Proto + 线性探针, 失败案例可视化。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (
    RadiolariaDataset, SEEDS_DEFAULT, build_radiolaria_index, ensure_dir,
    extract_features, get_device, get_eval_transform, load_dinov3_backbone,
    load_yaml_config, parse_model_size_alias, run_knn_classifier,
    run_prototype_classifier, save_result_tables, select_support_query,
    set_seed, setup_logger, summarize_metric, topk_accuracy_from_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description="1-shot extreme evaluation")
    parser.add_argument("--config", type=str, default="configs/experiments/one_shot_extreme.yaml")
    parser.add_argument("--model_size", type=str, default=None, help="small/base/large")
    return parser.parse_args()


def save_failure_cases(
    query_records,
    y_true,
    y_pred,
    idx_to_class,
    out_dir: str,
    max_cases: int = 12,
):
    ensure_dir(out_dir)

    alb_mask = []
    for i, t in enumerate(y_true):
        cname = idx_to_class[int(t)]
        alb_mask.append("albaillella" in cname.lower())

    mis_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p and alb_mask[i]]
    if len(mis_idx) == 0:
        return None

    mis_idx = mis_idx[:max_cases]

    ncols = 4
    nrows = int(np.ceil(len(mis_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for ax in axes.flatten():
        ax.axis("off")

    for j, idx in enumerate(mis_idx):
        rec = query_records[idx]
        try:
            img = Image.open(rec.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        r, c = divmod(j, ncols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(
            f"T:{idx_to_class[int(y_true[idx])]}\nP:{idx_to_class[int(y_pred[idx])]}",
            fontsize=9,
        )
        axes[r, c].axis("off")

    plt.tight_layout()
    png = os.path.join(out_dir, "failure_cases_albaillellaria.png")
    pdf = os.path.join(out_dir, "failure_cases_albaillellaria.pdf")
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    return png


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg["repo_root"] = str(Path(__file__).resolve().parents[1])
    cfg["seeds"] = cfg.get("seeds", SEEDS_DEFAULT)

    model_size = parse_model_size_alias(args.model_size) if args.model_size else parse_model_size_alias(cfg["model_size"])

    out_dir = ensure_dir(os.path.join(cfg["results_root"], cfg["experiment_name"], f"model_{model_size}"))
    logger = setup_logger(os.path.join(out_dir, "run.log"))

    records, class_to_idx, idx_to_class = build_radiolaria_index(cfg["data_root"])
    num_classes = len(class_to_idx)

    fractions = ["0", "100"]
    all_rows = []

    for frac in fractions:
        ckpt = cfg.get("checkpoints", {}).get(model_size, {}).get(frac)
        if not ckpt:
            logger.warning(f"[Skip] No checkpoint configured for model={model_size}, fraction={frac}")
            continue
        if not os.path.exists(ckpt):
            logger.warning(f"[Skip] Checkpoint not found: {ckpt}")
            continue

        logger.info(f"Running 1-shot for fraction={frac}, model={model_size}")

        seed_metrics = []
        for seed in tqdm(cfg["seeds"], desc=f"fraction={frac}"):
            set_seed(seed)
            support_records, query_records = select_support_query(records, num_classes, 1, seed)

            backbone = load_dinov3_backbone(cfg["repo_root"], model_size, ckpt, prefer_torch_hub=True)
            device = get_device()
            backbone.to(device)
            backbone.eval()

            support_ds = RadiolariaDataset(support_records, transform=get_eval_transform(cfg["img_size"]))
            query_ds = RadiolariaDataset(query_records, transform=get_eval_transform(cfg["img_size"]))
            support_loader = DataLoader(support_ds, batch_size=min(cfg["batch_size"], len(support_ds)), shuffle=False, num_workers=cfg["num_workers"])
            query_loader = DataLoader(query_ds, batch_size=min(cfg["batch_size"], len(query_ds)), shuffle=False, num_workers=cfg["num_workers"])

            support_feats, support_labels, _ = extract_features(backbone, support_loader, device)
            query_feats, query_labels, query_paths = extract_features(backbone, query_loader, device)

            # Prototype
            proto_acc, proto_f1, proto_top3, proto_top5, proto_pred = run_prototype_classifier(
                support_feats, support_labels, query_feats, query_labels
            )

            # kNN (1-NN cosine)
            knn_acc, knn_f1, _ = run_knn_classifier(support_feats, support_labels, query_feats, query_labels, k=1)

            # Linear probe
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

            seed_metrics.append({
                "fraction": frac,
                "model_size": model_size,
                "seed": seed,
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
            })

            # failure cases for 100% only (first seed)
            if frac == "100" and seed == cfg["seeds"][0]:
                save_failure_cases(query_records, query_labels, lp_pred, idx_to_class, os.path.join(out_dir, "failure_cases"))

        all_rows.extend(seed_metrics)

    seed_df = pd.DataFrame(all_rows)
    save_result_tables(seed_df, out_dir, "one_shot_seed_results")

    summary_rows = []
    for frac, g in seed_df.groupby("fraction"):
        summary_rows.append(
            {
                "fraction": frac,
                "model_size": model_size,
                "proto_acc": summarize_metric(g["proto_acc"].values),
                "proto_top3": summarize_metric(g["proto_top3"].values),
                "proto_top5": summarize_metric(g["proto_top5"].values),
                "proto_macro_f1": summarize_metric(g["proto_macro_f1"].values),
                "knn_acc": summarize_metric(g["knn_acc"].values),
                "knn_macro_f1": summarize_metric(g["knn_macro_f1"].values),
                "lp_acc": summarize_metric(g["lp_acc"].values),
                "lp_top3": summarize_metric(g["lp_top3"].values),
                "lp_top5": summarize_metric(g["lp_top5"].values),
                "lp_macro_f1": summarize_metric(g["lp_macro_f1"].values),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("fraction")
    save_result_tables(summary_df, out_dir, "one_shot_summary")

    logger.info("Done.")


if __name__ == "__main__":
    main()
