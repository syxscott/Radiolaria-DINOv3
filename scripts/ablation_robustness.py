#!/usr/bin/env python3
"""消融 + 鲁棒性实验 (LLRD / Mixup+CutMix / RandAugment 开关; 高斯噪声+碎片化;
可选 cross-domain)。输出 LaTeX/CSV/Markdown/Excel。

示例:
  python scripts/ablation_robustness.py --config configs/experiments/ablation_robustness.yaml --tapt_fraction 100 --model_size base --k_shot 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (
    DinoClassifier, MixupCutmixCollator, RadiolariaDataset, SEEDS_DEFAULT,
    SoftTargetCrossEntropy, build_optimizer_llrd, build_radiolaria_index,
    ensure_dir, evaluate_classifier, get_backbone_module, get_device,
    get_eval_transform, get_train_transform, load_dinov3_backbone,
    load_yaml_config, maybe_wrap_dataparallel, parse_model_size_alias,
    save_result_tables, select_support_query, set_seed, set_train_mode,
    setup_logger, summarize_metric,
)


class PerturbedDataset(Dataset):
    def __init__(self, base_records, transform, gaussian_std: float = 0.0, fragment_prob: float = 0.0):
        self.base = RadiolariaDataset(base_records, transform=transform)
        self.gaussian_std = gaussian_std
        self.fragment_prob = fragment_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label, path, cname = self.base[idx]
        if self.gaussian_std > 0:
            noise = torch.randn_like(img) * self.gaussian_std
            img = (img + noise).clamp(-3, 3)
        if self.fragment_prob > 0 and np.random.rand() < self.fragment_prob:
            c, h, w = img.shape
            x1 = np.random.randint(0, max(1, w // 2))
            y1 = np.random.randint(0, max(1, h // 2))
            x2 = np.random.randint(max(x1 + 1, w // 2), w)
            y2 = np.random.randint(max(y1 + 1, h // 2), h)
            img[:, y1:y2, x1:x2] = 0
        return img, label, path, cname


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation and robustness for Radiolaria DINOv3")
    parser.add_argument("--config", type=str, default="configs/experiments/ablation_robustness.yaml")
    parser.add_argument("--tapt_fraction", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--k_shot", type=int, default=None)
    return parser.parse_args()


def train_eval_two_stage(
    backbone,
    num_classes: int,
    support_records,
    query_records,
    device,
    train_cfg: dict,
    use_llrd: bool,
    use_mixcut: bool,
    rand_mag: int,
    perturb_eval: Tuple[float, float] = (0.0, 0.0),
):
    train_ds = RadiolariaDataset(support_records, transform=get_train_transform(train_cfg["img_size"], rand_mag))
    eval_transform = get_eval_transform(train_cfg["img_size"])
    if perturb_eval != (0.0, 0.0):
        eval_ds = PerturbedDataset(query_records, transform=eval_transform, gaussian_std=perturb_eval[0], fragment_prob=perturb_eval[1])
    else:
        eval_ds = RadiolariaDataset(query_records, transform=eval_transform)

    collate_fn = None
    criterion = nn.CrossEntropyLoss()
    if use_mixcut:
        collate_fn = MixupCutmixCollator(
            mixup_alpha=train_cfg["mixup_alpha"],
            cutmix_alpha=train_cfg["cutmix_alpha"],
            prob=train_cfg["mix_prob"],
            num_classes=num_classes,
        )
        criterion = SoftTargetCrossEntropy()

    train_loader = DataLoader(
        train_ds,
        batch_size=min(train_cfg["batch_size"], len(train_ds)),
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=train_cfg["num_workers"] > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=min(train_cfg["eval_batch_size"], len(eval_ds)),
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = DinoClassifier(backbone=backbone, num_classes=num_classes).to(device)
    model = maybe_wrap_dataparallel(model)

    # Stage1 linear
    set_train_mode(get_backbone_module(model), "linear")
    optim1 = torch.optim.AdamW(get_backbone_module(model).head.parameters(), lr=train_cfg["lr_linear"], weight_decay=train_cfg["weight_decay"])

    for _ in range(train_cfg["epochs_linear"]):
        model.train()
        get_backbone_module(model).backbone.eval()  # 冻结 backbone 防止 BN 统计量漂移
        for batch in train_loader:
            if use_mixcut:
                imgs, targets, _, _ = batch
            else:
                imgs, y, _, _ = batch
                targets = y
            imgs = imgs.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)

            optim1.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optim1.step()

    # Stage2 full ft
    set_train_mode(get_backbone_module(model), "full_ft")
    if use_llrd:
        optim2 = build_optimizer_llrd(model, train_cfg["lr_full_ft"], train_cfg["weight_decay"], train_cfg["layer_decay"])
    else:
        optim2 = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr_full_ft"], weight_decay=train_cfg["weight_decay"])

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=max(1, train_cfg["epochs_full_ft"]))

    for _ in range(train_cfg["epochs_full_ft"]):
        model.train()
        for batch in train_loader:
            if use_mixcut:
                imgs, targets, _, _ = batch
            else:
                imgs, y, _, _ = batch
                targets = y
            imgs = imgs.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)

            optim2.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            loss.backward()
            optim2.step()
        sch.step()

    acc, f1, _top3, _top5, _ytrue, _ypred, _paths = evaluate_classifier(model, eval_loader, device)
    return acc, f1


def split_cross_domain(records, domain_csv: str):
    if domain_csv is None or not os.path.exists(domain_csv):
        return None

    df = pd.read_csv(domain_csv)
    path_col = "filepath" if "filepath" in df.columns else df.columns[0]
    dom_col = "domain" if "domain" in df.columns else df.columns[1]

    dom_map = {str(Path(p).name): str(d).lower() for p, d in zip(df[path_col], df[dom_col])}

    src, tgt = [], []
    for r in records:
        key = Path(r.path).name
        d = dom_map.get(key, "")
        if "south" in d:
            src.append(r)
        elif "japan" in d:
            tgt.append(r)

    if len(src) == 0 or len(tgt) == 0:
        return None
    return src, tgt


def build_cross_domain_episode(src_records, tgt_records, k_shot: int, seed: int):
    """Build support/query for cross-domain using intersection classes only."""
    rng = np.random.default_rng(seed)

    src_by_cls = {}
    for r in src_records:
        src_by_cls.setdefault(r.label, []).append(r)

    tgt_by_cls = {}
    for r in tgt_records:
        tgt_by_cls.setdefault(r.label, []).append(r)

    common = sorted([c for c in src_by_cls.keys() if c in tgt_by_cls])
    support = []
    query = []

    for c in common:
        src_cls = src_by_cls[c]
        if len(src_cls) < 1:
            continue
        take_k = min(k_shot, len(src_cls))
        idx = np.arange(len(src_cls))
        rng.shuffle(idx)
        support.extend([src_cls[i] for i in idx[:take_k]])
        query.extend(tgt_by_cls[c])

    if len(support) == 0 or len(query) == 0:
        return None
    return support, query


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    cfg["repo_root"] = str(Path(__file__).resolve().parents[1])
    cfg["seeds"] = cfg.get("seeds", SEEDS_DEFAULT)

    fractions = [args.tapt_fraction] if args.tapt_fraction else [str(x) for x in cfg["fractions"]]
    model_sizes = [parse_model_size_alias(args.model_size)] if args.model_size else [parse_model_size_alias(x) for x in cfg["model_sizes"]]
    k_shot = int(args.k_shot) if args.k_shot else int(cfg["k_shot"])

    out_dir = ensure_dir(os.path.join(cfg["results_root"], cfg["experiment_name"]))
    logger = setup_logger(os.path.join(out_dir, "run.log"))

    records, class_to_idx, _ = build_radiolaria_index(cfg["data_root"])
    num_classes = len(class_to_idx)

    ablation_settings = [
        {"name": "llrd_on_mix_on_aug2", "use_llrd": True, "use_mixcut": True, "rand_mag": 2},
        {"name": "llrd_off_mix_on_aug2", "use_llrd": False, "use_mixcut": True, "rand_mag": 2},
        {"name": "llrd_on_mix_off_aug2", "use_llrd": True, "use_mixcut": False, "rand_mag": 2},
        {"name": "llrd_on_mix_on_aug0", "use_llrd": True, "use_mixcut": True, "rand_mag": 0},
    ]

    rows = []
    total_jobs = len(fractions) * len(model_sizes) * len(cfg["seeds"]) * len(ablation_settings)
    pbar = tqdm(total=total_jobs, desc="Ablation")

    for fraction in fractions:
        for model_size in model_sizes:
            ckpt = cfg.get("checkpoints", {}).get(model_size, {}).get(str(fraction))
            total_skip = len(cfg["seeds"]) * len(ablation_settings)
            if not ckpt:
                logger.warning(f"[Skip] No checkpoint configured for model={model_size}, fraction={fraction}")
                pbar.update(total_skip)
                continue
            if not os.path.exists(ckpt):
                logger.warning(f"[Skip] Checkpoint not found: {ckpt}")
                pbar.update(total_skip)
                continue

            for seed in cfg["seeds"]:
                set_seed(seed)
                support_records, query_records = select_support_query(records, num_classes, k_shot, seed)

                for ab in ablation_settings:
                    backbone = load_dinov3_backbone(cfg["repo_root"], model_size, ckpt, prefer_torch_hub=True)
                    backbone.to(get_device())
                    acc, f1 = train_eval_two_stage(
                        backbone=backbone,
                        num_classes=num_classes,
                        support_records=support_records,
                        query_records=query_records,
                        device=get_device(),
                        train_cfg=cfg,
                        use_llrd=ab["use_llrd"],
                        use_mixcut=ab["use_mixcut"],
                        rand_mag=ab["rand_mag"],
                    )

                    # robustness noise
                    backbone_noise = load_dinov3_backbone(cfg["repo_root"], model_size, ckpt, prefer_torch_hub=True)
                    backbone_noise.to(get_device())
                    nacc, nf1 = train_eval_two_stage(
                        backbone=backbone_noise,
                        num_classes=num_classes,
                        support_records=support_records,
                        query_records=query_records,
                        device=get_device(),
                        train_cfg=cfg,
                        use_llrd=ab["use_llrd"],
                        use_mixcut=ab["use_mixcut"],
                        rand_mag=ab["rand_mag"],
                        perturb_eval=(cfg["noise_std"], cfg["fragment_prob"]),
                    )

                    row = {
                        "fraction": str(fraction),
                        "model_size": model_size,
                        "seed": seed,
                        "k_shot": k_shot,
                        "setting": ab["name"],
                        "acc": acc,
                        "macro_f1": f1,
                        "noise_acc": nacc,
                        "noise_macro_f1": nf1,
                    }

                    # cross-domain if metadata available
                    cross = split_cross_domain(records, cfg.get("domain_split_csv"))
                    if cross is not None:
                        src_records, tgt_records = cross
                        episode = build_cross_domain_episode(src_records, tgt_records, k_shot=k_shot, seed=seed)
                        if episode is None:
                            row["cross_domain_acc"] = np.nan
                            row["cross_domain_macro_f1"] = np.nan
                            rows.append(row)
                            pbar.update(1)
                            continue
                        src_support, tgt_query = episode

                        backbone_cd = load_dinov3_backbone(cfg["repo_root"], model_size, ckpt, prefer_torch_hub=True)
                        backbone_cd.to(get_device())
                        cacc, cf1 = train_eval_two_stage(
                            backbone=backbone_cd,
                            num_classes=num_classes,
                            support_records=src_support,
                            query_records=tgt_query,
                            device=get_device(),
                            train_cfg=cfg,
                            use_llrd=ab["use_llrd"],
                            use_mixcut=ab["use_mixcut"],
                            rand_mag=ab["rand_mag"],
                        )
                        row["cross_domain_acc"] = cacc
                        row["cross_domain_macro_f1"] = cf1
                    else:
                        row["cross_domain_acc"] = np.nan
                        row["cross_domain_macro_f1"] = np.nan

                    rows.append(row)
                    pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    save_result_tables(df, out_dir, "ablation_seed_results")

    summary_rows = []
    for keys, g in df.groupby(["fraction", "model_size", "setting"]):
        fraction, model_size, setting = keys
        summary_rows.append(
            {
                "fraction": fraction,
                "model_size": model_size,
                "setting": setting,
                "acc": summarize_metric(g["acc"].values),
                "macro_f1": summarize_metric(g["macro_f1"].values),
                "noise_acc": summarize_metric(g["noise_acc"].values),
                "noise_macro_f1": summarize_metric(g["noise_macro_f1"].values),
                "cross_domain_acc": summarize_metric(g["cross_domain_acc"].dropna().values) if g["cross_domain_acc"].notna().any() else "N/A",
                "cross_domain_macro_f1": summarize_metric(g["cross_domain_macro_f1"].dropna().values) if g["cross_domain_macro_f1"].notna().any() else "N/A",
            }
        )

    summary = pd.DataFrame(summary_rows)
    save_result_tables(summary, out_dir, "ablation_summary")

    latex_path = os.path.join(out_dir, "ablation_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False, escape=False))

    logger.info(f"Done. LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
