#!/usr/bin/env python3
"""Audit existing torchvision benchmark results for two known issues.

Issue 1 (resume bug): resuming from a "full_ft" checkpoint re-ran the whole
linear stage, producing training histories where linear epochs appear again
(duplicated / out-of-order stages). Detected from train_history.json.

Issue 2 (BN drift): during the linear stage the frozen backbone was flipped
back to train mode by model.train(), so BatchNorm backbones' running stats
drifted. This affects the FT metrics of every BN backbone regardless of
resuming, so all their runs are listed for a potential re-run.

Read-only: never modifies result files. Outputs a CSV inventory plus a
console summary with re-run time estimates (from metrics_detail.json).
"""

import argparse
import json
import os
from glob import glob

import pandas as pd

BN_BACKBONES = {
    "resnet18",
    "resnet50",
    "resnet101",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v3_large",
}


def audit_history(history_path):
    """Return (has_resume_anomaly, anomaly_desc) for one train_history.json."""
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        history = payload.get("history", [])
    except Exception as exc:
        return True, f"unreadable history: {exc}"

    linear_epochs = [e["epoch"] for e in history if e.get("stage") == "linear"]
    full_epochs = [e["epoch"] for e in history if e.get("stage") == "full_ft"]

    if not linear_epochs and not full_epochs:
        return False, ""

    # A clean run has linear = 1..N strictly increasing, then full_ft = 1..M.
    if linear_epochs != sorted(set(linear_epochs)) or (linear_epochs and linear_epochs[0] != 1):
        return True, f"linear epochs not 1..N: {linear_epochs[:15]}..."

    # Resume bug fingerprint: linear entries recorded again after full_ft began.
    last_linear_idx = max((i for i, e in enumerate(history) if e.get("stage") == "linear"), default=-1)
    first_full_idx = min((i for i, e in enumerate(history) if e.get("stage") == "full_ft"), default=len(history))
    if last_linear_idx > first_full_idx and full_epochs:
        return True, "linear entries appear after full_ft started (resume re-ran linear stage)"

    if full_epochs and full_epochs != sorted(set(full_epochs)):
        return True, f"full_ft epochs not 1..M: {full_epochs[:15]}..."

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark train histories")
    parser.add_argument("--results_root", default="results/torchvision_fewshot_benchmark_full")
    parser.add_argument("--output_csv", default="results/audit_resume_and_bn.csv")
    args = parser.parse_args()

    histories = sorted(glob(os.path.join(args.results_root, "*", "*", "seed_*", "train_history.json")))
    metrics = sorted(glob(os.path.join(args.results_root, "*", "*", "seed_*", "metrics_detail.json")))
    print(f"Found {len(histories)} train_history.json / {len(metrics)} metrics_detail.json")

    rows = []
    for hpath in histories:
        parts = hpath.split(os.sep)
        model_name, kshot, seed = parts[-4], parts[-3], parts[-2]
        anomaly, desc = audit_history(hpath)

        ft_acc = ft_f1 = elapsed = None
        mpath = os.path.join(os.path.dirname(hpath), "metrics_detail.json")
        if os.path.exists(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    m = json.load(f)
                ft_acc, ft_f1 = m.get("ft_acc"), m.get("ft_macro_f1")
                elapsed = m.get("elapsed_seconds")
            except Exception:
                pass

        is_bn = model_name in BN_BACKBONES
        rows.append(
            {
                "model_name": model_name,
                "k_shot": kshot,
                "seed": seed,
                "is_bn_backbone": is_bn,
                "bn_drift_affected": is_bn,  # all runs of BN backbones, independent of resume
                "resume_anomaly": anomaly,
                "resume_anomaly_desc": desc,
                "ft_acc": ft_acc,
                "ft_macro_f1": ft_f1,
                "elapsed_seconds": elapsed,
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    n_anomaly = int(df["resume_anomaly"].sum())
    n_bn = int(df["bn_drift_affected"].sum())
    print(f"\n=== 审计结果 ===")
    print(f"resume 重跑污染(需重跑): {n_anomaly} 个组合")
    if n_anomaly:
        print(df[df["resume_anomaly"]][["model_name", "k_shot", "seed", "resume_anomaly_desc"]].to_string(index=False))

    bn_df = df[df["bn_drift_affected"]]
    total_hours = bn_df["elapsed_seconds"].dropna().sum() / 3600
    print(f"\nBN 漂移影响(建议重跑): {n_bn} 个组合, 预计单卡耗时 {total_hours:.1f} 小时")
    print(f"非 BN backbone (transformer 类) 不受影响, 无需重跑")
    print(f"\n明细已保存: {args.output_csv}")


if __name__ == "__main__":
    main()
