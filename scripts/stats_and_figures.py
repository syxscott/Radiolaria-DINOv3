#!/usr/bin/env python3
"""试验田/推广结果的统计检验 + 论文图表汇总。

输出 (results/testbed/):
  stats_significance.csv            配对显著性检验 (5 seeds, paired-t + Wilcoxon)
  figures/fig*.png/pdf/svg          全部图表 (PDF/SVG 文字可编辑)
  eval_matrix_ensemble.csv          backbone 特征级集成结果
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import build_radiolaria_index  # noqa: E402

FIG = Path("results/testbed/figures")
FIG.mkdir(parents=True, exist_ok=True)


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def load_per_seed():
    frames = []
    for f in ["results/testbed/eval_matrix_long_per_seed.csv",
              "results/testbed/eval_matrix_scaleup_per_seed.csv",
              "results/testbed/eval_matrix_vitl16_per_seed.csv"]:
        if Path(f).exists():
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def significance(seed_df):
    """同 tag/res/feat/k_shot 内的配对比较 (按 seed 配对)。"""
    rows = []
    key_cols = ["tag", "res", "feat", "k_shot"]
    for keys, g in seed_df.groupby(key_cols):
        pivot = g.pivot_table(index="seed", columns="clf", values="acc")
        if "proto" not in pivot:
            continue
        for clf in ["knn", "ptmap", "lp", "labelprop"]:
            if clf not in pivot:
                continue
            a, b = pivot["proto"].values, pivot[clf].values
            if len(a) < 3 or np.allclose(a, b):
                continue
            t, pt = stats.ttest_rel(b, a)
            try:
                w, pw = stats.wilcoxon(b, a)
            except ValueError:
                pw = np.nan
            rows.append({**dict(zip(key_cols, keys)), "clf": clf,
                         "acc_diff": float(np.mean(b - a)),
                         "paired_t": float(t), "p_ttest": float(pt),
                         "p_wilcoxon": float(pw) if pw == pw else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv("results/testbed/stats_significance.csv", index=False)
    print(f"[stats] saved {len(df)} comparisons")
    return df


def fig_cross_model_bars(best10_csv):
    t = pd.read_csv(best10_csv)
    order = ["vits16-official", "splus-official", "vitb16-official", "vitl16-official"]
    t = t.loc[[m for m in order if m in t.index]]
    x = np.arange(len(t))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (col, label) in enumerate([("proto", "Prototype"), ("ptmap", "PT-MAP (transductive)"),
                                      ("knn", "1-NN")]):
        bars = ax.bar(x + (i - 1) * w, t[col], w, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, [m.replace("-official", "") for m in t.index])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Frozen DINOv3 features, 10-shot (448px + flip-TTA)")
    ax.legend()
    ax.set_ylim(0, 82)
    save(fig, "fig1_cross_model_10shot_bars")


def fig_methods_by_shot(seed_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, tag in zip(axes, ["vits16-official", "vitb16-official"]):
        v = seed_df[(seed_df.tag == tag) & (seed_df.res == "448_fliptta") & (seed_df.feat == "cls")]
        if v.empty:
            continue
        for clf, label in [("proto", "Prototype"), ("lp", "Linear probe"),
                           ("ptmap", "PT-MAP (transductive)"), ("knn", "1-NN")]:
            g = v[v.clf == clf].groupby("k_shot")["acc"].mean()
            ax.plot(g.index, g.values, marker="o", label=label)
        ax.set_title(tag.replace("-official", ""))
        ax.set_xlabel("K-shot")
        ax.set_xticks([1, 3, 5, 10])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend(fontsize=8)
    save(fig, "fig2_methods_by_shot")


def fig_confusion_knn(seed=42, k_shot=10):
    from scripts.exp_utils import select_support_query, run_knn_classifier
    from sklearn.metrics import confusion_matrix
    import seaborn as sns  # noqa: F401

    records, class_to_idx, idx_to_class = build_radiolaria_index("./data")
    num_classes = len(class_to_idx)
    cache = Path("features/official_r448_fliptta")
    F = norm(np.load(cache / "cls.npy"))
    labels = np.load(cache / "labels.npy")
    path2idx = {r.path: i for i, r in enumerate(records)}
    support_records, query_records = select_support_query(records, num_classes, k_shot, seed)
    si = np.array([path2idx[r.path] for r in support_records])
    qi = np.array([path2idx[r.path] for r in query_records])
    _, _, preds = run_knn_classifier(F[si], labels[si], F[qi], labels[qi], k=1)
    cm = confusion_matrix(labels[qi], preds, labels=list(range(num_classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(cm, cmap="mako", cbar=True, ax=ax,
                cbar_kws={"label": "Row-normalized proportion"})
    ax.set_xlabel("Predicted taxon")
    ax.set_ylabel("True taxon")
    ax.set_title(f"1-NN confusion matrix, ViT-S/16, {k_shot}-shot ({num_classes} taxa)")
    save(fig, "fig3_confusion_knn_10shot")


def fig_tapt_loss_curves():
    def series(path, key="total_loss"):
        dec = json.JSONDecoder()
        s = open(path).read()
        i, iters, vals = 0, [], []
        while i < len(s):
            s2 = s[i:].lstrip()
            if not s2:
                break
            i += len(s[i:]) - len(s2)
            obj, end = dec.raw_decode(s, i)
            i = end
            t = obj.get("Training", obj)
            if "iteration" in t and key in t:
                iters.append(t["iteration"])
                vals.append(t[key])
        return iters, vals

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"vits16": "-", "vitb16": "--", "vits16plus": "-."}
    for m, ls in styles.items():
        for variant, color in [("nogram", "tab:red"), ("gram", "tab:blue")]:
            p = Path(f"outputs/tapt_{m}_100{'_nogram' if variant == 'nogram' else ''}/training_metrics.json")
            if not p.exists():
                continue
            it, v = series(str(p))
            ax.plot(it, v, ls, color=color, alpha=0.85,
                    label=f"{m} ({variant})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")
    ax.set_title("TAPT training loss (blue: Gram anchoring, red: without)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig4_tapt_loss_curves")


def fig_weight_drift():
    import torch

    def load_sd(p):
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if "teacher" in sd and isinstance(sd.get("teacher"), dict):
            sd = sd["teacher"]
        if "backbone" in sd and isinstance(sd.get("backbone"), dict):
            sd = sd["backbone"]
        return sd

    official = load_sd("model/dinov3_vits16_pretrain.pth")
    groups = {
        "cls_token": ["cls_token"], "storage_tokens": ["storage_tokens"],
        "patch_embed": ["patch_embed.proj.weight", "patch_embed.proj.bias"],
        "LayerScale γ": [k for k in official if ".ls1.gamma" in k],
        "attn.qkv": [k for k in official if ".attn.qkv.weight" in k],
        "mlp": [k for k in official if ".mlp." in k],
        "norm": [k for k in official if ".norm" in k],
    }
    rows = []
    for tag, p in [("no-Gram", "outputs/tapt_vits16_100_nogram/teacher_checkpoint.pth"),
                   ("Gram", "outputs/tapt_vits16_100/teacher_checkpoint.pth")]:
        sd = load_sd(p)
        for g, keys in groups.items():
            keys = [k for k in keys if k in sd]
            if not keys:
                continue
            b = torch.cat([official[k].flatten().float() for k in keys]).norm().item()
            d = torch.cat([sd[k].flatten().float() - official[k].flatten().float()
                           for k in keys]).norm().item()
            rows.append({"group": g, "variant": tag, "drift_pct": 100 * d / max(b, 1e-9)})
    df = pd.DataFrame(rows)
    df.to_csv("results/testbed/weight_drift.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for variant, color in [("no-Gram", "tab:red"), ("Gram", "tab:blue")]:
        v = df[df.variant == variant]
        ax.bar(v.group + (" " if variant == "no-Gram" else "  "), v.drift_pct,
               width=0.38, color=color, label=f"TAPT ({variant})")
    ax.set_ylabel("Relative drift vs official weights (%)")
    ax.set_title("ViT-S/16 weight drift after TAPT (RoPE buffer unchanged: 0%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig5_weight_drift")


def ensemble_eval():
    """官方 S/B/L 特征拼接集成 (各模型最优配置) → proto / 1-NN。"""
    from scripts.exp_utils import run_knn_classifier, run_prototype_classifier, select_support_query

    records, num_c, _ = build_radiolaria_index("./data")
    caches = {
        "S": Path("features/official_r448_fliptta/cls.npy"),
        "B": Path("features/vitb16-official_r448_fliptta/cls.npy"),
        "L": Path("features/vitl16-official_r224_fliptta/cls.npy"),
    }
    feats = {}
    for k, p in caches.items():
        if p.exists():
            feats[k] = norm(np.load(p))
        else:
            print(f"[ensemble] skip {k} (cache missing: {p})")
    if "S" not in feats or "B" not in feats:
        print("[ensemble] need at least S+B caches")
        return
    labels = np.load("features/official_r448_fliptta/labels.npy")
    i2 = {r.path: i for i, r in enumerate(records)}
    combos = {"S+B": ["S", "B"]}
    if "L" in feats:
        combos["S+B+L"] = ["S", "B", "L"]
    rows = []
    for combo_name, keys in combos.items():
        F = norm(np.concatenate([feats[k] for k in keys], axis=1))
        for k_shot in [1, 5, 10]:
            accs_k, accs_p = [], []
            for seed in [42, 123, 456, 789, 999]:
                sup, qry = select_support_query(records, num_c, k_shot, seed)
                a = np.array([i2[r.path] for r in sup])
                b = np.array([i2[r.path] for r in qry])
                pa, _, _ = run_prototype_classifier(F[a], labels[a], F[b], labels[b])
                ka, _, _ = run_knn_classifier(F[a], labels[a], F[b], labels[b], k=1)
                accs_p.append(pa)
                accs_k.append(ka)
            rows.append({"ensemble": combo_name, "k_shot": k_shot,
                         "proto_acc": round(float(np.mean(accs_p)), 2),
                         "knn_acc": round(float(np.mean(accs_k)), 2)})
    df = pd.DataFrame(rows)
    df.to_csv("results/testbed/eval_matrix_ensemble.csv", index=False)
    print("[ensemble]\n" + df.to_string(index=False))


def main():
    seed_df = load_per_seed()
    if not seed_df.empty:
        significance(seed_df)
        fig_methods_by_shot(seed_df)
    if Path("results/testbed/all_models_10shot_best.csv").exists():
        fig_cross_model_bars("results/testbed/all_models_10shot_best.csv")
    try:
        fig_confusion_knn()
    except Exception as e:
        print(f"[fig3] failed: {e}")
    try:
        fig_tapt_loss_curves()
    except Exception as e:
        print(f"[fig4] failed: {e}")
    try:
        fig_weight_drift()
    except Exception as e:
        print(f"[fig5] failed: {e}")
    try:
        ensemble_eval()
    except Exception as e:
        print(f"[ensemble] failed: {e}")


if __name__ == "__main__":
    main()
