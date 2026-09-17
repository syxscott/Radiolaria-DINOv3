#!/usr/bin/env python3
"""在缓存特征上评估全部推理变体 (分辨率 × TTA × 特征类型 × 分类器)。

分类器: proto / lp / knn (inductive), labelprop / sinkhorn-PT (transductive)。
输出长表 CSV + official_r224/cls 对已发表基线的自校验。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (  # noqa: E402
    build_radiolaria_index,
    run_knn_classifier,
    run_prototype_classifier,
    select_support_query,
    topk_accuracy_from_scores,
)

K_SHOTS = [1, 3, 5, 10]
SEEDS = [42, 123, 456, 789, 999]


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def lp_predict(support_feats, support_labels, query_feats, num_classes,
               alpha=0.99, n_iters=15, knn=20):
    # GPU 加速版 (torch): 全图 4215 节点的传播在 CPU 上过慢
    import torch
    X = np.concatenate([query_feats, support_feats], 0).astype(np.float32)
    n, nq = len(X), len(query_feats)
    knn = max(1, min(knn, n - 1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X).to(dev)
    sims = Xt @ Xt.T
    sims.fill_diagonal_(-1.0)
    idx = sims.topk(knn, dim=1).indices
    W = torch.zeros_like(sims)
    W.scatter_(1, idx, torch.clamp(sims.gather(1, idx), min=0))
    W = (W + W.T) / 2
    D = W.sum(1).clamp(min=1e-9)
    S = W / D[:, None]
    Y = torch.zeros((n, num_classes), device=dev)
    Y[torch.arange(nq, n, device=dev),
      torch.as_tensor(support_labels, device=dev)] = 1
    F = Y.clone()
    for _ in range(n_iters):
        F = alpha * (S @ F) + (1 - alpha) * Y
    return F[:nq].argmax(1).cpu().numpy(), F[:nq].cpu().numpy()


def sinkhorn_pt(support_feats, support_labels, query_feats, num_classes,
                tau=0.05, n_iter=15):
    classes = np.unique(support_labels)
    protos = np.stack([
        support_feats[support_labels == c].mean(0) for c in classes
    ])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    logits = (query_feats @ protos.T) / tau
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum()
    nq = len(query_feats)
    for _ in range(n_iter):
        P /= P.sum(axis=0, keepdims=True) + 1e-9
        P *= nq / (P.sum(axis=1, keepdims=True) + 1e-9)
    return classes[P.argmax(1)]


def evaluate_variant(support_feats, support_labels, query_feats, query_labels,
                     num_classes, seed):
    out = {}
    acc, f1, t3, t5, _ = run_prototype_classifier(support_feats, support_labels,
                                                  query_feats, query_labels)
    out.update({"proto_acc": acc, "proto_macro_f1": f1})
    knn_acc, knn_f1, _ = run_knn_classifier(support_feats, support_labels,
                                            query_feats, query_labels, k=1)
    out.update({"knn_acc": knn_acc, "knn_macro_f1": knn_f1})
    lp = LogisticRegression(random_state=seed, C=1.0, solver="lbfgs",
                            max_iter=1000, n_jobs=1)
    lp.fit(support_feats, support_labels)
    lp_pred = lp.predict(query_feats)
    out["lp_acc"] = float(np.mean(lp_pred == query_labels) * 100.0)
    out["lp_macro_f1"] = float(f1_score(query_labels, lp_pred, average="macro"))
    lp_p, _ = lp_predict(support_feats, support_labels, query_feats, num_classes)
    out["labelprop_acc"] = float(np.mean(lp_p == query_labels) * 100.0)
    out["labelprop_macro_f1"] = float(f1_score(query_labels, lp_p, average="macro"))
    pt_p = sinkhorn_pt(support_feats, support_labels, query_feats, num_classes)
    out["ptmap_acc"] = float(np.mean(pt_p == query_labels) * 100.0)
    out["ptmap_macro_f1"] = float(f1_score(query_labels, pt_p, average="macro"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--tags", required=True,
                    help="逗号分隔的 cache tag 列表, 如 official,tapt-nogram,supcon")
    ap.add_argument("--feats", default="cls,patchmean,cat",
                    help="逗号分隔的特征类型子集")
    ap.add_argument("--out", default="results/testbed/eval_matrix_long.csv")
    args = ap.parse_args()

    records, class_to_idx, _ = build_radiolaria_index(args.data_root)
    num_classes = len(class_to_idx)
    print(f"records={len(records)} classes={num_classes}")

    rows = []
    seed_rows = []
    for tag in args.tags.split(","):
        tag = tag.strip()
        if not tag:
            continue
        feat_types = ["cls"] if tag == "supcon" else [f for f in args.feats.split(",") if f]
        for cache_dir in sorted(Path("features").glob(f"{tag}_r*")):
            if not (cache_dir / "cls.npy").exists():
                print(f"[eval] skip {cache_dir.name} (no cls.npy)")
                continue
            cls = np.load(cache_dir / "cls.npy")
            pm = np.load(cache_dir / "patchmean.npy")
            labels = np.load(cache_dir / "labels.npy")
            res = cache_dir.name.split("_r")[1]
            feats = {"cls": norm(cls), "patchmean": norm(pm), "cat": norm(np.concatenate([cls, pm], 1))}
            for ftype in feat_types:
                F = feats[ftype]
                for k_shot in K_SHOTS:
                    per_seed = {c: [] for c in ["proto_acc", "proto_macro_f1", "knn_acc",
                                                "knn_macro_f1", "lp_acc", "lp_macro_f1",
                                                "labelprop_acc", "labelprop_macro_f1",
                                                "ptmap_acc", "ptmap_macro_f1"]}
                    for seed in SEEDS:
                        support_records, query_records = select_support_query(
                            records, num_classes, k_shot, seed)
                        path2idx = {r.path: i for i, r in enumerate(records)}
                        si = np.array([path2idx[r.path] for r in support_records])
                        qi = np.array([path2idx[r.path] for r in query_records])
                        assert (labels[si] == np.array([r.label for r in support_records])).all()
                        out = evaluate_variant(F[si], labels[si], F[qi], labels[qi],
                                               num_classes, seed)
                        for c, v in out.items():
                            per_seed[c].append(v)
                    for clf in ["proto", "knn", "lp", "labelprop", "ptmap"]:
                        rows.append({
                            "tag": tag, "res": res, "feat": ftype, "clf": clf,
                            "k_shot": k_shot,
                            "acc_mean": float(np.mean(per_seed[f"{clf}_acc"])),
                            "acc_std": float(np.std(per_seed[f"{clf}_acc"], ddof=1)),
                            "f1_mean": float(np.mean(per_seed[f"{clf}_macro_f1"])),
                            "f1_std": float(np.std(per_seed[f"{clf}_macro_f1"], ddof=1)),
                        })
                        for i, seed in enumerate(SEEDS):
                            seed_rows.append({
                                "tag": tag, "res": res, "feat": ftype, "clf": clf,
                                "k_shot": k_shot, "seed": seed,
                                "acc": per_seed[f"{clf}_acc"][i],
                                "macro_f1": per_seed[f"{clf}_macro_f1"][i],
                            })
            print(f"[eval] {tag} r{res} done", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    seed_out = args.out.replace(".csv", "_per_seed.csv")
    pd.DataFrame(seed_rows).to_csv(seed_out, index=False)
    print(f"saved {args.out} ({len(df)} rows) + {seed_out} ({len(seed_rows)} rows)")

    # 自校验: official r224 norm cls proto 应复现已发表基线
    v = df[(df.tag == "official") & (df.res == "224") & (df.feat == "cls") & (df.clf == "proto")]
    v = v.sort_values("k_shot")
    expected = {1: 33.66, 3: 51.49, 5: 58.24, 10: 63.53}
    print("\n=== 管线自校验 (cache vs 已发表 vits16 proto) ===")
    ok = True
    for _, r in v.iterrows():
        e = expected[int(r.k_shot)]
        flag = "OK" if abs(r.acc_mean - e) < 1.0 else "MISMATCH!"
        ok = ok and abs(r.acc_mean - e) < 1.0
        print(f"  k={int(r.k_shot)}: cache {r.acc_mean:.2f} vs published {e} [{flag}]")
    if not ok:
        print("!!! 自校验失败, 缓存管线与主流水线不一致, 结果不可信")

    # proto@k10 pivot 速览
    pv = df[(df.clf == "proto") & (df.k_shot == 10)].pivot_table(
        index=["tag", "res", "feat"], columns="clf", values="acc_mean")
    pv = pv.sort_values("proto", ascending=False)
    print("\n=== Proto @10-shot Top15 ===")
    print(pv.head(15).round(2).to_string())


if __name__ == "__main__":
    main()
