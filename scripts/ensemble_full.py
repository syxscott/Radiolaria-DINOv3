#!/usr/bin/env python3
"""三 backbone 特征集成推广: S+B / S+B+L × k_shot {1,3,5,10} × {proto, 1-NN, PT-MAP}。

输出: results/testbed/eval_matrix_ensemble_full.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_utils import (  # noqa: E402
    build_radiolaria_index, run_knn_classifier, run_prototype_classifier,
    select_support_query,
)
from scripts.eval_on_cache import sinkhorn_pt, norm  # noqa: E402

CACHES = {
    "S": Path("features/official_r448_fliptta/cls.npy"),
    "B": Path("features/vitb16-official_r448_fliptta/cls.npy"),
    "L": Path("features/vitl16-official_r224_fliptta/cls.npy"),
}


def main():
    records, class_to_idx, _ = build_radiolaria_index("./data")
    num_c = len(class_to_idx)
    i2 = {r.path: i for i, r in enumerate(records)}
    labels = np.array([r.label for r in records])

    feats = {}
    for k, p in CACHES.items():
        if not p.exists():
            print(f"[ensemble] skip {k}: {p} 不存在")
            return
        feats[k] = norm(np.load(p))
    feats["L"] = norm(np.load(CACHES["L"]))  # L 用其最优配置 224_fliptta

    rows = []
    for combo, keys in [("S+B", ["S", "B"]), ("S+B+L", ["S", "B", "L"])]:
        F = norm(np.concatenate([feats[k] for k in keys], axis=1))
        for k_shot in [1, 3, 5, 10]:
            accs = {"proto": [], "knn": [], "ptmap": []}
            for seed in [42, 123, 456, 789, 999]:
                sup, qry = select_support_query(records, num_c, k_shot, seed)
                a = np.array([i2[r.path] for r in sup])
                b = np.array([i2[r.path] for r in qry])
                pa, _, _, _, _ = run_prototype_classifier(F[a], labels[a], F[b], labels[b])
                ka, _, _ = run_knn_classifier(F[a], labels[a], F[b], labels[b], k=1)
                pp = sinkhorn_pt(F[a], labels[a], F[b], labels[b], num_c)
                pa_pt = float(np.mean(pp == labels[b]) * 100.0)
                accs["proto"].append(pa)
                accs["knn"].append(ka)
                accs["ptmap"].append(pa_pt)
            rows.append({"ensemble": combo, "k_shot": k_shot,
                         **{c: round(float(np.mean(v)), 2) for c, v in accs.items()}})
            print(f"{combo} k={k_shot}: " + " ".join(f"{c}={np.mean(v):.2f}" for c, v in accs.items()),
                  flush=True)

    df = pd.DataFrame(rows)
    out = "results/testbed/eval_matrix_ensemble_full.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
