#!/usr/bin/env python3
"""汇总所有模型的 10-shot 最优推理配置对比表。"""
import pandas as pd

main = pd.read_csv("results/testbed/eval_matrix_scaleup.csv")
rows = []
for tag in ["vitb16-official", "vitb16-tapt-gram", "vitb16-tapt-nogram",
            "splus-official", "splus-tapt-gram", "splus-tapt-nogram"]:
    v = main[(main.tag == tag) & (main.res == "448_fliptta") & (main.feat == "cls") & (main.k_shot == 10)]
    rows.append({"model": tag,
                 "proto": v[v.clf == "proto"].acc_mean.iloc[0],
                 "knn": v[v.clf == "knn"].acc_mean.iloc[0],
                 "ptmap": v[v.clf == "ptmap"].acc_mean.iloc[0]})

v16 = pd.read_csv("results/testbed/eval_matrix_long.csv")
v = v16[(v16.tag == "official") & (v16.res == "448_fliptta") & (v16.feat == "cls") & (v16.k_shot == 10)]
rows.append({"model": "vits16-official",
             "proto": v[v.clf == "proto"].acc_mean.iloc[0],
             "knn": v[v.clf == "knn"].acc_mean.iloc[0],
             "ptmap": v[v.clf == "ptmap"].acc_mean.iloc[0]})

l = pd.read_csv("results/testbed/eval_matrix_vitl16.csv")
v = l[(l.res == "224_fliptta") & (l.feat == "cls") & (l.k_shot == 10)]
rows.append({"model": "vitl16-official",
             "proto": v[v.clf == "proto"].acc_mean.iloc[0],
             "knn": v[v.clf == "knn"].acc_mean.iloc[0],
             "ptmap": v[v.clf == "ptmap"].acc_mean.iloc[0]})

t = pd.DataFrame(rows).set_index("model").round(2).sort_values("knn", ascending=False)
pd.set_option("display.width", 200)
print("=== 全模型 @10-shot (448_fliptta, vitl16 为 224_fliptta) ===")
print(t.to_string())
t.to_csv("results/testbed/all_models_10shot_best.csv", encoding="utf-8-sig")
print("\n已保存: results/testbed/all_models_10shot_best.csv")
print("参照: 论文旧全局最高 = vitb16 FT 67.19 | 旧 proto: S 63.49 / S+ 59.76 / B 64.26 / L 59.76")
