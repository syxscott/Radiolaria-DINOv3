# 实验原始数据说明

汇集 2026-09-13 ~ 09-16 全部实验的 CSV 数据。配套报告:
《实验报告_20260913-14_全量重跑与算法改进.md》《实验报告_20260915-16_ViT-L与四项收尾.md》

## few-shot 主实验 (proto / LP / kNN / FT, Top-1 %, 5 seeds)

| 文件 | 内容 |
|---|---|
| `few_shot_summary_baseline+tapt-gram.csv` | 主网格 24 行: fraction {0,100} × {small, smallplus, base} × K{1,3,5,10}, 含 proto_top3/top5、knn、lp、ft 全指标 |
| `few_shot_seed_results_baseline+tapt-gram.csv` | 上表的 120 行逐 seed 明细 |
| `few_shot_summary_tapt-nogram.csv` / `..._seed_results...` | 无 Gram TAPT 对照 (fraction=100, 12 行) |
| `few_shot_summary_vitl16-tapt-gram.csv` / `..._seed...` | ViT-L TAPT-Gram 下游 (20 组合) |
| `few_shot_summary_vitl16-tapt-nogram.csv` / `..._seed...` | ViT-L 无 Gram 下游 (20 组合) |

## 消融

| 文件 | 内容 |
|---|---|
| `ablation_summary_消融全套.csv` | 16 行: {0,100} × {small,base} × 4 设置 (LLRD/Mixup/Aug), 含 clean 与 noise 两列; **cross_domain 列为 NaN (已知问题)** |
| `ablation_seed_results_消融全套.csv` | 逐 seed 明细 |
| `few_shot_summary_消融-旋转增强.csv` | 180° 旋转增强 FT (5/10-shot, 负结果) |
| `few_shot_summary_消融-低学习率诊断.csv` | TAPT-FT 崩塌低学习率诊断 (无恢复) |

## 监督基线公平对比 (Table 5)

| 文件 | 内容 |
|---|---|
| `tv_benchmark_v2_修复BN后_summary.csv` / `..._seed_results...` | 7 个 BN backbone × K{1,3,5,10} (BN 修复 + bf16) |
| `tv_benchmark_v1_旧协议_summary.csv` | 旧协议 224px 结果 (含未被 v2 覆盖的 convnext/swin/tv-vit_b_16) |
| `testbed_监督基线同协议_评估矩阵.csv` | 10 模型 × {224,448}×TTA × proto/LP/kNN/PT-MAP (kNN 公平对比核心数据) |

## 推理因子矩阵 / 分辨率扫描 / 集成

| 文件 | 内容 |
|---|---|
| `testbed_vits16_推理因子矩阵.csv` (+per_seed) | vits16 × {224,336,448,560,672} × TTA × {proto,knn,lp,labelprop,ptmap} 全因子 |
| `testbed_推广vitb16-splus_评估矩阵.csv` (+per_seed) | vitb16/S+ × {official, TAPT-Gram, TAPT-无Gram} × {224,448}×TTA |
| `testbed_vitl16_评估矩阵.csv` (+per_seed) | vitl16 官方 × {224,448}×TTA |
| `testbed_supcon_评估矩阵.csv` | SupCon 适配特征评估 (无增益) |
| `testbed_分辨率扫描_评估矩阵.csv` | 合并后的完整分辨率-性能曲线 |
| `testbed_集成推广_评估矩阵.csv` | S+B / S+B+L 特征集成 × 全 shot × 3 分类器 |

## 诊断与统计

| 文件 | 内容 |
|---|---|
| `all_models_10shot_best.csv` | 跨模型 10-shot 最优推理配置总表 (kNN 71.56~75.07) |
| `vitl16_tapt_rows.csv` | ViT-L 的 Baseline/TAPT 行 (Table 2/3/4 的 L 数据) |
| `stats_significance_显著性检验.csv` | 525 组配对检验 (paired-t + Wilcoxon) |
| `weight_drift_权重漂移.csv` | TAPT 后权重漂移分组统计 (~100% 机理证据) |
| `audit_旧结果审计.csv` | 旧 200 组合的 resume/BN 影响审计 |

## 注意

- 所有 acc/f1 为 5 seeds 的 mean ± std (per_seed 文件含原始值, 供显著性复算)
- `fraction` 列: 0 = 官方权重 Baseline, 100 = TAPT
- 旧版 TAPT checkpoint (历史文件/) 的数字已被本目录新数据取代
