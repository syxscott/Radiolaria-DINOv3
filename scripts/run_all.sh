#!/usr/bin/env bash
set -euo pipefail

# 一键运行全部论文实验（请先检查 configs/experiments/*.yaml 中 checkpoint 路径）

python scripts/few_shot_classifier.py \
  --config configs/experiments/few_shot_classifier.yaml \
  --run_full_grid

python scripts/ablation_robustness.py \
  --config configs/experiments/ablation_robustness.yaml

python scripts/one_shot_extreme.py \
  --config configs/experiments/one_shot_extreme.yaml \
  --model_size base

python scripts/one_shot_extreme.py \
  --config configs/experiments/one_shot_extreme.yaml \
  --model_size large

echo "All experiments finished. Results are under ./results"
