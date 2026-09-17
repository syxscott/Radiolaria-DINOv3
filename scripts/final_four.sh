#!/usr/bin/env bash
# 四项收尾实验: 1 集成推广 -> 2 分辨率扫描(560/672) -> 3 消融全套 -> 4 监督基线 448px FT
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg
PY=/home/user/anaconda3/envs/CV/bin/python
LOG=logs/overnight/final_four_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "############ FINAL FOUR START: $(date) ############"

echo "=== [1/5] 集成推广 (S+B, S+B+L × 全 shot × 3 分类器): $(date) ==="
$PY scripts/ensemble_full.py || echo "!!! [WARN] 集成推广失败"

echo "=== [2/5] 分辨率扫描 560/672 (4 模型冻结特征): $(date) ==="
$PY scripts/feature_cache.py --checkpoint ./model/dinov3_vits16_pretrain.pth \
  --tag official --model_size small --resolutions 560,672 --flip_resolutions 560,672 \
  --batch_size 32 || echo "!!! [WARN] vits16 高分辨率缓存失败"
$PY scripts/feature_cache.py --checkpoint ./model/dinov3_vits16plus_pretrain.pth \
  --tag splus-official --model_size smallplus --resolutions 560,672 --flip_resolutions 560,672 \
  --batch_size 32 || echo "!!! [WARN] splus 高分辨率缓存失败"
$PY scripts/feature_cache.py --checkpoint ./model/dinov3_vitb16_pretrain.pth \
  --tag vitb16-official --model_size base --resolutions 560,672 --flip_resolutions 560,672 \
  --batch_size 16 || echo "!!! [WARN] vitb16 高分辨率缓存失败"
$PY scripts/feature_cache.py --checkpoint ./model/dinov3_vitl16_pretrain.pth \
  --tag vitl16-official --model_size large --resolutions 560,672 --flip_resolutions 560,672 \
  --batch_size 8 || echo "!!! [WARN] vitl16 高分辨率缓存失败"
$PY scripts/eval_on_cache.py \
  --tags official,splus-official,vitb16-official,vitl16-official \
  --feats cls --out results/testbed/eval_matrix_resolution.csv \
  || echo "!!! [WARN] 分辨率评估失败"

echo "=== [3/5] 消融全套 (LLRD/Mixup/增强 + 噪声 + cross-domain): $(date) ==="
$PY scripts/ablation_robustness.py --config configs/experiments/ablation_robustness.yaml \
  || echo "!!! [WARN] 消融实验失败"

echo "=== [4/5] 监督基线 448px FT (9 模型 × 5/10-shot): $(date) ==="
$PY torchvision_fewshot_benchmark.py --img_size 448 --batch_size 24 \
  --models resnet18 resnet50 resnet101 densenet121 efficientnet_b0 efficientnet_b3 mobilenet_v3_large convnext_tiny swin_t \
  --k_shots 5 10 --experiment_name tv_ft448 \
  --no_save_checkpoints --no_save_features --no_save_episode_metadata \
  || echo "!!! [WARN] 监督 448FT 失败"

echo "=== [5/5] 汇总: $(date) ==="
$PY scripts/summary_10shot.py || true
$PY scripts/stats_and_figures.py || true

echo "############ FINAL FOUR DONE: $(date) ############"
touch logs/overnight/FINAL_FOUR_DONE
