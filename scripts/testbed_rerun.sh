#!/usr/bin/env bash
# vits16 试验田: 三梯队改进全因子实验
#  A: TAPT-FT 崩塌诊断(权重统计)
#  B: 特征缓存(官方×{224,336,448}×{TTA} + TAPT-nogram×224)
#  C: 评估矩阵(proto/LP/kNN + 转导式 labelprop/PT-map) 含管线自校验
#  D: 低学习率 FT 诊断
#  E: 旋转增强 FT
#  F: SupCon 监督适配 + 评估
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg
PY=/home/user/anaconda3/envs/CV/bin/python
mkdir -p logs/overnight
LOG=logs/overnight/testbed_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "############ VITS16 TESTBED START: $(date) ############"

echo "=== [A] TAPT-FT 崩塌诊断 (权重统计): $(date) ==="
$PY scripts/diagnose_tapt_weights.py || echo "!!! [WARN] 权重诊断失败"

echo "=== [B] 特征缓存提取: $(date) ==="
$PY scripts/feature_cache.py --checkpoint ./model/dinov3_vits16_pretrain.pth \
  --tag official --model_size small || echo "!!! [FATAL] official 缓存失败"
$PY scripts/feature_cache.py --checkpoint ./outputs/tapt_vits16_100_nogram/teacher_checkpoint.pth \
  --tag tapt-nogram --model_size small --resolutions 224 --flip_resolutions 224 \
  || echo "!!! [WARN] tapt-nogram 缓存失败"

echo "=== [D] 低学习率 FT 诊断 (10 组合): $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_nogram_lowlr.yaml \
  || echo "!!! [WARN] lowlr 诊断失败"

echo "=== [E] 旋转增强 FT (10 组合): $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_rot.yaml \
  || echo "!!! [WARN] 旋转增强失败"

echo "=== [F] SupCon 监督适配: $(date) ==="
$PY scripts/supcon_adapt.py || echo "!!! [WARN] SupCon 失败"

echo "=== [C] 评估矩阵 (含自校验 + 转导式): $(date) ==="
$PY scripts/eval_on_cache.py --tags official,tapt-nogram,supcon \
  || echo "!!! [FATAL] 评估矩阵失败"

echo "############ VITS16 TESTBED DONE: $(date) ############"
touch logs/overnight/TESTBED_DONE
