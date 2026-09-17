#!/usr/bin/env bash
# 提升方法推广: kNN / PT-MAP / 448px+flipTTA 扩展到 vitb16 与 S+ (Baseline + Gram/无Gram TAPT)
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg
PY=/home/user/anaconda3/envs/CV/bin/python
LOG=logs/overnight/scaleup_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "############ SCALEUP START: $(date) ############"

extract () {
  local ckpt=$1 tag=$2 msize=$3
  $PY scripts/feature_cache.py --checkpoint "$ckpt" --tag "$tag" --model_size "$msize" \
    --resolutions 224,448 --flip_resolutions 224,448 \
    || echo "!!! [WARN] 缓存失败: $tag"
}

echo "=== [extract] vitb16 三权重: $(date) ==="
extract ./model/dinov3_vitb16_pretrain.pth                    vitb16-official     base
extract ./outputs/tapt_vitb16_100/teacher_checkpoint.pth      vitb16-tapt-gram    base
extract ./outputs/tapt_vitb16_100_nogram/teacher_checkpoint.pth vitb16-tapt-nogram base

echo "=== [extract] S+ 三权重: $(date) ==="
extract ./model/dinov3_vits16plus_pretrain.pth                splus-official      smallplus
extract ./outputs/tapt_vits16plus_100/teacher_checkpoint.pth  splus-tapt-gram     smallplus
extract ./outputs/tapt_vits16plus_100_nogram/teacher_checkpoint.pth splus-tapt-nogram smallplus

echo "=== [eval] 评估矩阵 (cls 特征, 6 tags × 4 configs): $(date) ==="
$PY scripts/eval_on_cache.py \
  --tags vitb16-official,vitb16-tapt-gram,vitb16-tapt-nogram,splus-official,splus-tapt-gram,splus-tapt-nogram \
  --feats cls \
  --out results/testbed/eval_matrix_scaleup.csv \
  || echo "!!! [FATAL] 评估矩阵失败"

echo "############ SCALEUP DONE: $(date) ############"
touch logs/overnight/SCALEUP_DONE
