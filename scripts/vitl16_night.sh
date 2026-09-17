#!/usr/bin/env bash
# 通宵主链 (9月14晚): A1 监督基线同协议评估 -> A2 ViT-L TAPT (Gram/无Gram) + 下游
#                     -> B3 显著性检验 + B4 三格式图表 -> 跨模型总表
# 训练数据尽可能多保存: ViT-L 的中间 checkpoint 全部保留 (不清理)
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg
PY=/home/user/anaconda3/envs/CV/bin/python
LOG=logs/overnight/vitl16_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

TV_MODELS="resnet18,resnet50,resnet101,densenet121,efficientnet_b0,efficientnet_b3,mobilenet_v3_large,convnext_tiny,swin_t,vit_b_16"

echo "############ OVERNIGHT VITL16 PIPELINE START: $(date) ############"

echo "=== [1/7] 监督基线特征缓存 (10 模型 × 224/448 × flipTTA): $(date) ==="
$PY scripts/feature_cache_tv.py --models "$TV_MODELS" \
  || echo "!!! [FATAL] TV 特征缓存失败"

echo "=== [2/7] 监督基线评估矩阵 (Table 5 同协议对比): $(date) ==="
$PY scripts/eval_on_cache.py --tags $(echo $TV_MODELS | tr ',' '\n' | sed 's/^/tv-/' | paste -sd, -) \
  --feats cls --out results/testbed/eval_matrix_tv.csv \
  || echo "!!! [FATAL] TV 评估失败"

echo "=== [3/7] ViT-L TAPT (Gram, 40000 步, 保留全部中间 checkpoint): $(date) ==="
DIR=outputs/tapt_vitl16_100
if [ -f "$DIR/teacher_checkpoint.pth" ]; then
  echo "[skip] 已完成"
else
  $PY core/train_ssl.py --output-dir "$DIR" \
    --config-file configs/pretrain/vitl16_domain_adapt.yaml \
    train.output_dir="$DIR" \
    train.dataset_path=./data/ssl_subsets/100/train \
    student.pretrained_weights=./model/dinov3_vitl16_pretrain.pth \
    gram.use_loss=true gram.loss_weight=1.0 gram.img_level=true \
    gram.ckpt=./model/gram_anchor_vitl16.pth gram.rep_update=false \
    crops.gram_teacher_crops_size=448 crops.gram_teacher_no_distortions=true \
    train.compile=true \
    || { echo "!!! [FATAL] ViT-L Gram TAPT 失败"; }
  LAST=$(ls "$DIR/ckpt" 2>/dev/null | sort -n | tail -1)
  [ -n "$LAST" ] && $PY scripts/convert_ckpt.py "$DIR/ckpt/$LAST" "$DIR/teacher_checkpoint.pth" \
    && echo "=== ViT-L Gram teacher 已转换 ==="
fi

echo "=== [4/7] ViT-L TAPT (无 Gram 对照): $(date) ==="
DIR=outputs/tapt_vitl16_100_nogram
if [ -f "$DIR/teacher_checkpoint.pth" ]; then
  echo "[skip] 已完成"
else
  $PY core/train_ssl.py --output-dir "$DIR" \
    --config-file configs/pretrain/vitl16_domain_adapt.yaml \
    train.output_dir="$DIR" \
    train.dataset_path=./data/ssl_subsets/100/train \
    student.pretrained_weights=./model/dinov3_vitl16_pretrain.pth \
    train.compile=true \
    || { echo "!!! [FATAL] ViT-L 无Gram TAPT 失败"; }
  LAST=$(ls "$DIR/ckpt" 2>/dev/null | sort -n | tail -1)
  [ -n "$LAST" ] && $PY scripts/convert_ckpt.py "$DIR/ckpt/$LAST" "$DIR/teacher_checkpoint.pth" \
    && echo "=== ViT-L 无Gram teacher 已转换 ==="
fi

echo "=== [5/7] ViT-L 下游 few-shot (Gram 20 组合): $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_tapt_l.yaml \
  || echo "!!! [WARN] L-Gram 下游失败"

echo "=== [6/7] ViT-L 下游 few-shot (无Gram 20 组合): $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_tapt_l_nogram.yaml \
  || echo "!!! [WARN] L-无Gram 下游失败"

echo "=== [7/7] 重生成逐 seed 数据 + 显著性检验 + 图表: $(date) ==="
$PY scripts/eval_on_cache.py --tags official,tapt-nogram,supcon_r224 --feats cls \
  --out results/testbed/eval_matrix_long.csv \
  || echo "!!! [WARN] vits16 矩阵重生成失败"
$PY scripts/eval_on_cache.py \
  --tags vitb16-official,vitb16-tapt-gram,vitb16-tapt-nogram,splus-official,splus-tapt-gram,splus-tapt-nogram \
  --feats cls --out results/testbed/eval_matrix_scaleup.csv \
  || echo "!!! [WARN] scaleup 矩阵重生成失败"
$PY scripts/eval_on_cache.py --tags vitl16-official --feats cls \
  --out results/testbed/eval_matrix_vitl16.csv \
  || echo "!!! [WARN] vitl16 矩阵重生成失败"
$PY scripts/summary_10shot.py || echo "!!! [WARN] 总表生成失败"
$PY scripts/stats_and_figures.py || echo "!!! [WARN] 统计与图表失败"

echo "############ OVERNIGHT VITL16 PIPELINE DONE: $(date) ############"
touch logs/overnight/VITL16_NIGHT_DONE
