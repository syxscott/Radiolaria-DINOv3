#!/usr/bin/env bash
# Overnight full re-run pipeline (single RTX 4090)
# 串行阶段: TAPT+Gram(S) -> TAPT+Gram(B) -> TAPT+Gram(S+) -> 权重转换
#           -> few-shot 全网格(Table 3/4) -> benchmark 7 BN backbones(Table 5)
# 每阶段独立容错: 单阶段失败不阻塞后续阶段, 但会在日志中标记 FATAL。
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg   # 无显示环境, matplotlib 必须用非交互后端
PY=/home/user/anaconda3/envs/CV/bin/python
mkdir -p logs/overnight
LOG=logs/overnight/run_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

GRAM="gram.use_loss=true gram.loss_weight=1.0 gram.img_level=true gram.rep_update=false crops.gram_teacher_crops_size=448 crops.gram_teacher_no_distortions=true"

run_tapt () {
  local name=$1 cfg=$2
  local dir=outputs/tapt_${name}_100
  if [ -f "$dir/teacher_checkpoint.pth" ]; then
    echo "[skip] $dir/teacher_checkpoint.pth 已存在, 跳过该 TAPT"
    return 0
  fi
  echo "============================================================"
  echo "=== [TAPT] ${name} (Gram anchoring, 40000 steps) 开始: $(date) ==="
  $PY core/train_ssl.py --output-dir "$dir" \
    --config-file "configs/pretrain/${cfg}_domain_adapt.yaml" \
    train.output_dir="$dir" \
    train.dataset_path=./data/ssl_subsets/100/train \
    student.pretrained_weights=./model/dinov3_${name}_pretrain.pth \
    gram.ckpt=./model/gram_anchor_${name}.pth \
    $GRAM train.compile=true
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! [FATAL] TAPT ${name} 训练失败 rc=$rc"
    return $rc
  fi
  local last_iter
  last_iter=$(ls "$dir/ckpt" 2>/dev/null | sort -n | tail -1)
  if [ -z "$last_iter" ]; then
    echo "!!! [FATAL] TAPT ${name} 未产生 checkpoint"
    return 1
  fi
  echo "=== [convert] ${name}: ckpt/$last_iter -> teacher_checkpoint.pth ==="
  $PY scripts/convert_ckpt.py "$dir/ckpt/$last_iter" "$dir/teacher_checkpoint.pth"
  if [ $? -ne 0 ] || [ ! -f "$dir/teacher_checkpoint.pth" ]; then
    echo "!!! [FATAL] ${name} 权重转换失败"
    return 1
  fi
  # 释放中间 checkpoint, 只保留最后一个
  find "$dir/ckpt" -maxdepth 1 -type d ! -name "$last_iter" ! -path "$dir/ckpt" -exec rm -rf {} + 2>/dev/null
  echo "=== [TAPT] ${name} 完成: $(date) ==="
}

echo "############ OVERNIGHT PIPELINE START: $(date) ############"

run_tapt vits16 vits16      || echo "!!! [WARN] vits16 TAPT 失败, 其下游 few-shot 将跳过 fraction=100/small"
run_tapt vitb16 vitb16      || echo "!!! [WARN] vitb16 TAPT 失败, 其下游 few-shot 将跳过 fraction=100/base"
run_tapt vits16plus vits16plus || echo "!!! [WARN] vits16plus TAPT 失败, 其下游 few-shot 将跳过 fraction=100/smallplus"

echo "============================================================"
echo "=== [few-shot] 全网格 (S/S+/B × 0/100 × k1/3/5/10 × 5seeds) 开始: $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_classifier.yaml --run_full_grid \
  || echo "!!! [FATAL] few-shot 网格失败"

echo "============================================================"
echo "=== [benchmark] 7 BN backbones -> v2 (Table 5) 开始: $(date) ==="
$PY torchvision_fewshot_benchmark.py \
  --models resnet18 resnet50 resnet101 densenet121 efficientnet_b0 efficientnet_b3 mobilenet_v3_large \
  --experiment_name torchvision_fewshot_benchmark_full_v2 \
  --no_save_checkpoints --no_save_features --no_save_episode_metadata \
  || echo "!!! [FATAL] benchmark 失败"

echo "############ OVERNIGHT PIPELINE DONE: $(date) ############"
touch logs/overnight/ALL_DONE
