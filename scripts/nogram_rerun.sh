#!/usr/bin/env bash
# 无 Gram TAPT 对照实验: TAPT(S→B→S+, 不带任何 gram 参数) -> 权重转换
#   -> few-shot fraction=100 TAPT 行评估(结果在 results/few_shot_classifier_nogram/)
set -u
cd /home/user/shenyaxuan/Radiolaria-DINOv3
export MPLBACKEND=Agg
PY=/home/user/anaconda3/envs/CV/bin/python
mkdir -p logs/overnight
LOG=logs/overnight/nogram_$(date +%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

run_tapt () {
  local name=$1 cfg=$2
  local dir=outputs/tapt_${name}_100_nogram
  if [ -f "$dir/teacher_checkpoint.pth" ]; then
    echo "[skip] $dir/teacher_checkpoint.pth 已存在, 跳过"
    return 0
  fi
  echo "============================================================"
  echo "=== [TAPT-nogram] ${name} (40000 steps) 开始: $(date) ==="
  $PY core/train_ssl.py --output-dir "$dir" \
    --config-file "configs/pretrain/${cfg}_domain_adapt.yaml" \
    train.output_dir="$dir" \
    train.dataset_path=./data/ssl_subsets/100/train \
    student.pretrained_weights=./model/dinov3_${name}_pretrain.pth \
    train.compile=true
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! [FATAL] TAPT-nogram ${name} 训练失败 rc=$rc"
    return $rc
  fi
  local last_iter
  last_iter=$(ls "$dir/ckpt" 2>/dev/null | sort -n | tail -1)
  if [ -z "$last_iter" ]; then
    echo "!!! [FATAL] TAPT-nogram ${name} 未产生 checkpoint"
    return 1
  fi
  echo "=== [convert] ${name}: ckpt/$last_iter -> teacher_checkpoint.pth ==="
  $PY scripts/convert_ckpt.py "$dir/ckpt/$last_iter" "$dir/teacher_checkpoint.pth" || { echo "!!! [FATAL] ${name} 转换失败"; return 1; }
  find "$dir/ckpt" -maxdepth 1 -type d ! -name "$last_iter" ! -path "$dir/ckpt" -exec rm -rf {} + 2>/dev/null
  echo "=== [TAPT-nogram] ${name} 完成: $(date) ==="
}

echo "############ NOGRAM PIPELINE START: $(date) ############"

run_tapt vits16 vits16         || echo "!!! [WARN] vits16 nogram 失败"
run_tapt vitb16 vitb16         || echo "!!! [WARN] vitb16 nogram 失败"
run_tapt vits16plus vits16plus || echo "!!! [WARN] vits16plus nogram 失败"

echo "============================================================"
echo "=== [few-shot-nogram] fraction=100 TAPT 行 (60 组合) 开始: $(date) ==="
$PY scripts/few_shot_classifier.py --config configs/experiments/few_shot_classifier_nogram.yaml \
  || echo "!!! [FATAL] few-shot nogram 失败"

echo "############ NOGRAM PIPELINE DONE: $(date) ############"
touch logs/overnight/NOGRAM_DONE
