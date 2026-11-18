#!/bin/bash
# 统一协议（P-统一）对比启动器：任意数据集、全部算法。
#
#   bash tools/run_compare_dataset.sh PU            # 全部 21 方法 × 4 任务 × 5 种子
#   bash tools/run_compare_dataset.sh PU --plan     # 只看计划
#
# 说明
# ----
# · 训练与测试在**同一次运行**内完成：每个 epoch 评估一次（checkpoint_freq =
#   iters_per_epoch），固定产出 60 个评估点，供"末十轮均值"读数。因此不需要
#   单独的测试脚本，测试集精度已随训练逐轮记入日志。
# · `AGM_FORCE_PY_SCAN=1` 必开：官方 mamba CUDA 扫描内核反传含原子累加，
#   会使同一 (方法,任务,种子) 逐次漂移（实测 ~1.3 点）。纯 PyTorch 扫描
#   完全确定；只影响 AG-Mamba（基线不走 SSM）。详见 docs/COMPARE_PROTOCOL.md。
# · 协议、口径与变更记录见 docs/COMPARE_PROTOCOL.md。
set -u

DATASET="${1:?用法: bash tools/run_compare_dataset.sh <DATASET> [额外参数...]}"
shift || true

ROOT=/home/fzg/Fault/AG-Mamba
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python
LOG="$ROOT/logs/cmp_${DATASET}.log"

mkdir -p "$ROOT/logs"
cd "$ROOT"

echo "[run] dataset=$DATASET  start $(date '+%F %T')" | tee -a "$LOG"
AGM_FORCE_PY_SCAN=1 "$PY" -u experiments/run_compare.py \
    --dataset "$DATASET" --methods all "$@" 2>&1 | tee -a "$LOG"
echo "[run] dataset=$DATASET  done  $(date '+%F %T')" | tee -a "$LOG"
