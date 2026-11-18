#!/usr/bin/env bash
# 容量对齐实验队列：等 nondet_ctl 结束后再跑 phm_capacity（避免 GPU 争用影响训练时长）。
#   phm_capacity = front_chan=960（≈3.85M，对 WIDAN 3.98M 为 0.97×）
#                 x PHM2009 12 任务（High/Low/双因素）x 5 种子 = 60 次运行
set +u
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"

echo "[cap-queue] armed $(date +%F_%T)"
while pgrep -f "run_stage.py nondet_ctl" > /dev/null; do sleep 30; done
echo "[cap-queue] nondet done $(date +%F_%T)"
python -u run_stage.py phm_capacity
echo "[cap-queue] ALL DONE $(date +%F_%T)"
