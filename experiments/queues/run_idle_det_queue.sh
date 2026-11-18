#!/usr/bin/env bash
# 空闲 GPU 上的确定性复测队列：
#   1) 等 phm_capacity（容量对齐 110 次运行）结束；
#   2) 在独占 GPU 上跑 det_ctl3（同配置各两次：有/无 SSM，共 4 次运行）。
# 目的：判定"同配置同种子结果不同"是否来自共享 GPU（cuDNN 算法选择随占用变化）。
set +u
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"

echo "[idle-queue] armed $(date +%F_%T)"
while pgrep -f "run_stage.py phm_capacity" > /dev/null; do sleep 60; done
echo "[idle-queue] capacity done, GPU idle $(date +%F_%T)"
sleep 30
python -u run_stage.py det_ctl3
echo "[idle-queue] det_ctl3 done $(date +%F_%T)"
python -u check_det_ctl.py
echo "[idle-queue] ALL DONE $(date +%F_%T)"
