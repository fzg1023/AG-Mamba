#!/usr/bin/env bash
# 队列：对齐头消融（交付前端固定；w_mmd / w_adv 三组合 + 同批次对照）
# 用途：填补局限 L5 —— 交付配置的 MMD / 对抗项此前未在交付前端上消融。
# 预算：480 步探针协议（与表 6/14/15 同协议）；HUST T0/T4/T9 x 5 种子 x 4 臂 = 60 次。
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"
echo "[align-queue] armed $(date +%F_%T)"
python -u run_stage.py align_ablate_hust
echo "[align-queue] ALL DONE $(date +%F_%T)"
