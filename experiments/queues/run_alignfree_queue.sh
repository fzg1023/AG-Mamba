#!/usr/bin/env bash
# 队列：无对齐头交付配置（AF6x）—— 重跑交付证据链
# 依据：align_ablate_hust（2026-09-27）显示 MMD/对抗项在交付前端上不承重
#       （峰值中性 +1.29 / 末十 −8.63 点）。
# 内容：① HUST 全 10 任务 × 5 种子（2880 步 + 余弦）= sc_steps6x 的无头对照
#       ② PU / PHM2009 各 4 任务 × 5 种子 × 2 预算臂（std / 6x）
# 臂名 af6x / af_std / af_6x —— 与旧日志（sc_steps6x / xg_*）不重名，旧数据保留。
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"
echo "[af-queue] armed $(date +%F_%T)"
echo "[af-queue] STAGE 1 af_final_hust $(date +%F_%T)"
python -u run_stage.py af_final_hust
echo "[af-queue] STAGE 2 af_cross_final $(date +%F_%T)"
python -u run_stage.py af_cross_final
echo "[af-queue] ALL DONE $(date +%F_%T)"
