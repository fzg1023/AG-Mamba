#!/usr/bin/env bash
# 队列：PU / PHM2009 四臂同批次对照（有对齐头 vs 无对齐头 × 两种预算）
# 目的：为"是否删掉对齐头"提供同批次配对证据（用户 2026-09-28 要求）。
# 内容：stage af_vs_ref_cross = PU 先跑、PHM2009 后跑；每数据集 4 任务 × 5 种子 × 4 臂。
# 臂名 xg2_* / af_* 与旧日志（xg_std / xg_6x、sc_steps6x）不重名。
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"
echo "[afref-queue] armed $(date +%F_%T)"
python -u run_stage.py af_vs_ref_cross
echo "[afref-queue] ALL DONE $(date +%F_%T)"
