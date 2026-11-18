#!/usr/bin/env bash
# §6.3 前端设计消融（frontend_ablate_clean）
#   5 臂 × T0(20Hz)/T4(40Hz)/T9(80Hz) × 5 种子 = 75 单元，HUST 480 步
#   fr_rand / fr_mix / fr_narrow / fr_cap768 / fr_cap960
# 参照臂不重跑：交付配置的参照 = results_probe_align 的 aln_noall（= 主对比 pub）
cd /home/fzg/Fault/AG-Mamba || exit 1
export AGM_FORCE_PY_SCAN=1
export AGM_RESULTS=/home/fzg/Fault/AG-Mamba/results_probe_front
export PYTHONUTF8=1
exec /home/fzg/anaconda3/envs/DSwinTrack/bin/python -u experiments/run_stage.py frontend_ablate_clean
