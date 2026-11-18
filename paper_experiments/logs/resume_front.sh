#!/usr/bin/env bash
# §6.3 前端消融「补漏」续跑（v0.3 交付配置口径）
#
# 停点（2026-10-02 23:5x 作者要求全部停机）：
#   fr_rand / fr_mix / fr_narrow / fr_cap768  各 15/15 ✅
#   fr_cap960  11/15 + 1 个半截单元(T9/s1, 9 点)
#   fr_lap      0/15
#
# 本脚本跑 stage `frontend_ablate_clean`（含全部 6 臂），其 skip 判据是
#   `len(pts) >= 60`（experiments/run_stage.py:770），因此：
#   · 已完成的 60 个单元会被 `[skip]` 跳过；
#   · 9 点的半截单元**不会**被跳过 → 重跑，且 run_cell.run_one 用
#     `logging.FileHandler(logfile, mode='w')` 截断重写，不会与新结果叠加。
# 预计剩余 19 个单元（fr_cap960 × 4 + fr_lap × 15），~250 s/单元。
#
# 用：tmux new-session -d -s fr 'bash logs/resume_front.sh'
set -u
cd /home/fzg/Fault/AG-Mamba || exit 1
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python

export AGM_FORCE_PY_SCAN=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export AGM_RESULTS=/home/fzg/Fault/AG-Mamba/results_probe_front

LOG=logs/front_ablate_resume.log
echo "=== [FR] start $(date '+%F %T') ===" | tee -a "$LOG"
"$PY" -u experiments/run_stage.py frontend_ablate_clean >> "$LOG" 2>&1
echo "=== [FR] run_stage exit=$? $(date '+%F %T') ===" | tee -a "$LOG"

echo "=== [FR] rebuilding _s63_front.md ===" | tee -a "$LOG"
"$PY" analysis/summary_front_ablate.py >> "$LOG" 2>&1
echo "=== [FR] rebuilding Fig. 6 ===" | tee -a "$LOG"
"$PY" analysis/make_fig6_frontend.py >> "$LOG" 2>&1
echo "=== [FR] ALL DONE $(date '+%F %T') ===" | tee -a "$LOG"
