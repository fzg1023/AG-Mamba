#!/usr/bin/env bash
# §6.6 噪声鲁棒性（SNR 网格）续跑 —— **与前端批次并行**，因此不使用
# logs/run_snr_grid.sh 里的"等 fr/frlap 结束"等待逻辑。
#
# 规模：SNRS = [None, 10, 5, 0, -5] × METHODS = [AGMamba, WIDAN, MRA_SNN]
#       × TASKS = [T0, T4, T9] × SEEDS = [0,1,2] = 27 个训练单元
# 产物：logs/_s66_snr.json（增量写）→ analysis/summary_snr.py →
#       docs/manuscript/_s66_snr.md → analysis/make_fig9_efficiency.py
#       升级 Fig. 9 的 (d) 面板（此前是声明性空面板）。
#
# 噪声口径：每个窗口加 AWGN，σ = 窗口 RMS / 10^(SNR/20)，
#          噪声 RNG 由 hash((tag, i, snr, seed)) 决定 → **与方法无关**，
#          三方法看到的是同一批噪声样本（可比）。
#
# 用：tmux new-session -d -s snr 'bash logs/resume_snr.sh'
set -u
cd /home/fzg/Fault/AG-Mamba || exit 1
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python

export AGM_FORCE_PY_SCAN=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export AGM_RESULTS=/home/fzg/Fault/AG-Mamba/results_probe_snr

LOG=logs/snr_grid.out
echo "=== [SNR] start $(date '+%F %T') ===" | tee -a "$LOG"
"$PY" -u experiments/run_snr_grid.py >> "$LOG" 2>&1
echo "=== [SNR] run_snr_grid exit=$? $(date '+%F %T') ===" | tee -a "$LOG"

echo "=== [SNR] rebuilding _s66_snr.md ===" | tee -a "$LOG"
"$PY" analysis/summary_snr.py >> "$LOG" 2>&1
echo "=== [SNR] rebuilding Fig. 9 (d) ===" | tee -a "$LOG"
"$PY" analysis/make_fig9_efficiency.py >> "$LOG" 2>&1
echo "=== [SNR] ALL DONE $(date '+%F %T') ===" | tee -a "$LOG"
