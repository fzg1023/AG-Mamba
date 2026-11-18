#!/usr/bin/env bash
# §6.3 追加臂 fr_lap（Laplace 首层）—— 等现有 frontend 批次结束后自动起跑。
#
# 为什么串行：同一张 GPU，避免两个训练进程互相抢显存/带宽而改变时延与数值路径。
# 为什么复用同一个 stage：`run_compare` 的 skip 逻辑按 (arm, task, seed) 逐单元跳过，
#   已完成的前 5 臂会被 `[skip]`，实际只会跑 fr_lap 的 15 个单元。
# 日志单独一份，便于区分两次运行的出处。
set -u
cd /home/fzg/Fault/AG-Mamba

LOG=logs/front_lap_clean.log
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python

echo "=== [watch] 等待 tmux 会话 'fr' 结束 @ $(date '+%F %T') ===" >> "$LOG"
while tmux has-session -t fr 2>/dev/null; do sleep 60; done

echo "=== [watch] 'fr' 已结束，开始跑 fr_lap @ $(date '+%F %T') ===" >> "$LOG"
AGM_FORCE_PY_SCAN=1 AGM_RESULTS=/home/fzg/Fault/AG-Mamba/results_probe_front \
  "$PY" -u experiments/run_stage.py frontend_ablate_clean >> "$LOG" 2>&1

echo "=== [watch] fr_lap 阶段结束 @ $(date '+%F %T') ===" >> "$LOG"
"$PY" analysis/summary_front_ablate.py >> "$LOG" 2>&1
echo "=== [watch] 摘要已刷新 _s63_front.md @ $(date '+%F %T') ===" >> "$LOG"
