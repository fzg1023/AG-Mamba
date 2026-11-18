#!/usr/bin/env bash
# §6.6 SNR 噪声网格：等前端批次（tmux `fr` 与接力的 `frlap`）全部结束后自动起跑。
#
# 为什么需要等待：本网格要占用 GPU，而前端消融是**优先级更高**的批次（§6.3 的
# Table 6 还缺 fr_lap 与 fr_cap960）。两个跑完后再上，避免互相拖慢。
#
# 用法（在 tmux 里起，别用 nohup/setsid —— 见 README 的 WSL 限制）：
#   tmux new-session -d -s snr 'bash logs/run_snr_grid.sh > logs/snr_grid.out 2>&1'
set -u
cd "$(dirname "$0")/.." || exit 1

echo "[wait] start $(date '+%F %T')  waiting for tmux sessions fr / frlap ..."
while tmux has-session -t fr 2>/dev/null || tmux has-session -t frlap 2>/dev/null; do
  sleep 60
done
echo "[wait] frontend batches gone $(date '+%F %T')"

# 再确认没有 run_stage.py 在跑（防止 frlap 结束但后台仍有子进程）
for i in $(seq 1 30); do
  if ! pgrep -f "run_stage.py" > /dev/null 2>&1; then break; fi
  echo "[wait] run_stage.py still alive, sleeping 60s ($i)"
  sleep 60
done

sleep 20
echo "[run] SNR grid start $(date '+%F %T')"
AGM_FORCE_PY_SCAN=1 PYTHONUTF8=1 \
  /home/fzg/anaconda3/envs/DSwinTrack/bin/python -u experiments/run_snr_grid.py
rc=$?
echo "[run] SNR grid exit=$rc $(date '+%F %T')"
if [ "$rc" = "0" ]; then
  echo "[sum] rebuilding _s66_snr.md ..."
  /home/fzg/anaconda3/envs/DSwinTrack/bin/python analysis/summary_snr.py || true
fi
