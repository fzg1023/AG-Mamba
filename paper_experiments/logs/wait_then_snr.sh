#!/usr/bin/env bash
# 串行调度：等前端批次所在的 tmux 会话 'fr' 结束后，再跑 SNR 网格。
#
# 为什么串行：两条流都打同一块 GPU，实测各被拖慢 ~2 倍
#   （FR 首单元 501 s vs 独占 ~250 s；SNR 首格 625 s vs 独占 ~320 s）。
#   并行总墙钟 ≈ 277 min，串行 ≈ 75 + 139 = 214 min，且资源占用减半。
#
# SNR 结果按格增量写入 logs/_s66_snr.json，因此被中断后重跑会自动跳过已完成的格。
#
# 用：tmux new-session -d -s snrw 'bash logs/wait_then_snr.sh'
set -u
cd /home/fzg/Fault/AG-Mamba || exit 1

echo "[wait] $(date '+%F %T') 等 tmux 会话 'fr' 结束 ..."
while tmux has-session -t fr 2>/dev/null; do sleep 60; done
echo "[wait] 'fr' 已结束 $(date '+%F %T')，开始 SNR 网格"

exec bash logs/resume_snr.sh
