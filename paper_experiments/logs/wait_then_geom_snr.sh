#!/usr/bin/env bash
# 串行链：等前端批次（tmux 'fr'）结束后 → 跑 §6.8 特征几何（t-SNE + 混淆矩阵）
#        → 再跑 §6.6 SNR 网格。
#
# 为什么串行：两条流并行会各自慢 ~2 倍（同抢一块 GPU），串行总墙钟更短、
# 资源占用减半。实测见 logs/resume_*.sh 注释。
#
# 为什么 §6.8 插在 SNR 之前：SNR 网格 26 格约 2.3 h，而特征几何只要 ~35 min
# （HUST 480 步 ×2 臂 + PU 2640 步 ×2 臂），先出图数据更早可用。
#
# 用：tmux new-session -d -s snrw 'bash logs/wait_then_geom_snr.sh'
set -u
cd /home/fzg/Fault/AG-Mamba || exit 1
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python

echo "[wait] $(date '+%F %T') 等 tmux 会话 'fr' 结束 ..."
while tmux has-session -t fr 2>/dev/null; do sleep 60; done
echo "[wait] 'fr' 已结束 $(date '+%F %T')"

# ---- §6.8 特征几何（t-SNE + 目标域混淆矩阵），完全隔离到 results_geom/weights_geom
LOG=logs/geom_tsne_cm.out
echo "=== [GEOM] §6.8 开始 $(date '+%F %T') ===" | tee -a "$LOG"
AGM_FORCE_PY_SCAN=1 PYTHONUTF8=1 \
  AGM_RESULTS=/home/fzg/Fault/AG-Mamba/results_geom \
  AGM_WEIGHTS=/home/fzg/Fault/AG-Mamba/weights_geom \
  "$PY" -u experiments/run_tsne_cm.py >> "$LOG" 2>&1
echo "=== [GEOM] 退出码 $? $(date '+%F %T') ===" | tee -a "$LOG"

# ---- §6.6 SNR 网格（自带 summary_snr.py + make_fig9_efficiency.py 收尾）
exec bash logs/resume_snr.sh
