#!/bin/bash
# 链路脚本：等 PHM2009 的统一协议批**完全结束**后，自动启动 PU 的同一协议批。
#
# 用法（在独立终端常驻）:
#   bash tools/run_after_phm2009.sh
#
# 设计说明
# --------
# · 用 pgrep 等待 PHM2009 进程消失，而不是 sleep 固定时长——PHM2009 的
#   完成时间取决于 MRA_SNN/MLDG 等慢方法，无法预估。
# · PU 的胜负判据、种子、预算与 PHM2009 完全一致（见 docs/COMPARE_PROTOCOL.md），
#   只是 TASKS/数据不同，因此复用同一个运行器，不另写训练/测试代码。
# · 若 PHM2009 中途崩溃，本脚本会照常启动 PU（避免整夜空等）；崩溃信息在
#   logs/cmp_PHM2009_baselines.log 里可查，PU 结果不受影响。
set -u

ROOT=/home/fzg/Fault/AG-Mamba
CHAIN_LOG="$ROOT/logs/chain.log"
mkdir -p "$ROOT/logs"

echo "[chain] 等待 PHM2009 批结束 @ $(date '+%F %T')" >> "$CHAIN_LOG"

while pgrep -f 'run_compare.py --dataset PHM2009' > /dev/null 2>&1; do
    sleep 60
done

echo "[chain] PHM2009 已结束 @ $(date '+%F %T')，启动 PU 批" >> "$CHAIN_LOG"
bash "$ROOT/tools/run_compare_dataset.sh" PU >> "$CHAIN_LOG" 2>&1
echo "[chain] PU 批完成 @ $(date '+%F %T')" >> "$CHAIN_LOG"
