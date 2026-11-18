#!/bin/bash
# 数据集链路：等 <等待的数据集> 批**完全结束**后，自动启动 <下一个数据集> 批。
#
# 用法（各占一个常驻终端）:
#   bash tools/run_chain.sh PHM2009 PU     # PHM2009 -> PU
#   bash tools/run_chain.sh PU HUST        # PU -> HUST
#   bash tools/run_chain.sh HUST UO        # HUST -> UO（后续按需）
#
# 为什么用 pgrep 而不是估时
# ------------------------
# 各批的结束时间取决于最慢的方法（PHM2009 的 MRA_SNN 用了 137.8 min、CCN 117.7 min），
# 无法预估；pgrep 轮询进程真实存活状态，前一批一结束立刻接上，且不会早启动。
#
# 前置条件（务必先做，否则脚本不会代劳）
# ------------------------------------
# 下一个数据集的**旧结果必须先归档**：
#   python tools/archive_dataset.py --dataset <NEXT_DS> --apply
# 原因：旧日志中格式兼容的会被 run_compare 的 `[skip]` 复用，把不同协议的数字
# 混入新表（PHM2009 / PU / HUST 均已归档，见 docs/COMPARE_PROTOCOL.md §2.1）。
set -u

WAIT_DS="${1:?用法: bash tools/run_chain.sh <等待的数据集> <下一个数据集>}"
NEXT_DS="${2:?用法: bash tools/run_chain.sh <等待的数据集> <下一个数据集>}"

ROOT=/home/fzg/Fault/AG-Mamba
CHAIN_LOG="$ROOT/logs/chain.log"
mkdir -p "$ROOT/logs"

echo "[chain] 等待 ${WAIT_DS} 批结束 @ $(date '+%F %T')" >> "$CHAIN_LOG"
while pgrep -f "run_compare.py --dataset ${WAIT_DS}" > /dev/null 2>&1; do
    sleep 60
done

echo "[chain] ${WAIT_DS} 已结束 @ $(date '+%F %T')，启动 ${NEXT_DS} 批" >> "$CHAIN_LOG"
bash "$ROOT/tools/run_compare_dataset.sh" "$NEXT_DS" >> "$CHAIN_LOG" 2>&1
echo "[chain] ${NEXT_DS} 批完成 @ $(date '+%F %T')" >> "$CHAIN_LOG"
