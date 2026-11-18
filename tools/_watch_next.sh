#!/bin/bash
# 等待"下一个方法完成"后打印最新榜单（临时脚本，任务结束后删除）。
# 用法: bash tools/_watch_next.sh
LOG=/home/fzg/Fault/AG-Mamba/logs/cmp_PHM2009_baselines.log
PROG=/home/fzg/Fault/AG-Mamba/logs/progress_PHM2009.txt

n=$(grep -c 'method done' "$LOG" 2>/dev/null || echo 0)
while [ "$(grep -c 'method done' "$LOG" 2>/dev/null || echo 0)" -le "$n" ]; do
  sleep 20
done
grep 'method done' "$LOG" | tail -1
echo
cat "$PROG"
