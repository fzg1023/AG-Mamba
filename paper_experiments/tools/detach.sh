#!/bin/bash
# 后台化启动器：把 <command...> 放进**新会话**里跑，脚本自身立即返回。
#
# 用途：在终端工具失联 / 需要"拔掉终端也不死"的场景下无人值守地跑长批次。
# 为什么要有这个脚本：直接在命令行里写 `&` 会被上层工具改写（实测 `&` → `;&`），
# 把后台化收进脚本文件即可完全规避。
#
#   bash tools/detach.sh <out.log> <cmd> [args...]
#
# 例:
#   bash tools/detach.sh logs/cmp_HUST_nohup.log \
#        /home/fzg/anaconda3/envs/DSwinTrack/bin/python -u experiments/run_compare.py \
#        --dataset HUST --methods all
set -u

LOG="${1:?用法: bash tools/detach.sh <out.log> <cmd> [args...]}"
shift

setsid nohup "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown 2>/dev/null || true
echo "[detach] pid=$PID log=$LOG"
