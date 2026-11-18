#!/bin/bash
# ---------------------------------------------------------------------------
# AGMamba 单点提醒钩子
#
# 用途：长批次（PU / HUST / UO）里只关心"我们方法出结果的那一刻"。
#       命中后立刻：
#         1) 打印含 AGMamba 的完整排行榜 + 与最强基线的差距（tee 到标记文件）
#         2) 退出 —— 退出的终端会被上层自动感知 = 收到提醒
#
# 用法: bash tools/watch_agmamba.sh PU HUST
#       POLL=60 bash tools/watch_agmamba.sh PU HUST    # 自定义轮询秒数(默认30)
#
# 退出的两种情况（都会结束终端，从而触发提醒）：
#   · 正常：所有指定数据集都捕获到 AGMamba 结果行
#   · 异常：某数据集已启动、但其 run_compare 进程已退出且始终没抓到 AGMamba 行
#           （写 logs/AGMAMBA_MISSING_<DS>.txt，退出码 3）
#
# 注意：本脚本不假设 AGMamba 在方法列表的什么位置，只按日志行匹配，
#       因此无论从字典顺序第几位执行都能正确命中。
# ---------------------------------------------------------------------------
ROOT=/home/fzg/Fault/AG-Mamba
cd "$ROOT" || exit 1

if [ $# -ge 1 ]; then DATASETS=("$@"); else
  echo "用法: bash tools/watch_agmamba.sh DS1 [DS2 ...]"; exit 2
fi
POLL=${POLL:-30}

declare -A REPORTED=()
left=${#DATASETS[@]}
mkdir -p "$ROOT/logs"

logs_of()  { ls -1 "$ROOT/logs/cmp_$1"*.log 2>/dev/null; }

# AGMamba 是否已出结果。两种运行方式要分别识别：
#   A) 与基线同批（--methods all，日志 logs/cmp_<DS>.log）
#      → 逐方法完成行 "[method done] AGMamba  用时 ..."（'.' 代 ']' 免转义）
#   B) AGMamba 单独成批（--methods AGMamba，日志 logs/cmp_<DS>_AGMamba.log）
#      → 单方法批次不打印逐方法行，只有整批结束行 "### 完成 done=40 ..."
agm_done() {
  local f
  for f in $(logs_of "$1"); do
    grep -q 'method done. AGMamba' "$f" && return 0
  done
  f="$ROOT/logs/cmp_$1_AGMamba.log"
  [ -s "$f" ] && grep -q '^### 完成' "$f" && return 0
  return 1
}

started()  { [ -s "$ROOT/logs/cmp_$1.log" ] \
             || [ -n "$(ls -1 "$ROOT/logs/cmp_$1"*AGMamba*.log 2>/dev/null)" ]; }

alive()    { pgrep -f "run_compare.py --dataset $1" >/dev/null 2>&1; }

# 打印排行榜（按 cross 降序，带名次）
board() {
  grep 'cross=' "$1" | sed 's/^[[:space:]]*//' \
  | awk '{c=""; for(i=1;i<=NF;i++) if($i~/^cross=/){c=$i; sub("cross=","",c)}
          printf "%9.2f\t%s\n", c+0, $0}' \
  | sort -g -r \
  | awk -F'\t' '{printf "  %2d  %7.2f  %s\n", NR, $1, $2}'
}

# 输出: AGMamba最优 最强基线 差距 最强基线名
gap() {
  awk '
    /cross=/ {
      c=""; for(i=1;i<=NF;i++) if($i~/^cross=/){c=$i; sub("cross=","",c)}
      c=c+0; name=$1
      if (name=="AGMamba") { if (c>ab) ab=c }
      else                 { if (c>bb) { bb=c; bn=name } }
    }
    END { printf "%.2f\t%.2f\t%.2f\t%s\n", ab+0, bb+0, ab-bb, bn }
  ' "$1"
}

report() {
  local ds=$1 prog="$ROOT/logs/progress_$1.txt"
  local ts out i ab bb g bn
  ts=$(date '+%F %T'); out="$ROOT/logs/AGMAMBA_DONE_$ds.txt"

  # 等 report_progress 把榜写盘（[method done] 之后紧接着就会写）
  for i in $(seq 1 20); do grep -q 'AGMamba' "$prog" 2>/dev/null && break; sleep 3; done

  read -r ab bb g bn < <(gap "$prog")

  {
    echo "=================================================================="
    echo "  ★ AGMamba 已出结果 —— 数据集 $ds —— $ts"
    echo "=================================================================="
    echo "  运行日志: $(ls -1 "$ROOT/logs/cmp_$ds"*.log 2>/dev/null | tr '\n' ' ')"
    echo "  进度榜单: $prog"
    echo
    echo "  AGMamba cross = $ab      最强基线 = $bb ($bn)      领先 = $g"
    echo
    echo "  ---------------- 完整排行榜（截至此刻） ----------------"
    board "$prog"
    echo "=================================================================="
  } | tee "$out"
  echo "[hook] 标记已写入 $out"
}

echo "[hook] 开始监控 AGMamba 结果：${DATASETS[*]}  (POLL=${POLL}s, 起始 $(date '+%F %T'))"
echo "[hook] 命中条件：logs/cmp_<DS>*.log 中出现 '[method done] AGMamba'"

MISSED=0
while :; do
  for ds in "${DATASETS[@]}"; do
    [ "${REPORTED[$ds]:-0}" = 1 ] && continue
    if agm_done "$ds"; then
      report "$ds"
      REPORTED[$ds]=1; left=$((left-1))
    elif started "$ds" && ! alive "$ds"; then
      {
        echo "[hook][警告] $ds 的 run_compare 进程已退出，但始终未捕获 AGMamba 结果行。"
        echo "            时间: $(date '+%F %T')"
        echo "            请检查: $(ls -1 "$ROOT/logs/cmp_$ds"*.log 2>/dev/null | tr '\n' ' ')"
      } | tee "$ROOT/logs/AGMAMBA_MISSING_$ds.txt"
      REPORTED[$ds]=1; left=$((left-1)); MISSED=1
    fi
  done
  [ "$left" -le 0 ] && break
  sleep "$POLL"
done

echo "[hook] 监控结束 @ $(date '+%F %T')"
[ "$MISSED" = 1 ] && exit 3
exit 0
