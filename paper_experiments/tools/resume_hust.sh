#!/bin/bash
# 断电续跑：HUST 批次的无人值守恢复脚本。
#
# 背景（2026-09-30 断电）
# ---------------------
#   HUST 批（1100 单元）在 18:01 跑 MRA_SNN/T1/seed3 时断电中断。
#   恢复时有两件事必须先处理，否则会"静默污染"最终的对比表：
#     ① 4 个方法（ADN/CCN/CDDG/DGNIS）因 `datasets_src` 硬编码 3 域而**全 50
#        单元崩溃**（stub 日志 177 B）；修复已在 09-30 06:02 落地，但当时
#        进程已导入旧模块，从未验证过。故先跑 7 个单元的验证。
#     ② 中断会留下"只有十几个评估点"的半截日志。run_compare 新增了完整性
#        闸门（要求满 EPOCHS 个点），半截日志会自动重跑。
#
# 本脚本：先验证修复 -> 验证通过才启动正式批次（run_compare 跳过已完成单元）。
set -u

ROOT=/home/fzg/Fault/AG-Mamba
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python
LOG="$ROOT/logs/resume_hust.log"

cd "$ROOT"
mkdir -p logs

echo "==========================================================" >> "$LOG"
echo "[resume] 开始 $(date '+%F %T')" >> "$LOG"

# ---- 1) 修复验证：4 个崩溃方法各 1 单元 + 3 个"安全"方法逐位等价复检
#        + DGNIS 三源域（PU）回归复检 ----
echo "[resume] 阶段 1/2：跑修复验证（8 单元）..." >> "$LOG"
"$PY" -u logs/_verify_fix.py > logs/_verify_fix.out 2>&1
echo "[resume] 验证输出（节选）：" >> "$LOG"
grep -E '^\[|^=====' logs/_verify_fix.out >> "$LOG" 2>&1 || true

if grep -q '\[FAIL\]' logs/_verify_fix.out; then
    echo "[resume] ✗ 验证**未通过**（存在 FAIL），已中止，未启动批次。" >> "$LOG"
    exit 1
fi
# 4 个崩溃方法必须全部 [ ok ] 且 pts=60
okp=$(grep -cE '^\[ ok \] +(ADN|CCN|CDDG|DGNIS) ' logs/_verify_fix.out || true)
if [ "$okp" != "4" ]; then
    echo "[resume] ✗ 4 个崩溃方法只有 $okp 个通过（应为 4），已中止。" >> "$LOG"
    exit 1
fi
echo "[resume] ✓ 验证通过" >> "$LOG"

# ---- 2) 启动正式批次（run_compare 自带跳过 + 完整性闸门）----
echo "[resume] 阶段 2/2：启动 HUST 批次（跳过已完成单元）..." >> "$LOG"
bash tools/run_compare_dataset.sh HUST >> "$LOG" 2>&1
echo "[resume] HUST 批次结束 $(date '+%F %T')" >> "$LOG"
