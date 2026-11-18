#!/usr/bin/env bash
# 归因实验队列：等冒烟测试与 MRA-SNN 结束后，依次跑
#   (1) pu_alpha_capacity  —— PU 上弥散 α × 前端宽度 2×2（80 次运行）
#   (2) phm_mixed_factor   —— PHM2009 转速x负载协议 L*/M*（200 次运行）
# 串行执行以避免同一日志文件被并发写入（run_variants 会跳过已完成日志）。
set +u
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGM_ENV:-mambavision_rgbt}"

echo "[queue] armed $(date +%F_%T)"
while pgrep -f smoke_attribution.py > /dev/null; do sleep 30; done
echo "[queue] smoke done $(date +%F_%T)"
while pgrep -f "run_stage.py mra_hust" > /dev/null; do sleep 30; done
echo "[queue] mra done $(date +%F_%T)"

echo "[queue] STAGE 1 pu_alpha_capacity $(date +%F_%T)"
python -u run_stage.py pu_alpha_capacity
echo "[queue] STAGE 2 phm_mixed_factor $(date +%F_%T)"
python -u run_stage.py phm_mixed_factor
echo "[queue] ALL DONE $(date +%F_%T)"
