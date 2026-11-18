#!/usr/bin/env bash
# 在 WSL 规范环境下重建全部图件（图件与记录的原始生成环境）
cd /home/fzg/Fault/AG-Mamba
PY=/home/fzg/anaconda3/envs/DSwinTrack/bin/python
for s in make_fig2_design_space make_fig3_arch make_fig4_filterbank make_fig5_cross \
         make_fig6_frontend make_fig7_audit make_fig8_dilation make_fig9_efficiency \
         make_fig10_hust; do
  echo "== $s"
  "$PY" analysis/$s.py >/dev/null 2>&1 || echo "FAIL $s"
done
echo ALLDONE
