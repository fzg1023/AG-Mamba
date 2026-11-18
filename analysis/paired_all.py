# -*- coding: utf-8 -*-
"""补齐 AG vs DANN / UDTL 在两种口径下的配对检验（供报告与 §6.1）。"""
import os
import re
import numpy as np
from scipy import stats as st

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
RD = os.path.join(ROOT, 'results')
TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]
PAT = {
    'AG': 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log',
    'DANN': 'HUST_{t}_DANN_seed{s}.log',
    'UDTL': 'HUST_{t}_UDTL_seed{s}.log',
    'WIDAN': 'HUST_{t}_WIDAN_seed{s}.log',
    'WIDAN_long': 'HUST_{t}_WIDAN_long_seed{s}.log',
}


def series(p):
    if not os.path.exists(p):
        return []
    with open(p, errors='ignore') as f:
        return [float(x) * 100 for x in re.findall(r'tgt_acc:\s*([0-9.]+)',
                                                   f.read())]


D = {k: {} for k in PAT}
for k, pat in PAT.items():
    for t in TASKS:
        for s in SEEDS:
            v = series(_findlog(pat.format(t=t, s=s)))
            if len(v) >= 10:
                D[k][(t, s)] = (max(v), float(np.mean(v[-10:])))

for other in ('WIDAN', 'WIDAN_long', 'DANN', 'UDTL'):
    for idx, rule in ((0, 'best'), (1, 'last10')):
        keys = sorted(set(D['AG']) & set(D[other]))
        a = np.array([D['AG'][k][idx] for k in keys])
        b = np.array([D[other][k][idx] for k in keys])
        d = a - b
        p = float(st.ttest_rel(a, b).pvalue)
        try:
            pw = float(st.wilcoxon(a, b).pvalue)
        except Exception:                            # pragma: no cover
            pw = float('nan')
        print('AG vs {:11s} {:7s} n={:2d}  Δ={:+7.2f}  p={:.3g}  '
              'p_wilcoxon={:.3g}  wins={}/{}'.format(
                  other, rule, len(keys), d.mean(), p, pw,
                  int((d > 0).sum()), len(keys)))
