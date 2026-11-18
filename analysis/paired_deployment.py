# -*- coding: utf-8 -*-
"""在“部署口径”（last-ten）下做配对检验：本方法 vs 最强基线/关键对照，PU 与 PHM2009。

配对单元 = (任务, 种子)，n = 20。同时给出 best-epoch 口径以便对比。
"""
import os
import re

import numpy as np
from scipy import stats as st
# --- 结果路径：统一走 agmamba.paths ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
TASKS = ['T0', 'T1', 'T2', 'T3']
SEEDS = list(range(5))
PORT_DIRS = ['results']


def series(p):
    if not os.path.exists(p) or os.path.getsize(p) < 500:
        return None
    with open(p, errors='ignore') as f:
        v = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return v or None


def both(p):
    v = series(p)
    if not v:
        return None
    return max(v), float(np.mean(v[-10:]))


def path_of(data, task, arm, seed):
    for cand in ('{}_{}_AGMamba_fc_{}_seed{}.log'.format(
                     data, task, arm, seed),
                 '{}_{}_{}_seed{}.log'.format(data, task, arm, seed)):
        p = _findlog(cand)
        if os.path.exists(p):
            return p
    return None


def collect(data, arm):
    out = {}
    for t in TASKS:
        for s in SEEDS:
            r = both(path_of(data, t, arm, s))
            if r:
                out[(t, s)] = r
    return out


def test(data, arm_a, arm_b, key):
    A, B = collect(data, arm_a), collect(data, arm_b)
    k = sorted(set(A) & set(B))
    a = np.array([A[i][key] for i in k])
    b = np.array([B[i][key] for i in k])
    d = a - b
    p = st.ttest_rel(a, b).pvalue
    print('{:6s} {:14s} - {:<12s} [{}] n={:2d}  Δ = {:+.2f}  p = {:.4f}  wins {}/{}'.format(
        data, arm_a, arm_b, key, len(k), d.mean(), p, int((d > 0).sum()), len(k)))


print('=== PU ===')
for b in ('WIDAN', 'Mixup', 'DANN', 'CORAL', 'base'):
    for key in ('best', 'last'):
        test('PU', 'xg_std', b, {'best': 0, 'last': 1}[key])
print('\n=== PHM2009 ===')
for a, b in (('xg_6x', 'WIDAN'), ('xg_std', 'WIDAN'), ('xg_6x', 'DANN'), ('xg_6x', 'base'),
             ('xg_std', 'base')):
    for key in ('best', 'last'):
        test('PHM2009', a, b, {'best': 0, 'last': 1}[key])
