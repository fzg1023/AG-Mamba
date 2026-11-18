# -*- coding: utf-8 -*-
"""第三种口径：最后 K 个检查点的平均目标精度（无预言选择、且抗单点抖动）。

对比 AG-Mamba（交付）与 WIDAN 两个预算档：best / final / last10 / last20。
输出追加到 docs/manuscript/final_vs_best_epoch.md 的第 7 节。
"""
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
FILE = os.path.join(ROOT, 'docs/manuscript', 'final_vs_best_epoch.md')

TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
PAT = {
    'AG': 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log',
    'WIDAN': 'HUST_{t}_WIDAN_seed{s}.log',
    'WIDAN_long': 'HUST_{t}_WIDAN_long_seed{s}.log',
}


def series(path):
    if not os.path.exists(path):
        return []
    with open(path, errors='ignore') as f:
        return [float(x) for x in re.findall(r'tgt_acc:\s*([0-9.]+)', f.read())]


def rules(v, k10=10, k20=20):
    if not v:
        return None
    a = np.array(v, float) * 100.0
    return {'best': a.max(), 'final': a[-1],
            'last10': a[-k10:].mean() if len(a) >= k10 else a.mean(),
            'last20': a[-k20:].mean() if len(a) >= k20 else a.mean()}


D = {}
for name, pat in PAT.items():
    per = {}
    for t in TASKS:
        for s in SEEDS:
            r = rules(series(_findlog(pat.format(t=t, s=s))))
            if r:
                per[(t, s)] = r
    D[name] = per

RULES = ['best', 'final', 'last10', 'last20']

L = []
L.append('')
L.append('## 7. 更公平的口径：末 K 个检查点均值（last10 / last20）')
L.append('')
L.append('| 方法 | 口径 | 全体 (n=50) | T0–T4,T6,T7 (易) | T5,T8,T9 (难) |')
L.append('|---|---|---|---|---|')
EASY = ['T0', 'T1', 'T2', 'T3', 'T4', 'T6', 'T7']
HARD = ['T5', 'T8', 'T9']
for name, per in D.items():
    for rule in RULES:
        vals = np.array([per[k][rule] for k in per])
        easy = np.array([per[k][rule] for k in per if k[0] in EASY])
        hard = np.array([per[k][rule] for k in per if k[0] in HARD])
        L.append('| {} | {} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} |'.format(
            name, rule, vals.mean(), vals.std(), easy.mean(), hard.mean()))
L.append('')

L.append('### 配对检验（AG-Mamba 对基线，按 (任务,种子) 配对）')
L.append('')
L.append('| 对比 | 口径 | Δ | t 检验 p | Wilcoxon p | 更优对数 |')
L.append('|---|---|---|---|---|---|')
for name in ('WIDAN', 'WIDAN_long'):
    for rule in RULES:
        keys = sorted(set(D['AG']) & set(D[name]))
        x = np.array([D['AG'][k][rule] for k in keys])
        y = np.array([D[name][k][rule] for k in keys])
        d = x - y
        pt = float(st.ttest_rel(x, y).pvalue)
        try:
            pw = float(st.wilcoxon(x, y).pvalue)
        except Exception:                            # pragma: no cover
            pw = float('nan')
        L.append('| AG vs {} | {} | {:+.2f} | {:.4f} | {:.4f} | {}/{} |'.format(
            name, rule, d.mean(), pt, pw, int((d > 0).sum()), len(d)))
L.append('')

L.append('### 逐任务 last10')
L.append('')
L.append('| 任务 | Hz | AG | WIDAN | WIDAN_long | Δ vs WIDAN_long |')
L.append('|---|---|---|---|---|---|')
for i, t in enumerate(TASKS):
    row = []
    for name in ('AG', 'WIDAN', 'WIDAN_long'):
        ks = sorted([k for k in D[name] if k[0] == t])
        row.append(np.mean([D[name][k]['last10'] for k in ks]))
    L.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} |'.format(
        t, SPEEDS[i], row[0], row[1], row[2], row[0] - row[2]))
L.append('')

with open(FILE, 'a', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))
