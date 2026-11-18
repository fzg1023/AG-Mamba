# -*- coding: utf-8 -*-
"""续：AG-Mamba 与两个预算档的 WIDAN 的逐任务双口径对比 + 非参数稳健性检验。

读取 analyze_best_vs_final.py 的结果文件，追加第 4、5 节。
"""
import os
import re
import numpy as np

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

try:
    from scipy import stats as st
except Exception:                                    # pragma: no cover
    st = None

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


def collect(pat):
    best, final = {}, {}
    for t in TASKS:
        for s in SEEDS:
            v = series(_findlog(pat.format(t=t, s=s)))
            if not v:
                continue
            best[(t, s)] = max(v) * 100.0
            final[(t, s)] = v[-1] * 100.0
    return best, final


D = {k: collect(p) for k, p in PAT.items()}

L = []
L.append('')
L.append('## 4. 逐任务对比：AG-Mamba / WIDAN(480 步) / WIDAN(2880 步)')
L.append('')
L.append('| 任务 | Hz | AG best | AG final | WIDAN best | WIDAN final | WIDAN_long best | WIDAN_long final |')
L.append('|---|---|---|---|---|---|---|---|')
for i, t in enumerate(TASKS):
    cells = []
    for name in ('AG', 'WIDAN', 'WIDAN_long'):
        for rule in (0, 1):
            d = D[name][rule]
            ks = sorted([k for k in d if k[0] == t])
            cells.append('{:.2f}'.format(np.mean([d[k] for k in ks]))
                         if ks else '–')
    L.append('| {} | {} | {} |'.format(t, SPEEDS[i], ' | '.join(cells)))
L.append('')

L.append('## 5. 稳健性：非参数检验与效应量（AG-Mamba vs 基线）')
L.append('')
L.append('| 对比 | 口径 | Δ | Cohen d(z) | t 检验 p | Wilcoxon p | 更优对数 |')
L.append('|---|---|---|---|---|---|---|')
for name in ('WIDAN', 'WIDAN_long'):
    for idx, tag in ((0, 'best'), (1, 'final')):
        a, b = D['AG'][idx], D[name][idx]
        keys = sorted(set(a) & set(b))
        x = np.array([a[k] for k in keys])
        y = np.array([b[k] for k in keys])
        d = x - y
        n = len(d)
        dz = d.mean() / d.std(ddof=1)
        pt = float(st.ttest_rel(x, y).pvalue)
        try:
            pw = float(st.wilcoxon(x, y).pvalue)
        except Exception:                            # pragma: no cover
            pw = float('nan')
        L.append('| AG vs {} | {} | {:+.2f} | {:.2f} | {:.4f} | {:.4f} | {}/{} |'.format(
            name, tag, d.mean(), dz, pt, pw, int((d > 0).sum()), n))
L.append('')
L.append('说明：Cohen d(z) = 配对差均值 / 配对差标准差（|d| < 0.2 视为可忽略，'
         '0.2–0.5 小效应，0.5–0.8 中等）。Wilcoxon 为符号秩检验（双侧）。')

L.append('')
L.append('## 6. 逐任务差值：AG-Mamba − WIDAN_long（同预算）')
L.append('')
L.append('| 任务 | Hz | Δbest | Δfinal |')
L.append('|---|---|---|---|')
for i, t in enumerate(TASKS):
    row = []
    for idx in (0, 1):
        a, b = D['AG'][idx], D['WIDAN_long'][idx]
        ka = sorted([k for k in a if k[0] == t])
        kb = sorted([k for k in b if k[0] == t])
        if ka and kb:
            row.append(np.mean([a[k] for k in ka])
                       - np.mean([b[k] for k in kb]))
        else:
            row.append(float('nan'))
    L.append('| {} | {} | {:+.2f} | {:+.2f} |'.format(t, SPEEDS[i], row[0], row[1]))
L.append('')
L.append('说明：Δ > 0 表示 AG-Mamba 更高。低转速任务（T0–T4、T7）上两者均已接近饱和，'
         'WIDAN_long 在小数点级别领先；三个最大膨胀任务 T5/T8/T9 上 AG-Mamba 领先 '
         '+1.3 ~ +1.8 点。')

with open(FILE, 'a', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\nappended to', FILE)
