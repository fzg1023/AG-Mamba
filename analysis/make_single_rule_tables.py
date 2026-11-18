# -*- coding: utf-8 -*-
"""单一报告口径（固定训练预算的末端读数 = 末十个检查点均值）下的全部表格与检验。

输出：docs/manuscript/_single_rule_tables.md
  (1) HUST 六方法（日志保留集）：逐任务 mean ± s.d. + 池化
  (2) PU 全方法（22 臂）
  (3) PHM2009 全方法（22 臂）
  (4) 配对检验：本方法 vs 关键基线（同口径）
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
OUT = os.path.join(ROOT, 'docs/manuscript', '_single_rule_tables.md')
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEEDS = list(range(5))
T4 = ['T0', 'T1', 'T2', 'T3']
PORT_DIRS = ['results']
ORIG = ['AGG', 'DANN', 'MMD', 'CORAL', 'Triplet_loss', 'Mixup', 'MLDG', 'DAEL']
BENCH = ['CNNC', 'ADN', 'IEDGNet', 'DGNIS', 'WhiteningNet', 'CCN', 'CCDG', 'CDDG', 'DDGFD']
DATA_DIR = {'PU': 'results', 'PHM2009': 'results'}
L = []


def last10(p):
    if p is None or not os.path.exists(p) or os.path.getsize(p) < 500:
        return None
    with open(p, errors='ignore') as f:
        v = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return float(np.mean(v[-10:])) if len(v) >= 10 else (float(np.mean(v)) if v else None)


def arms_of(data):
    out = [('AG-Mamba (xg_std)', 'results', '{d}_{t}_AGMamba_fc_xg_std_seed{s}.log'),
           ('AG-Mamba (xg_6x)', 'results', '{d}_{t}_AGMamba_fc_xg_6x_seed{s}.log'),
           ('AG-Mamba base', 'results', '{d}_{t}_AGMamba_fc_base_seed{s}.log')]
    for m in ORIG:
        out.append((m, DATA_DIR[data], '{t}_' + m + '_seed{s}.log'))
    for m in BENCH:
        out.append((m, None, m))
    for m, d in (('WIDAN', 'results'), ('UDTL', 'results')):
        out.append((m, d, data + '_{t}_seed{s}.log'))
    return out

def path_of(entry, data, task, seed):
    label, d, tmpl = entry
    cands = []
    if tmpl and '{' in tmpl:
        cands.append(tmpl.format(d=data, t=task, s=seed))
    cands.append('{}_{}_{}_seed{}.log'.format(data, task, label, seed))
    cands.append('{}_{}_seed{}.log'.format(task, label, seed))
    for cand in cands:
        p = _findlog(cand)
        if os.path.exists(p):
            return p
    return None


def row(entry, data, tasks):
    per = {}
    for t in tasks:
        v = [last10(path_of(entry, data, t, s)) for s in SEEDS]
        v = [x for x in v if x is not None]
        per[t] = (np.mean(v), np.std(v, ddof=0), len(v)) if v else None
    return per


def emit(title, data, tasks, entries):
    L.append('\n## ' + title)
    L.append('| Method | ' + ' | '.join(tasks) + ' | Mean |')
    L.append('|' + '---|' * (len(tasks) + 2))
    print('\n== {} =='.format(title))
    for e in entries:
        per = row(e, data, tasks)
        cells, allv = [], []
        for t in tasks:
            if per[t] is None:
                cells.append('–')
            else:
                cells.append('{:.2f} ± {:.2f}'.format(per[t][0], per[t][1]))
                allv.append(per[t][0])
        if not allv:
            print('  {:24s} (no logs)'.format(e[0]))
            L.append('| {} | {} |'.format(e[0], ' | '.join(cells)))
            continue
        pooled = []
        for t in tasks:
            pooled += [x for x in (last10(path_of(e, data, t, s)) for s in SEEDS) if x is not None]
        m, sd = np.mean(pooled), np.std(pooled, ddof=0)
        L.append('| {} | {} | **{:.2f}** |'.format(e[0], ' | '.join(cells), m))
        print('  {:24s} mean {:.2f}  (per-task sd {:.2f}-{:.2f})'.format(
            e[0], m, min(per[t][1] for t in tasks if per[t]), max(per[t][1] for t in tasks if per[t])))


# (1) HUST 六方法
L.append('# 单一口径（固定预算末端读数）')
hust = [('**AG-Mamba (delivered)**', 'results', 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log'),
        ('WIDAN (released budget)', 'results', 'HUST_{t}_WIDAN_seed{s}.log'),
        ('WIDAN (matched budget)', 'results', 'HUST_{t}_WIDAN_long_seed{s}.log'),
        ('DANN', 'results', 'HUST_{t}_DANN_seed{s}.log'),
        ('UDTL', 'results', 'HUST_{t}_UDTL_seed{s}.log'),
        ('MRA-SNN', 'results', 'HUST_{t}_MRA_SNN_seed{s}.log')]
emit('HUST ten tasks (logs retained)', 'HUST', ['T{}'.format(i) for i in range(10)], hust)

# (2)(3) PU / PHM2009 全方法
emit('PU bearing, four tasks', 'PU', T4, arms_of('PU'))
emit('PHM 2009 gearbox, four tasks', 'PHM2009', T4, arms_of('PHM2009'))

# (4) 配对检验（同一口径）
def pair(data, tasks, a, b):
    A, B = [], []
    for t in tasks:
        for s in SEEDS:
            x = last10(path_of(a, data, t, s))
            y = last10(path_of(b, data, t, s))
            if x is not None and y is not None:
                A.append(x)
                B.append(y)
    A, B = np.array(A), np.array(B)
    d = A - B
    return d.mean(), st.ttest_rel(A, B).pvalue, int((d > 0).sum()), len(A)


def lookup(label, data):
    for e in arms_of(data):
        if e[0] == label:
            return e
    return None


print('\n== paired tests (single rule) ==')
L.append('\n## Paired tests, single rule (n = 20 per benchmark)')
for data in ('PU', 'PHM2009'):
    for b in ('WIDAN', 'UDTL', 'DANN', 'Mixup', 'CORAL', 'AGG', 'MMD', 'MLDG', 'CCN', 'CDDG',
              'WhiteningNet', 'AG-Mamba base'):
        ea, eb = lookup('AG-Mamba (xg_std)', data), lookup(b, data)
        if ea is None or eb is None:
            continue
        m, p, w, n = pair(data, T4, ea, eb)
        L.append('| {}: AG-Mamba − {} | {:.2f} | {:.4f} | {}/{} |'.format(data, b, m, p, w, n))
        print('  {:8s} vs {:16s} Δ = {:+.2f}  p = {:.4f}  wins {}/{}'.format(data, b, m, p, w, n))
ea = lookup('AG-Mamba (xg_std)', 'PHM2009')
eb = lookup('AG-Mamba (xg_6x)', 'PHM2009')
print('  PHM 长预算臂 vs WIDAN:', pair('PHM2009', T4, eb, lookup('WIDAN', 'PHM2009')))

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n[OK] ->', OUT)
