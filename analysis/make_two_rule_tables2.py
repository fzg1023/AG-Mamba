# -*- coding: utf-8 -*-
"""两口径（best-epoch=oracle / last-ten=固定预算末端）主表数值，并与手稿表 9/10 已发表
的 best-epoch 均值逐方法核对，确保日志来源与手稿一致。

日志来源规则与 summary_cross_all.py 完全相同：
  ours      : results_dgfdb/{DATA}_{T}_AGMamba_fc_{xg_std|xg_6x|base}_seed{k}.log
  ORIG 八法 : results_{pu|phm2009}/{T}_{M}_seed{k}.log
  BENCH 九法: results_dgfdb_tuned2 -> results_dgfdb_tuned -> results_dgfdb 中
              {DATA}_{T}_{M}_seed{k}.log
  WIDAN/UDTL: results_widan | results_udtl，文件名 {DATA}_{T}_seed{k}.log
输出：docs/manuscript/_two_rule_tables.md
"""
import os
import re

import numpy as np
# --- 结果路径：统一走 agmamba.paths ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
OUT = os.path.join(ROOT, 'docs/manuscript', '_two_rule_tables.md')
TASKS = ['T0', 'T1', 'T2', 'T3']
SEEDS = list(range(5))
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
PORT_DIRS = ['results']
ORIG = ['AGG', 'DANN', 'MMD', 'CORAL', 'Triplet_loss', 'Mixup', 'MLDG', 'DAEL']
BENCH = ['CNNC', 'ADN', 'IEDGNet', 'DGNIS', 'WhiteningNet', 'CCN', 'CCDG', 'CDDG', 'DDGFD']
DATA_DIR = {'PU': 'results', 'PHM2009': 'results'}
L = []


def series(path):
    if not os.path.exists(path) or os.path.getsize(path) < 500:
        return None
    with open(path, errors='ignore') as f:
        v = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return v or None


def methods_of(data):
    out = [('AG-Mamba ' + a, lambda t, s, a=a, d=data: _findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(d, t, a, s)))
        for a in ('xg_std', 'xg_6x')]
    out.append(('AG-Mamba base', lambda t, s, d=data: _findlog('{}_{}_AGMamba_fc_base_seed{}.log'.format(d, t, s))))
    for m in ORIG:
        out.append((m, lambda t, s, m=m, d=data: _findlog(
            '{}_{}_{}_seed{}.log'.format(d, t, m, s))))

    def bench_fn(t, s, m, d):
        return _findlog('{}_{}_{}_seed{}.log'.format(d, t, m, s))
    for m in BENCH:
        out.append((m + ' [bench]', lambda t, s, m=m, d=data: bench_fn(t, s, m, d)))
    for m in ('WIDAN', 'UDTL'):
        out.append((m, lambda t, s, m=m, d=data: _findlog(
            '{}_{}_{}_seed{}.log'.format(d, t, m, s))))
    return out


def collect(fn):
    res = {}
    for t in TASKS:
        for s in SEEDS:
            v = series(fn(t, s))
            res[(t, s)] = None if not v else (max(v), float(np.mean(v[-10:])))
    return res


def manuscript_means(name):
    """从 v0.2 手稿的表 9/10 中解析某一行的 Mean 值（用于交叉核对）。"""
    p = os.path.join(ROOT, 'docs/manuscript', 'AG-Mamba_MSSP_manuscript_v0.2.md')
    txt = open(p, encoding='utf-8').read()
    for line in txt.split('\n'):
        if line.startswith('|') and name in line.split('|')[1]:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            try:
                return float(cells[-1].replace('**', ''))
            except ValueError:
                return None
    return None


def build(data, title):
    L.append('\n## ' + title)
    L.append('| Method | ' + ' | '.join('{} best / last'.format(t) for t in TASKS) +
             ' | Best (pooled) | Last-ten (pooled) | Best − last | 手稿 Mean (核对) |')
    L.append('|' + '---|' * (len(TASKS) + 5))
    rows = []
    for label, fn in methods_of(data):
        c = collect(fn)
        bs = [c[k][0] for k in c if c[k]]
        ls = [c[k][1] for k in c if c[k]]
        if not bs:
            L.append('| {} | {} |'.format(label, ' | '.join(['–'] * len(TASKS))))
            rows.append((label, None, None, None))
            continue
        pbt, plt = [], []
        for t in TASKS:
            b = [c[(t, s)][0] for s in SEEDS if c[(t, s)]]
            l = [c[(t, s)][1] for s in SEEDS if c[(t, s)]]
            pbt.append(np.mean(b) if b else float('nan'))
            plt.append(np.mean(l) if l else float('nan'))
        key = label.replace(' [bench]', '')
        if data == 'PU':
            man = {'AG-Mamba xg_std': 'AG-Mamba (xg_std)', 'AG-Mamba xg_6x': 'AG-Mamba (xg_6x)',
                   'AG-Mamba base': 'AG-Mamba base', 'Triplet_loss': 'Triplet'}.get(key, key)
        else:
            man = {'AG-Mamba xg_std': 'AG-Mamba (xg_std)', 'AG-Mamba xg_6x': 'AG-Mamba (xg_6x)',
                   'AG-Mamba base': 'AG-Mamba base', 'Triplet_loss': 'Triplet'}.get(key, key)
        mm = manuscript_means(man)
        pbm, plm = np.mean(bs), np.mean(ls)
        flag = '' if mm is None else ('ok' if abs(mm - pbm) <= 0.06 else 'MISMATCH')
        L.append('| {} | {} | **{:.2f} ± {:.2f}** | **{:.2f} ± {:.2f}** | {:+.2f} | {} {} |'.format(
            label, ' | '.join('{:.2f} / {:.2f}'.format(a, b) for a, b in zip(pbt, plt)),
            pbm, np.std(bs, ddof=1), plm, np.std(ls, ddof=1), pbm - plm,
            '–' if mm is None else '{:.2f}'.format(mm), flag))
        rows.append((label, pbm, plm, mm))
    print('\n=== {} ==='.format(data))
    for label, b, l, mm in rows:
        if b is None:
            print('  {:24s} (no logs)'.format(label))
        else:
            print('  {:24s} best {:6.2f}  last-ten {:6.2f}  drop {:+.2f}   手稿 {:>6}  {}'.format(
                label, b, l, b - l, '–' if mm is None else '{:.2f}'.format(mm),
                '' if (mm is None or abs(mm - b) <= 0.06) else '<-- MISMATCH'))


build('PU', 'PU bearing, four tasks: best-epoch (oracle) vs last-ten (deployment rule)')
build('PHM2009', 'PHM 2009 gearbox, four tasks: best-epoch (oracle) vs last-ten')

# HUST：仅 6 个方法保留 10 任务日志
print('\n=== HUST ===')
L.append('\n## HUST ten tasks: best-epoch (oracle) vs last-ten (deployment rule), '
         'methods whose 10-task logs are retained')
L.append('| Method | ' + ' | '.join('T{}'.format(i) for i in range(10)) +
         ' | Best (pooled) | Last-ten (pooled) | Best − last |')
L.append('|' + '---|' * 13)
for label, arm in (('**AG-Mamba (delivered)**', 'AGMamba_fc_sc_steps6x'), ('WIDAN (480)', 'WIDAN'),
                   ('WIDAN (2,880, matched)', 'WIDAN_long'), ('DANN (480)', 'DANN'),
                   ('UDTL (480)', 'UDTL'), ('MRA-SNN (480)', 'MRA_SNN')):
    pbt, plt, bs, ls = [], [], [], []
    for i in range(10):
        b_, l_ = [], []
        for s in SEEDS:
            v = series(_findlog(
                'HUST_T{}_{}_seed{}.log'.format(i, arm, s)))
            if v:
                b_.append(max(v))
                l_.append(float(np.mean(v[-10:])))
        if b_:
            pbt.append(np.mean(b_))
            plt.append(np.mean(l_))
            bs.extend(b_)
            ls.extend(l_)
    bm, lm = np.mean(bs), np.mean(ls)
    L.append('| {} | {} | **{:.2f} ± {:.2f}** | **{:.2f} ± {:.2f}** | {:+.2f} |'.format(
        label, ' | '.join('{:.2f} / {:.2f}'.format(a, b) for a, b in zip(pbt, plt)),
        bm, np.std(bs, ddof=1), lm, np.std(ls, ddof=1), bm - lm))
    print('  {:24s} best {:6.2f}  last-ten {:6.2f}  drop {:+.2f}'.format(label, bm, lm, bm - lm))

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n[OK] ->', OUT)
