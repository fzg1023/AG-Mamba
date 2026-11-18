#!/usr/bin/python
# -*- coding:utf-8 -*-
"""分析 v3 三变体消融（P0: SSM / P1: 液体机制）。

用法: python analyze_v3.py [--all]
  默认只分析已完成的格子（自动跳过缺失）。
"""
import os
import re
import sys

import numpy as np

from scipy import stats

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

OUT = 'results'
VAR = ['v3_full', 'v3_nossm', 'v3_noliquid']
TASKS = ['T0', 'T9']
NSEED = 10


def acc(p):
    if not os.path.exists(p) or os.path.getsize(p) < 500:
        return None
    v = []
    for line in open(p, 'r', errors='ignore'):
        m = re.search(r'tgt_acc:\s*([0-9.]+)', line)
        if m:
            v.append(float(m.group(1)))
    return max(v) * 100 if v else None


def paired(d, label):
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        print('{:34s}: 样本不足 (n={})'.format(label, n))
        return
    md = d.mean()
    sd = d.std(ddof=1)
    t = md / (sd / np.sqrt(n)) if sd > 1e-12 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
    flag = '  <<< 显著' if (np.isfinite(p) and p < 0.05) else ''
    print('{:34s}: Δ={:+.3f}  t={:6.2f}  p={:.4f}  {}+{}/{}{}'.format(
        label, md, t, p, int((d > 0).sum()), int((d < 0).sum()), n, flag))


def main():
    data = {}
    for t in TASKS:
        for v in VAR:
            vals = []
            for s in range(NSEED):
                vals.append(acc(_findlog('HUST_{}_AGMamba_fc_{}_seed{}.log'.format(t, v, s))))
            data[(t, v)] = vals

    print('=' * 88)
    print('v3 三变体消融: v3_full(SSM+液体) / v3_nossm(无SSM) / v3_noliquid(SSM+无液体)')
    print('=' * 88)
    for t in TASKS:
        print('\n--- HUST {} ---'.format(t))
        for v in VAR:
            a = [x for x in data[(t, v)] if x is not None]
            if a:
                print('  {:12s}: {:.2f} +- {:.2f}   n={}   {}'.format(
                    v, np.mean(a), np.std(a), len(a),
                    [round(x, 2) for x in a]))

    print()
    print('=' * 88)
    print('配对检验')
    print('=' * 88)
    for t in TASKS:
        print('\n[HUST {}]'.format(t))
        for x, y, lab in [('v3_full', 'v3_nossm', 'P0: full - nossm (SSM贡献)'),
                          ('v3_full', 'v3_noliquid',
                           'P1: full - noliquid (液体贡献)'),
                          ('v3_noliquid', 'v3_nossm',
                           'noliquid - nossm (标准Mamba vs 无时序)')]:
            d = [(a - b) for a, b in zip(data[(t, x)], data[(t, y)])
                 if a is not None and b is not None]
            paired(d, lab)

    # 合并两个任务
    common_tasks = [t for t in TASKS
                    if all(any(x is not None for x in data[(t, v)]) for v in VAR)
                    and all(len([x for x in data[(t, v)] if x is not None]) == NSEED
                            for v in VAR)]
    if len(common_tasks) > 1:
        print('\n' + '=' * 88)
        print('合并 {} 个任务（n={}）'.format(len(common_tasks),
                                            len(common_tasks) * NSEED))
        print('=' * 88)
        for x, y, lab in [('v3_full', 'v3_nossm', 'P0: full - nossm (SSM贡献)'),
                          ('v3_full', 'v3_noliquid',
                           'P1: full - noliquid (液体贡献)'),
                          ('v3_noliquid', 'v3_nossm',
                           'noliquid - nossm (标准Mamba vs 无时序)')]:
            d = []
            for t in common_tasks:
                d += [(a - b) for a, b in zip(data[(t, x)], data[(t, y)])
                      if a is not None and b is not None]
            paired(d, lab)

    print()
    print('=' * 88)
    print('判读')
    print('=' * 88)
    print('  P1 若 Δ≈0 且 p>0.05  -> 液体机制无独立贡献，SSM 收益来自标准 Mamba Δ')
    print('  P1 若 Δ>0 且 p<0.05  -> 液体机制有独立贡献')
    print('=' * 88)


if __name__ == '__main__':
    main()
