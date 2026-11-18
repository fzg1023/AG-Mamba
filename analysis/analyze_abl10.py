#!/usr/bin/python
# -*- coding:utf-8 -*-
"""配对检验：比较两个 AG-Mamba 变体的多数据集结果。

用法:
  python analyze_abl10.py                 # abl10_base vs abl10_const
  python analyze_abl10.py base film       # 任意两个变体名
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


def acc_of(path):
    if not os.path.exists(path):
        return None
    v = []
    for line in open(path, 'r', errors='ignore'):
        m = re.search(r'tgt_acc:\s*([0-9.]+)', line)
        if m:
            v.append(float(m.group(1)))
    return max(v) * 100 if v else None


def lm(data, task, name, seed):
    return acc_of(_findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(data, task, name, seed)))


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else 'abl10_base'
    var = sys.argv[2] if len(sys.argv) > 2 else 'abl10_const'
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    cases = [('HUST', 'T0'), ('PHM2009', 'T0'), ('PU', 'T0')]
    print('=' * 92)
    print('配对检验: {}  vs  {}   (种子 0..{})'.format(var, ref, nseed - 1))
    print('=' * 92)
    print('{:10s} {:>7s} {:>16s} {:>16s} {:>9s} {:>8s} {:>7s} {:>8s}'.format(
        'dataset', 'task', ref, var, 'delta', 't', 'p', 'n_pos'))
    all_d = []
    for data, task in cases:
        a = [lm(data, task, ref, s) for s in range(nseed)]
        b = [lm(data, task, var, s) for s in range(nseed)]
        pair = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pair) < 3:
            print('{:10s} {:>7s}   <insufficient data ({} pairs)>'.format(
                data, task, len(pair)))
            continue
        xa = np.array([p[0] for p in pair])
        xb = np.array([p[1] for p in pair])
        d = xb - xa
        n = len(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std(ddof=1) > 1e-12 else np.nan
        p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
        print('{:10s} {:>7s} {:8.2f}+-{:<6.2f} {:8.2f}+-{:<6.2f} '
              '{:>+9.3f} {:>8.2f} {:>7.4f} {:>5d}/{}'.format(
                  data, task, xa.mean(), xa.std(), xb.mean(), xb.std(),
                  d.mean(), t, p, int((d > 0).sum()), n))
        all_d.extend(d.tolist())
        print('{:10s} {:>7s}   per-seed delta: {}'.format(
            '', '', [round(v, 2) for v in d]))

    if all_d:
        d = np.array(all_d)
        n = len(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        p = 2 * stats.t.sf(abs(t), n - 1)
        print('-' * 92)
        print('合并: Δ = {:+.3f}  t = {:.2f}  p = {:.4f}  正向 {}/{}  '
              '(方差比 {:.2f})'.format(d.mean(), t, p, int((d > 0).sum()), n,
                                     np.nan))
    print('=' * 92)
    print('判读:')
    print('  p < 0.05 且 Δ > 0        -> 真实改进 ✅')
    print('  p >= 0.05 但多处同向     -> 趋势存在，需更多种子 🟡')
    print('  方向不一致               -> 无效 ❌')


if __name__ == '__main__':
    main()
