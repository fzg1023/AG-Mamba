# -*- coding: utf-8 -*-
"""v4_noL vs WIDAN(及 DANN/UDTL) 在 HUST 上的实时对比。

配对方式: 同 (task, seed) 日志配对; 缺日志的任务自动跳过。
"""
import glob
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
    from scipy import stats
except ImportError:
    stats = None

OUT = 'results'
TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]
SPEED = {'T0': '20', 'T1': '25', 'T2': '30', 'T3': '35', 'T4': '40',
         'T5': '60', 'T6': '65', 'T7': '70', 'T8': '75', 'T9': '80'}


def acc_of(path):
    if not os.path.exists(path):
        return None
    vals = [float(x) for x in re.findall(r'tgt_acc: ([\d.]+)',
                                         open(path, errors='ignore').read())]
    return max(vals) * 100 if vals else None


def collect(pattern):
    """pattern: 含 {task} {seed} 的模板（先格式化再用 _findlog 定位）"""
    res = {}
    for t in TASKS:
        vals = []
        for s in SEEDS:
            p = _findlog(pattern.format(task=t, seed=s))
            v = acc_of(p)
            if v is not None:
                vals.append(v)
        res[t] = vals
    return res


methods = {
    'v4_noL': collect('HUST_{task}_AGMamba_fc_v4_noL_seed{seed}.log'),
    'WIDAN': collect('HUST_{task}_WIDAN_seed{seed}.log'),
    'DANN': collect('HUST_{task}_DANN_seed{seed}.log'),
    'UDTL': collect('HUST_{task}_UDTL_seed{seed}.log'),
}

print('=' * 100)
print('HUST LODO 对比 (每任务 5 种子; 显示 mean +- std; n<5 为进行中)')
print('=' * 100)
hdr = '{:4s} {:>5s} |'.format('task', 'Hz')
for m in methods:
    hdr += ' {:>16s} |'.format(m)
print(hdr)
print('-' * 100)

paired_v4, paired_wid = [], []   # 配对样本 (同 task+seed)
for t in TASKS:
    row = '{:4s} {:>5s} |'.format(t, SPEED[t])
    for m in methods:
        v = methods[m][t]
        if v:
            cell = '{:.2f}+-{:.2f}({})'.format(np.mean(v), np.std(v), len(v))
        else:
            cell = '--'
        row += ' {:>16s} |'.format(cell)
    print(row)
    # 配对
    a, b = methods['v4_noL'][t], methods['WIDAN'][t]
    for i in range(min(len(a), len(b))):
        paired_v4.append(a[i])
        paired_wid.append(b[i])

print('-' * 100)
# 总均值（已完成任务的均值）
for m in methods:
    allv = [x for t in TASKS for x in methods[m][t]]
    done = sum(1 for t in TASKS if methods[m][t])
    print('{:>8s}: overall(已完成 {:2d} 任务) {:.2f}   (n={})'
      .format(m, done, np.mean(allv) if allv else float('nan'), len(allv)))

print()
if paired_v4 and len(paired_v4) == len(paired_wid):
    a = np.array(paired_v4); b = np.array(paired_wid)
    d = a - b
    n = len(d)
    print('配对检验 v4_noL - WIDAN (n={}):'.format(n))
    print('  Δ = {:+.3f}   我方均值 {:.2f} / WIDAN {:.2f}'
          .format(d.mean(), a.mean(), b.mean()))
    if stats is not None and n > 1:
        t, p = stats.ttest_rel(a, b)
        print('  paired t = {:.2f}, p = {:.4f}'.format(t, p))
    print('  正向 {}/{}  负向 {}/{}'
          .format((d > 0).sum(), n, (d < 0).sum(), n))
    for i, t_ in enumerate(TASKS):
        pass
else:
    print('配对样本不完整: v4={} widan={}'.format(len(paired_v4), len(paired_wid)))
