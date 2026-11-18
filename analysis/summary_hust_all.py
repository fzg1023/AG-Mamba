# -*- coding: utf-8 -*-
"""HUST 全方法汇总（best-epoch 与 last-10 两种口径），并输出 markdown。

扫描所有 results_* 目录中的 HUST_T{k}_{方法}_seed{s}.log；
AGMamba 变体保留完整名（AGMamba_fc_xxx）。
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]
PATS = [re.compile(r'tgt_acc:?\s*([0-9.]+)'), re.compile(r'tgt_train_acc:?\s*([0-9.]+)'),
        re.compile(r'tar-Acc:\s*([0-9.]+)')]

files = defaultdict(dict)          # method -> (task, seed) -> path
import sys as _sys
_sys.path.insert(0, ROOT)
from agmamba.paths import iter_logs as _iter_logs  # noqa: E402
for f in _iter_logs(dataset='HUST'):
    if True:
        m = re.match(r'HUST_(T\d+)_(.+)_seed(\d+)\.log$', os.path.basename(f))
        if not m:
            continue
        task, method, seed = m.group(1), m.group(2), int(m.group(3))
        if seed not in SEEDS:
            continue
        files[method][(task, seed)] = f


def series(p):
    with open(p, errors='ignore') as fh:
        txt = fh.read()
    for pat in PATS:
        v = [float(x) * 100 for x in pat.findall(txt)]
        if v:
            return v
    return []


print('=== 方法清单 ===')
for m in sorted(files):
    print('  {:34s} {:3d} files'.format(m, len(files[m])))

print('\n=== 汇总（≥40 个运行的完整方法）===')
rows = []
for m in sorted(files):
    d = files[m]
    if len(d) < 40:
        continue
    best, last = [], []
    for k, p in d.items():
        v = series(p)
        if len(v) >= 10:
            best.append(max(v))
            last.append(float(np.mean(v[-10:])))
    if len(best) < 40:
        continue
    rows.append((m, float(np.mean(best)), float(np.std(best)),
                 float(np.mean(last)), float(np.std(last)), len(best)))

print('| 方法 | best-epoch | last-10 | runs |')
print('|---|---|---|---|')
for m, b, bs, l, ls, n in sorted(rows, key=lambda x: -x[1]):
    print('| {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {} |'.format(
        m, b, bs, l, ls, n))
