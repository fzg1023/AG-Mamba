#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为表 13（容量对齐控制）补齐所需均值：PU 与 HUST 两档宽度、两档预算。"""
import os
import re

import numpy as np

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
OUT = os.path.join(ROOT, 'results')
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEEDS = list(range(5))


def best(path):
    if not os.path.exists(path) or os.path.getsize(path) < 500:
        return None
    with open(path, errors='ignore') as f:
        v = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return max(v) if v else None


def lm(data, task, name, seed):
    return _findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(
        data, task, name, seed))


def mean(data, tasks, name):
    v = [best(lm(data, t, name, s)) for t in tasks for s in SEEDS]
    v = [x for x in v if x is not None]
    return np.mean(v) if v else float('nan'), len(v)


print('## PU（4 任务 x 5 种子）')
for nm, label in [('puac_a2_c128', '128ch (α=2)'), ('puac_a2_c512', '512ch (α=2)'),
                  ('cap960', '960ch (α=2)'), ('base', 'base (128ch 窄带)'),
                  ('xg_std', 'xg_std (交付)')]:
    m, n = mean('PU', ['T0', 'T1', 'T2', 'T3'], nm)
    print('  {:<18s} {:.2f}  (n={})'.format(label, m, n))

print('\n## HUST T0/T4/T9（3 任务 x 5 种子）')
for nm, label in [('sc_steps6x', '交付配置 512ch / 2,880 步'),
                  ('cap960x', 'cap960 3.85M / 2,880 步'),
                  ('sa_lowf512', '512ch / 480 步（探针）'),
                  ('sa_cap768_lowf', '768ch / 480 步'),
                  ('cap960p', '960ch / 480 步'),
                  ('sa_cap1024', '1024ch / 480 步')]:
    m, n = mean('HUST', ['T0', 'T4', 'T9'], nm)
    print('  {:<26s} {:.2f}  (n={})'.format(label, m, n))
