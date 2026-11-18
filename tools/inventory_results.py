# -*- coding: utf-8 -*-
"""盘点 results_* 目录：每个目录里有哪些数据集/任务/方法/种子数。"""
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
DS = ('HUST', 'PU', 'PHM2009', 'UO')
TASK = re.compile(r'^T\d+$')
SEED = re.compile(r'^seed\d+$')

for d in sorted(x for x in os.listdir(ROOT)
                if x.startswith('results_') and
                os.path.isdir(os.path.join(ROOT, x))):
    path = os.path.join(ROOT, d)
    info = defaultdict(lambda: defaultdict(set))      # ds -> method -> seeds
    other = 0
    for f in sorted(os.listdir(path)):
        if not f.endswith('.log'):
            other += 1
            continue
        stem = f[:-4]
        toks = stem.split('_')
        ds = toks[0] if toks[0] in DS else '(none)'
        rest = toks[1:] if ds != '(none)' else toks
        task = next((t for t in rest if TASK.match(t)), '?')
        seed = next((t for t in rest if SEED.match(t)), '?')
        m = [t for t in rest if not TASK.match(t) and not SEED.match(t)]
        method = '_'.join(m) if m else '?'
        info[ds][method].add(seed)
    if not info and other == 0:
        continue
    print('=== {} ==='.format(d))
    for ds in sorted(info):
        print('  [{}]'.format(ds))
        for method in sorted(info[ds]):
            seeds = sorted(info[ds][method])
            print('      {:<22s} seeds={:2d}  ({})'.format(
                method, len(seeds), ','.join(s[4:] for s in seeds)))
    if other:
        print('  (non-log files: {})'.format(other))
    print()
