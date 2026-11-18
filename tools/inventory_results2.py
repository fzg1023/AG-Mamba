# -*- coding: utf-8 -*-
"""盘点（精简版）：只列非 AGMamba 的方法，用于撰写基线章节。"""
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
    info = defaultdict(lambda: defaultdict(set))
    for f in sorted(os.listdir(path)):
        if not f.endswith('.log'):
            continue
        toks = f[:-4].split('_')
        ds = toks[0] if toks[0] in DS else '(none)'
        rest = toks[1:] if ds != '(none)' else toks
        seed = next((t for t in rest if SEED.match(t)), '?')
        m = [t for t in rest if not TASK.match(t) and not SEED.match(t)]
        method = '_'.join(m) if m else '?'
        if method.startswith('AGMamba') or method.startswith('P2C_LM'):
            continue
        info[ds][method].add(seed)
    if not info:
        continue
    print('=== {} ==='.format(d))
    for ds in sorted(info):
        ms = info[ds]
        print('  [{}] {} methods: {}'.format(
            ds, len(ms),
            ', '.join('{} (s={})'.format(k, len(v)) for k, v in sorted(ms.items()))))
    print()
