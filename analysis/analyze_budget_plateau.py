# -*- coding: utf-8 -*-
"""旁证：不同训练预算下的 last10 平台高度（T0/T4/T9 三任务，5 种子）。

用已有日志比较 480 / 1440 / 2880 / 5760 步（同前端、同调度），
判断"末段平台低"是否为步数过长/过拟合所致。
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
RD = os.path.join(ROOT, 'results')
TASKS = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2, 3, 4]

VARIANTS = [
    ('v4_noL (480)', 'HUST_{t}_AGMamba_fc_v4_noL_seed{s}.log', 480),
    ('sb_long3x (1440)', 'HUST_{t}_AGMamba_fc_sb_long3x_seed{s}.log', 1440),
    ('sc_steps6x (2880)', 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log', 2880),
    ('sd_12x (5760)', 'HUST_{t}_AGMamba_fc_sd_12x_seed{s}.log', 5760),
    ('em999 (2880+EMA)', 'HUST_{t}_AGMamba_fc_em999_seed{s}.log', 2880),
]


def series(p):
    if not os.path.exists(p):
        return []
    with open(p, errors='ignore') as f:
        return [float(x) * 100 for x in re.findall(r'tgt_acc:\s*([0-9.]+)',
                                                   f.read())]


print('{:20s} {:>5s} {:>7s} {:>7s} {:>7s} {:>7s}'.format(
    'variant', 'n', 'best', 'last10', 'last20', 'final'))
for name, pat, steps in VARIANTS:
    recs = {}
    for t in TASKS:
        for s in SEEDS:
            v = series(_findlog(pat.format(t=t, s=s)))
            if len(v) >= 10:
                recs[(t, s)] = v
    if not recs:
        print('{:20s}  -- 无日志 --'.format(name))
        continue
    best = np.mean([max(v) for v in recs.values()])
    l10 = np.mean([np.mean(v[-10:]) for v in recs.values()])
    l20 = np.mean([np.mean(v[-20:]) for v in recs.values()])
    fin = np.mean([v[-1] for v in recs.values()])
    print('{:20s} {:5d} {:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(
        name, len(recs), best, l10, l20, fin))
    # 逐任务 last10
    per = []
    for t in TASKS:
        vs = [v for (tt, _), v in recs.items() if tt == t]
        per.append('{}={:.2f}'.format(t, np.mean([np.mean(v[-10:]) for v in vs])))
    print('{:20s} last10 逐任务: {}'.format('', '  '.join(per)))

# 对照：同任务上的两个 WIDAN 档
for name, pat in (('WIDAN (480)', 'HUST_{t}_WIDAN_seed{s}.log'),
                  ('WIDAN_long (2880)', 'HUST_{t}_WIDAN_long_seed{s}.log')):
    recs = {}
    for t in TASKS:
        for s in SEEDS:
            v = series(_findlog(pat.format(t=t, s=s)))
            if len(v) >= 10:
                recs[(t, s)] = v
    per = []
    for t in TASKS:
        vs = [v for (tt, _), v in recs.items() if tt == t]
        per.append('{}={:.2f}'.format(t, np.mean([np.mean(v[-10:]) for v in vs])))
    print('{:20s} {:5d} {:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(
        name, len(recs), np.mean([max(v) for v in recs.values()]),
        np.mean([np.mean(v[-10:]) for v in recs.values()]),
        np.mean([np.mean(v[-20:]) for v in recs.values()]),
        np.mean([v[-1] for v in recs.values()])))
    print('{:20s} last10 逐任务: {}'.format('', '  '.join(per)))
