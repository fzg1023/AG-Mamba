# -*- coding: utf-8 -*-
"""生成 HUST 上本方法（交付配置）的逐任务 best / last10 mean±std，供报告与 §6.1 使用。"""
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
TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]
PAT = {
    'AG-delivered': 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log',
    'WIDAN_long': 'HUST_{t}_WIDAN_long_seed{s}.log',
    'WIDAN-std': 'HUST_{t}_WIDAN_seed{s}.log',
    'DANN': 'HUST_{t}_DANN_seed{s}.log',
    'UDTL': 'HUST_{t}_UDTL_seed{s}.log',
}
RULES = {'best': lambda v: max(v), 'last10': lambda v: float(np.mean(v[-10:]))}


def series(p):
    if not os.path.exists(p):
        return []
    with open(p, errors='ignore') as f:
        return [float(x) * 100 for x in re.findall(r'tgt_acc:\s*([0-9.]+)',
                                                   f.read())]


print('| 方法/口径 | ' + ' | '.join(TASKS) + ' | 平均 |')
print('|' + '---|' * 12)
for name, pat in PAT.items():
    for rule, fn in RULES.items():
        cells, allv = [], []
        for t in TASKS:
            vals = [fn(series(_findlog(pat.format(t=t, s=s))))
                    for s in SEEDS]
            vals = [v for v in vals if v == v and v > 0]
            if not vals:
                cells.append('–')
                continue
            cells.append('{:.2f} ± {:.2f}'.format(np.mean(vals), np.std(vals)))
            allv.extend(vals)
        pool = np.array(allv)
        print('| {} ({}) | {} | **{:.2f} ± {:.2f}** |'.format(
            name, rule, ' | '.join(cells), pool.mean(), pool.std()))
