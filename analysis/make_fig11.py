# -*- coding: utf-8 -*-
"""Fig. 11：α 扫描精度曲线（T0/T9，5 种子，dc_frac=0.3 统一）。
数据源：results_dgfdb 逐种子日志 max(tgt_acc)；缺失的 α 点自动跳过。
输出: docs/manuscript/figures/fig11.pdf / fig11.png
"""
import os
import re
import glob
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
RD = os.path.join(ROOT, 'results')
os.makedirs(FIGD, exist_ok=True)

# (alpha, variant, matched_dc)
POINTS = [
    (0.0, 'gw_sw_a00', True),
    (0.0, 'gw_wide_gabor', False),      # 备选锚点（dc_frac=0.5，Fig.3 用）
    (0.5, 'gw_sw_a05', True),
    (1.0, 'gw_alias1', True),
    (1.5, 'gw_sw_a15', True),
    (2.0, 'gw_alias2', True),
    (3.0, 'gw_sw_a30', True),
]


def parse_max(f):
    s = open(f, errors='ignore').read()
    v = [float(x) for x in re.findall(r'tgt_acc:\s*([0-9.]+)', s)]
    return max(v) * 100 if v else None


def acc_of(variant, task):
    vals = []
    for f in sorted(glob.glob(_findlog('HUST_%s_AGMamba_fc_%s_seed*.log' % (task, variant)))):
        a = parse_max(f)
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


rows = []
for al, var, matched in POINTS:
    r = {t: acc_of(var, t) for t in ['T0', 'T9']}
    if r['T0'] and r['T9']:
        rows.append((al, var, matched, r))
        print('alpha=%.1f %-14s dc=%s  T0=%.2f±%.2f  T9=%.2f±%.2f (n=%d)'
              % (al, var, '0.3' if matched else '0.5',
                 r['T0'][0], r['T0'][1], r['T9'][0], r['T9'][1], r['T0'][2]))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.8, 'legend.fontsize': 6.5,
                     'legend.frameon': False})

fig, ax = plt.subplots(figsize=(3.9, 3.0))
xs = [r[0] for r in rows]
t0 = [r[3]['T0'][0] for r in rows]
t0e = [r[3]['T0'][1] for r in rows]
t9 = [r[3]['T9'][0] for r in rows]
t9e = [r[3]['T9'][1] for r in rows]
m = [r[2] for r in rows]

ax.errorbar([x for x, mm in zip(xs, m) if mm],
            [v for v, mm in zip(t0, m) if mm],
            yerr=[v for v, mm in zip(t0e, m) if mm],
            fmt='o-', color='#1f77b4', ms=3.6, lw=0.9, capsize=2,
            label='T0 (20 Hz)')
ax.errorbar([x for x, mm in zip(xs, m) if mm],
            [v for v, mm in zip(t9, m) if mm],
            yerr=[v for v, mm in zip(t9e, m) if mm],
            fmt='s--', color='#d62728', ms=3.4, lw=0.9, capsize=2,
            label='T9 (80 Hz)')
if m and not all(m):
    hm = [not mm for mm in m]
    ax.errorbar([x for x, mm in zip(xs, hm) if mm],
                [v for v, mm in zip(t0, hm) if mm],
                fmt='o', mfc='none', mec='#1f77b4', ms=3.6,
                label='T0 (dc$_f$=0.5)')
    ax.errorbar([x for x, mm in zip(xs, hm) if mm],
                [v for v, mm in zip(t9, hm) if mm],
                fmt='s', mfc='none', mec='#d62728', ms=3.4,
                label='T9 (dc$_f$=0.5)')

ax.axvline(2.0, color='0.5', lw=0.7, ls=':')
ax.text(2.06, 57.5, 'operating point', fontsize=6.0, color='0.35')
ax.set_xlim(-0.15, 3.3)
ax.set_ylim(55, 101)
ax.set_xlabel(r'aliasing strength $\alpha$', labelpad=1.5)
ax.set_ylabel('target accuracy (%)', labelpad=1.5)
ax.legend(loc='lower right', fontsize=6.0, ncol=2, columnspacing=0.8)

fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(FIGD, 'fig11.pdf'))
fig.savefig(os.path.join(FIGD, 'fig11.png'), dpi=300)
print('saved: fig11  (%d points)' % len(rows))
