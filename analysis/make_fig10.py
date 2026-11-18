# -*- coding: utf-8 -*-
"""Fig. 10：α 机制统计。左轴：实测过零率 ζ 与集中度 χ（随 α）；右轴：载波功率占比 J0²(α)（理论）。
输出: docs/manuscript/figures/fig10.pdf / fig10.png
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.special import jv                      # noqa: E402
from models.agmamba_backbone import gabor_bank  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')

K, KS, Q, DC = 64, 64, 0.6, 0.3
alphas = [0.0, 0.5, 1.0, 2.0, 3.0]

zc, ch = [], []
for al in alphas:
    b = gabor_bank(K, KS, 0.005, 0.45, Q, DC, alias=al).numpy()[:, 0, :]
    zc.append(np.mean(np.sum(np.diff(np.sign(b), axis=-1) != 0, axis=-1)))
    A = np.abs(np.fft.rfft(b, n=1024, axis=-1))
    A = A / A.sum(-1, keepdims=True)
    ch.append(A.max(-1).mean())

aa = np.linspace(0, 3, 121)
carrier = jv(0, aa) ** 2

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.8})

fig, ax = plt.subplots(figsize=(3.9, 2.9))
fig.subplots_adjust(left=0.135, right=0.865, top=0.95, bottom=0.145)

ax.plot(alphas, zc, 'o-', color='#1f77b4', ms=3.4, lw=0.9,
        label=r'$\bar{\zeta}$ (zero-crossing rate)')
ax.plot(alphas, np.array(ch) * 100, 's--', color='#d62728', ms=3.2, lw=0.9,
        label=r'$100\,\bar{\chi}$ (concentration)')
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(0, 30)
ax.set_xlabel(r'aliasing strength $\alpha$', labelpad=1.5)
ax.set_ylabel('measured kernel statistics', labelpad=1.5)
ax.legend(loc='upper left', fontsize=6.0, handlelength=1.3)

ax2 = ax.twinx()
ax2.plot(aa, carrier, color='#2e8b57', lw=1.2)
ax2.set_ylim(0, 1.05)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_ylabel(r'carrier power $J_0^2(\alpha)$', labelpad=1.5,
               color='#2e8b57')
ax2.tick_params(axis='y', colors='#2e8b57')

ax.axvline(2.0, color='0.5', lw=0.7, ls=':')
ax.text(2.06, 27.6, 'operating\npoint', fontsize=6.0, color='0.35', va='top')

fig.savefig(os.path.join(FIGD, 'fig10.pdf'))
fig.savefig(os.path.join(FIGD, 'fig10.png'), dpi=300)
print('saved: fig10   zcr=', ['%.1f' % v for v in zc],
      ' chi=', ['%.4f' % v for v in ch])
