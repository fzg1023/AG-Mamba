# -*- coding: utf-8 -*-
"""Fig. 9：Bessel 折叠梳验证（α=2）。
灰色填充 = 实测原子功率谱；红色竖线 = 式 (12)(13) 预测的折叠梳位置（高度=|J_n| 合并和）。
输出: docs/manuscript/figures/fig9.pdf / fig9.png
"""
import os
import sys
import math

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.special import jv                      # noqa: E402
from models.agmamba_backbone import gabor_bank  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')

K, KS, Q, DC, AL = 64, 64, 0.6, 0.3, 2.0
bank = gabor_bank(K, KS, 0.005, 0.45, Q, DC, alias=AL).numpy()[:, 0, :]
fk = np.exp(np.linspace(math.log(0.005), math.log(0.45), K))
fm = np.exp(np.linspace(math.log(0.20), math.log(0.48), K))

NFFT = 8192
U = np.fft.rfftfreq(NFFT)
P = np.abs(np.fft.rfft(bank, n=NFFT, axis=-1)) ** 2


def fold(u):
    m = u - np.floor(u)
    return np.minimum(m, 1.0 - m)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.8})

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16,
                    wspace=0.20)

for ax, k in zip(axes, [32, 12]):
    m = U <= 0.5
    ax.fill_between(U[m], 0, P[k][m] / P[k][m].max(), color='0.72', lw=0,
                    label='measured')
    ax.plot(U[m], P[k][m] / P[k][m].max(), color='0.35', lw=0.7)
    stems = {}
    for n in range(-14, 15):
        w = float(jv(n, AL))
        if abs(w) < 0.02:
            continue
        pos = fold(fk[k] + n * fm[k])
        stems[pos] = stems.get(pos, 0.0) + abs(w)
    wmax = max(stems.values())
    for pos, w in stems.items():
        ax.plot([pos, pos], [0, w / wmax * 1.05], color='#d62728', lw=0.9)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel('frequency (cycles/sample)', labelpad=1.5)
    if k == 32:
        ax.set_ylabel('norm. power', labelpad=1.5)
    ax.set_title('atom $f$=%.4f,  $f_m$=%.3f' % (fk[k], fm[k]),
                 fontsize=7.5, pad=3)
    ax.text(0.985, 0.97, '$J_0(\\alpha)$=%.2f' % jv(0, AL), fontsize=6.2,
            ha='right', va='top', color='#a51c1c')

axes[0].legend(loc='upper left', fontsize=6.0, handlelength=1.0,
               frameon=False)
fig.savefig(os.path.join(FIGD, 'fig9.pdf'))
fig.savefig(os.path.join(FIGD, 'fig9.png'), dpi=300)
print('saved: fig9')
