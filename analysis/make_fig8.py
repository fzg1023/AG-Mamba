# -*- coding: utf-8 -*-
"""Fig. 8：原子三对照（解析 Gabor α=0 / 混叠 Gabor α=2 / 宽核参考）。
上行核波形，下行归一化幅度谱；谱内标注相应的整组统计 (ζ̄, χ̄)。
输出: docs/manuscript/figures/fig8.pdf / fig8.png
"""
import os
import sys
import math

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
sys.path.insert(0, ROOT)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

from models.agmamba_backbone import gabor_bank  # noqa: E402
from models.widan_net import Laplace_fast         # noqa: E402

K, KS, Q, DC = 64, 64, 0.6, 0.3
FK = 32   # f = 0.049

bank_a0 = gabor_bank(K, KS, 0.005, 0.45, Q, DC, alias=0.0).numpy()[:, 0, :]
bank_a2 = gabor_bank(K, KS, 0.005, 0.45, Q, DC, alias=2.0).numpy()[:, 0, :]
lap = Laplace_fast(out_channels=64, kernel_size=64, eps=0.1,
                   frequency=64000, mode='sigmoid').forward().numpy()[:, 0, :]


def zc(kern):
    return np.mean(np.sum(np.diff(np.sign(kern), axis=-1) != 0, axis=-1))


def chi(kern):
    A = np.abs(np.fft.rfft(kern, n=1024, axis=-1))
    A = A / A.sum(-1, keepdims=True)
    return A.max(-1).mean()


NFFT = 4096
U = np.fft.rfftfreq(NFFT)

# 宽核参考：选峰频最接近 0.049 的核
pk = [U[np.argmax(np.abs(np.fft.rfft(l, n=NFFT)))] for l in lap]
JL = int(np.argmin(np.abs(np.array(pk) - 0.049)))

cols = [('analytic Gabor ($\\alpha$=0)', bank_a0[FK], bank_a0),
        ('aliased Gabor ($\\alpha$=2)', bank_a2[FK], bank_a2),
        ('wide-kernel reference', lap[JL], lap)]
colors = ['#1f77b4', '#d62728', '#555555']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.8})

fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.3))
fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.13,
                    hspace=0.62, wspace=0.24)
tt = np.arange(KS)
for c, (title, sig, bank) in enumerate(cols):
    ax = axes[0, c]
    ax.plot(tt, sig / np.abs(sig).max(), color=colors[c], lw=0.75)
    if c < 2:
        env = np.exp(-(tt - (KS - 1) / 2.0) ** 2 / (2.0 * (Q / 0.0492) ** 2))
        env = env / env.max() * 0.42
        ax.plot(tt, env, color=colors[c], lw=0.5, ls='--', alpha=0.6)
        ax.plot(tt, -env, color=colors[c], lw=0.5, ls='--', alpha=0.6)
    ax.set_xlim(0, KS - 1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_yticks([])
    if c == 0:
        ax.set_ylabel('kernel', labelpad=1.5)

    ax = axes[1, c]
    A = np.abs(np.fft.rfft(sig, n=NFFT))
    ax.semilogx(U[1:], A[1:] / A.max(), color=colors[c], lw=0.8)
    ax.set_xlim(0.004, 0.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(0.0045, 0.98, 'bank  $\\bar{\\zeta}$=%.1f,  $\\bar{\\chi}$=%.3f'
            % (zc(bank), chi(bank)), fontsize=6.0, va='top', color='0.25')
    if c == 0:
        ax.set_ylabel('norm. amplitude', labelpad=1.5)
    ax.set_xlabel('freq. (cycles/sample)', labelpad=1.2, fontsize=6.8)

fig.savefig(os.path.join(FIGD, 'fig8.pdf'))
fig.savefig(os.path.join(FIGD, 'fig8.png'), dpi=300)
print('saved: fig8 (Laplace kernel idx %d, peak %.3f)' % (JL, pk[JL]))
