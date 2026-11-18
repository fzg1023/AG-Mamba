# -*- coding: utf-8 -*-
"""Fig. 5/6/7 单独出图（§4.2.1 配图拆分）+ Fig. 5–7 合成总览。
Fig 5：常 Q 原子波形与包络；Fig 6：原子幅度谱（log-f）；Fig 7：覆盖 C(u) vs C~(u)。
输出: docs/manuscript/figures/fig5.pdf|png, fig6.pdf|png, fig7.pdf|png, fig5-7_composite.pdf|png
"""
import os
import math
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
sys.path.insert(0, ROOT)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

from models.agmamba_backbone import gabor_bank  # noqa: E402

K, KS = 64, 64
F_LO, F_HI, Q, DC = 0.005, 0.45, 0.6, 0.3
bank = gabor_bank(K, KS, F_LO, F_HI, Q, DC).numpy()[:, 0, :]
fk = np.exp(np.linspace(math.log(F_LO), math.log(F_HI), K))

NFFT = 16384
U = np.fft.rfftfreq(NFFT)
SP = np.abs(np.fft.rfft(bank, n=NFFT, axis=-1)) ** 2
AMP = np.sqrt(SP)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.8, 'legend.fontsize': 6.5,
                     'legend.frameon': False})

ks = [60, 44, 24, 8]
cols = {60: '#d62728', 44: '#e07b00', 24: '#1f77b4', 8: '#2ca02c'}
offs = {60: 3.3, 44: 2.2, 24: 1.1, 8: 0.0}
FLAB = {k: '%.4f' % fk[k] for k in ks}


def draw_waves(ax):
    tt = np.arange(KS)
    for k in ks:
        sig = bank[k] / np.abs(bank[k]).max()
        env = np.exp(-(tt - (KS - 1) / 2.0) ** 2
                     / (2.0 * (Q / fk[k]) ** 2))
        env = env / max(env.max(), 1e-9) * 0.42
        ax.plot(tt, sig + offs[k], color=cols[k], lw=0.75)
        ax.plot(tt, offs[k] + env, color=cols[k], lw=0.5, ls='--',
                alpha=0.6)
        ax.plot(tt, offs[k] - env, color=cols[k], lw=0.5, ls='--',
                alpha=0.6)
        ax.text(63, offs[k] + 0.66, '$f$=' + FLAB[k], fontsize=5.8,
                color=cols[k], ha='right')
    ax.set_xlim(0, KS - 1)
    ax.set_ylim(-0.78, 4.45)
    ax.set_xlabel('sample index', labelpad=1.5)
    ax.set_ylabel('atoms (offset)', labelpad=1.5)
    ax.set_yticks([])


def draw_spectra(ax):
    for k in ks:
        ax.semilogx(U[1:], AMP[k][1:] / AMP[k].max(), color=cols[k], lw=0.8,
                    label='$f$=' + FLAB[k])
    ax.set_xlim(0.0035, 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('frequency (cycles/sample, log)', labelpad=1.5)
    ax.set_ylabel('norm. amplitude', labelpad=1.5)
    ax.legend(loc='upper right', fontsize=5.6, handlelength=1.1)
    ax.text(0.0042, 0.92, 'constant-Q: equal width in log-$f$',
            fontsize=5.8, color='0.25')


def draw_coverage(ax):
    C = SP.sum(axis=0)
    m_int = (U >= 0.015) & (U <= 0.375)
    C_n = C / C[m_int].max()
    Ct = U * C
    Ct_n = Ct / Ct[m_int].max()
    ax.loglog(U[1:], C_n[1:], color='0.55', lw=0.9,
              label=r'$C(u)$ (density)')
    ax.loglog(U[1:], Ct_n[1:], color='#1f4ea1', lw=1.6,
              label=r'$\tilde{C}(u)=u\,C(u)$ (energy per octave)')
    ax.axvspan(0.015, 0.375, color='#4c9a4c', alpha=0.10, lw=0)
    ax.axvline(F_LO, color='0.4', lw=0.6, ls=':')
    ax.axvline(F_HI, color='0.4', lw=0.6, ls=':')
    ax.text(0.016, 0.05, 'within $\\times$1.4 over 4.6 octaves',
            fontsize=6.2, color='#2E6B2E')
    ax.set_xlim(0.0035, 0.5)
    ax.set_ylim(0.02, 1.6)
    ax.set_xlabel('frequency (cycles/sample, log)', labelpad=1.5)
    ax.set_ylabel('normalized (log)', labelpad=1.5)
    ax.legend(loc='upper right', fontsize=5.8, handlelength=1.4)


# ---------------- 单独出图 ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.6))
draw_waves(ax)
fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(FIGD, 'fig5.pdf'))
fig.savefig(os.path.join(FIGD, 'fig5.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.7))
draw_spectra(ax)
fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(FIGD, 'fig6.pdf'))
fig.savefig(os.path.join(FIGD, 'fig6.png'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.6, 2.9))
draw_coverage(ax)
fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(FIGD, 'fig7.pdf'))
fig.savefig(os.path.join(FIGD, 'fig7.png'), dpi=300)
plt.close(fig)

# ---------------- 合成总览 ----------------
fig = plt.figure(figsize=(7.4, 4.4))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05],
                       hspace=0.55, wspace=0.28)
draw_waves(fig.add_subplot(gs[0, 0]))
draw_spectra(fig.add_subplot(gs[0, 1]))
draw_coverage(fig.add_subplot(gs[1, :]))
fig.savefig(os.path.join(FIGD, 'fig5-7_composite.pdf'))
fig.savefig(os.path.join(FIGD, 'fig5-7_composite.png'), dpi=300)
plt.close(fig)

print('saved: fig5/fig6/fig7 (+ composite fig5-7_composite) to', FIGD)
