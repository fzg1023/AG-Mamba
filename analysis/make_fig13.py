#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 13：E3 训练前后核审计（数据来自 exp_e3_recalc.py 的 figures/e3_stats.npz）。

(a) 过零数 ζ（每 64 点支撑）；(b) 谱集中度 χ（2048 点补零 FFT，§3.3 口径）；
(c) 膨胀闭包 η(1.067)（表 15 同一探测集）。
虚线为该指标在 3.3 节参照核组上的水平（窄带解析 / 宽核 Laplace / 随机）。
输出 figures/fig13.pdf / fig13.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIG = os.path.join(ROOT, 'docs', 'manuscript', 'figures')

d = np.load(os.path.join(FIG, 'e3_stats.npz'), allow_pickle=True)
T9 = ['alpha=0|T9|0', 'alpha=2|T9|0']
LAB = ['analytic Gabor\n(α = 0)', 'aliased Gabor\n(α = 2, delivered)']
COL = ['#1f77b4', '#d62728']

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))
x = np.arange(2)
w = 0.34

# ------------------------------------------------------------------ (a) zeta
ax = axes[0]
v0 = [float(d[k + '|zeta0']) for k in T9]
v1 = [float(d[k + '|zeta1']) for k in T9]
ax.bar(x - w / 2, v0, w, color=COL, alpha=0.38, edgecolor='k', lw=0.5,
       label='frozen (initialised)')
ax.bar(x + w / 2, v1, w, color=COL, edgecolor='k', lw=0.5, label='after 480 steps')
for xi, (a, b) in enumerate(zip(v0, v1)):
    ax.annotate('{:.1f}'.format(a), (xi - w / 2, a), ha='center', va='bottom', fontsize=7.5)
    ax.annotate('{:.1f}'.format(b), (xi + w / 2, b), ha='center', va='bottom', fontsize=7.5)
for y, lab, ls in [(float(d['ref:laplace|zeta']), 'wide-kernel ref. [10]', ':'),
                   (float(d['ref:random|zeta']), 'random init', '--'),
                   (float(d['ref:nb_analytic|zeta']), 'analytic narrow band', '-.')]:
    ax.axhline(y, color='#444444', ls=ls, lw=1.0)
    ax.annotate(lab, (1.48, y), fontsize=6.5, va='bottom', ha='right', color='#333333')
ax.set_xticks(x)
ax.set_xticklabels(LAB, fontsize=8)
ax.set_ylabel(r'zero crossings $\zeta$ (per 64 samples)')
ax.set_ylim(0, 44)
ax.set_xlim(-0.6, 1.6)
ax.set_title('(a) dispersion: training disperses, and stops short', fontsize=9)
ax.legend(fontsize=7, loc='upper left')
ax.grid(axis='y', alpha=0.25, lw=0.4)

# ------------------------------------------------------------------- (b) chi
ax = axes[1]
v0 = [1e3 * float(d[k + '|chi0']) for k in T9]
v1 = [1e3 * float(d[k + '|chi1']) for k in T9]
ax.bar(x - w / 2, v0, w, color=COL, alpha=0.38, edgecolor='k', lw=0.5)
ax.bar(x + w / 2, v1, w, color=COL, edgecolor='k', lw=0.5)
for xi, (a, b) in enumerate(zip(v0, v1)):
    ax.annotate('{:.1f}'.format(a), (xi - w / 2, a), ha='center', va='bottom', fontsize=7.5)
    ax.annotate('{:.1f}'.format(b), (xi + w / 2, b), ha='center', va='bottom', fontsize=7.5)
for y, lab, ls in [(1e3 * float(d['ref:laplace|chi']), 'wide-kernel ref. [10]', ':'),
                   (1e3 * float(d['ref:nb_analytic|chi']), 'analytic narrow band', '-.'),
                   (1e3 * float(d['ref:random|chi']), 'random init', '--')]:
    ax.axhline(y, color='#444444', ls=ls, lw=1.0)
    ax.annotate(lab, (1.48, y), fontsize=6.5, va='bottom', ha='right', color='#333333')
ax.set_xticks(x)
ax.set_xticklabels(LAB, fontsize=8)
ax.set_ylabel(r'spectral concentration $\chi$ ($\times 10^{-3}$)')
ax.set_ylim(0, 31)
ax.set_xlim(-0.6, 1.6)
ax.set_title('(b) concentration: essentially unchanged', fontsize=9)
ax.grid(axis='y', alpha=0.25, lw=0.4)

# ------------------------------------------------------------------- (c) eta
ax = axes[2]
v0 = [float(d[k + '|eta0']) for k in T9]
v1 = [float(d[k + '|eta1']) for k in T9]
for xi in x:
    ax.annotate('', xy=(xi, v1[xi]), xytext=(xi, v0[xi]),
                arrowprops=dict(arrowstyle='-|>', color=COL[xi], lw=1.6))
ax.plot(x, v0, 'o', color='none', mec='k', mew=0.8, ms=7, label='initialised')
ax.plot(x, v1, 'o', color='none', mec='k', mew=0.8, ms=7, mfc='k', label='after 480 steps')
for xi in x:
    ax.annotate('{:.3f}'.format(v0[xi]), (xi - 0.06, v0[xi]), ha='right', fontsize=7.5)
    ax.annotate('{:.3f}'.format(v1[xi]), (xi + 0.06, v1[xi]), ha='left', fontsize=7.5)
for y, lab, ls in [(float(d['ref:random|eta']), 'random init', '--'),
                   (float(d['ref:laplace|eta']), 'wide-kernel ref. [10]', ':')]:
    ax.axhline(y, color='#444444', ls=ls, lw=1.0)
    ax.annotate(lab, (1.48, y), fontsize=6.5, va='bottom', ha='right', color='#333333')
ax.set_xticks(x)
ax.set_xticklabels(LAB, fontsize=8)
ax.set_ylabel(r'closure $\eta(1.067)$')
ax.set_ylim(0.2, 1.05)
ax.set_xlim(-0.6, 1.6)
ax.set_title('(c) closure: the winner loses it, the loser keeps it', fontsize=9)
ax.grid(axis='y', alpha=0.25, lw=0.4)

for ax, k in zip(axes, T9):
    ax.annotate('T9 acc. {:.1f}'.format(float(d[k + '|acc'])), (0.5, 0.02),
                xycoords='axes fraction', ha='center', fontsize=7, color='#333333')

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig13.pdf'))
fig.savefig(os.path.join(FIG, 'fig13.png'), dpi=220)
print('wrote figures/fig13.pdf and fig13.png')
