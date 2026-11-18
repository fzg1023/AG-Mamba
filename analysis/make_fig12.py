#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 12：膨胀鲁棒性（E1 静态闭包 + E2 非膨胀成分）。

(a) eta(s) 曲线（各核组），竖线为 T9 所需的 s = 1.067；
(b) eta(1.067) 对 T9 best-epoch 精度（α 扫描 6 点）——说明二者不相关；
(c) E2：物理非膨胀信号下“通道平移模型”能解释的程度 max_d cos，三种转速步长。
输出 figures/fig12.pdf / fig12.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIG = os.path.join(ROOT, 'docs', 'manuscript', 'figures')

d1 = np.load(os.path.join(FIG, 'e1e2.npz'), allow_pickle=True)
d2 = np.load(os.path.join(FIG, 'e2.npz'), allow_pickle=True)
ETA_KEY = d1['eta_key'].item()
T9M = d1['T9'].item()

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

# ---------------------------------------------------------------- (a) eta(s)
ax = axes[0]
style = [('alpha=0.0', 'analytic Gabor (α = 0)', '#1f77b4', '-'),
         ('alpha=1.0', 'aliased (α = 1)', '#ff7f0e', '-'),
         ('alpha=2.0', 'aliased (α = 2, delivered)', '#d62728', '-'),
         ('alpha=3.0', 'aliased (α = 3)', '#9467bd', '-'),
         ('random', 'random init', '#7f7f7f', '--'),
         ('ref_Laplace_64k', 'wide-kernel reference [10]', '#2ca02c', ':')]
s = d1['s']
for key, lab, c, ls in style:
    ax.plot(s, d1['curve_' + key], ls, color=c, lw=1.6, label=lab)
ax.axvline(80 / 75, color='k', lw=0.8, alpha=0.6)
ax.annotate('T9 step\n$s=1.067$', xy=(80 / 75, 0.45), xytext=(1.10, 0.45),
            fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7))
ax.set_xlabel('dilation factor $s$')
ax.set_ylabel(r'closure $\eta(s)$')
ax.set_ylim(0.2, 1.02)
ax.set_title('(a) static dilation closure of the frozen bank', fontsize=9)
ax.legend(fontsize=6.5, loc='lower left')
ax.grid(alpha=0.25, lw=0.4)

# ------------------------------------------------- (b) eta(1.067) vs T9 acc
ax = axes[1]
alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
eta = [float(ETA_KEY[item]) for item in
       ['alpha=0.0', 'alpha=0.5', 'alpha=1.0', 'alpha=1.5', 'alpha=2.0', 'alpha=3.0']]
t9 = [float(T9M['alpha={}'.format(a)]) for a in alphas]
ax.plot(eta, t9, 'o-', color='#d62728', lw=1.4, ms=5)
for e_, t_, a in zip(eta, t9, alphas):
    ax.annotate('α={}'.format(a), (e_, t_), textcoords='offset points',
                xytext=(4, -3), fontsize=7)
ax.plot([float(ETA_KEY['random'])], [78.85], 's', color='#7f7f7f', ms=6)
ax.annotate('random init', (float(ETA_KEY['random']), 78.85),
            textcoords='offset points', xytext=(4, 2), fontsize=7)
ax.set_xlabel(r'$\eta(1.067)$ of the frozen bank')
ax.set_ylabel('T9 best-epoch accuracy (%)')
ax.set_title('(b) closure does not predict extrapolation', fontsize=9)
ax.grid(alpha=0.25, lw=0.4)

# --------------------------------------------- (c) E2 shift explainability
ax = axes[2]
rows = d2['rows']
rows = list(rows.item()) if getattr(rows, 'ndim', 1) == 0 else list(rows)
banks = ['alpha=0.0', 'alpha=1.0', 'alpha=2.0', 'alpha=3.0', 'random', 'ref_Laplace']
steps = [(25, 30), (40, 60), (75, 80)]
labels = ['25→30 Hz\n$s=1.20$', '40→60 Hz\n$s=1.50$', '75→80 Hz\n$s=1.067$']
width = 0.13
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#7f7f7f', '#2ca02c']
for j, b in enumerate(banks):
    vals = []
    for (a_, b_) in steps:
        for r in rows:
            if r[0] == b and int(r[1]) == a_ and int(r[2]) == b_:
                vals.append(float(r[5]))
    xpos = np.arange(len(steps)) + (j - (len(banks) - 1) / 2) * width
    ax.bar(xpos, vals, width=width, color=colors[j], label=b)
ax.set_xticks(np.arange(len(steps)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('best shift-model cosine')
ax.set_ylim(0.3, 1.02)
ax.set_title('(c) what a pure shift explains (physical signal)', fontsize=9)
ax.legend(fontsize=6, ncol=2, loc='lower right')
ax.grid(alpha=0.25, lw=0.4, axis='y')

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(FIG, 'fig12.' + ext), dpi=300)
print('[OK] fig12.pdf / fig12.png')
