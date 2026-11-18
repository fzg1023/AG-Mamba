# -*- coding: utf-8 -*-
"""Fig. 4 方法总览：AG-Mamba 网络架构与数据流图（含证据来源标注）。
输出: docs/manuscript/figures/fig4.pdf / fig4.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                     'mathtext.fontset': 'dejavusans'})

fig = plt.figure(figsize=(7.6, 3.25))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 115.5)
ax.set_ylim(0, 48)
ax.axis('off')

C_BLUE, C_ORANGE, C_GREEN, C_GRAY, C_RED = ('#2E6DA4', '#E07B00',
                                            '#4C9A4C', '0.35', '#B03A2E')


def box(x, y, w, h, fc, ec, ls='-', lw=1.0, rad=1.1, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0,rounding_size=%.1f' % rad,
                 fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))


def arrow(x1, y1, x2, y2, color='0.15', lw=0.9, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=4)


def line(xs, ys, color='0.15', lw=0.9, ls='-'):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3,
            solid_capstyle='round')


# ---------------- 1) 输入 ----------------
box(2, 25, 8, 9, '#ECECEC', C_GRAY)
ax.text(6, 32.3, 'Raw signal', ha='center', fontsize=7, fontweight='bold')
ax.text(6, 30.4, '1 × 1024', ha='center', fontsize=5.6)
ax.text(6, 28.7, '25.6 kHz', ha='center', fontsize=5.6)
t = np.linspace(0, 1, 160)
ax.plot(2.9 + 6.2 * t,
        26.2 + 1.05 * np.sin(2 * np.pi * 4.2 * t) * (1 - 0.55 * t),
        color='#5B7FA6', lw=0.7, zorder=3)

arrow(10, 29.5, 13.5, 29.5)

# ---------------- 2) AG 滤波器组 ----------------
box(13.5, 20, 18, 19, '#DCE9F7', C_BLUE)
ax.text(22.5, 36.6, 'AG filterbank (§4.2)', ha='center', fontsize=6.6,
        fontweight='bold', color='#1F4E79')
ax.text(22.5, 34.1, 'CNN 64→128→256→512', ha='center', fontsize=5.6)
ax.text(22.5, 32.1, 'Gabor init + controlled', ha='center', fontsize=5.6)
ax.text(22.5, 30.1, 'aliasing (α = 2.0)', ha='center', fontsize=5.6)
ax.text(22.5, 28.1, '1024 → 128', ha='center', fontsize=5.6)
ax.text(22.5, 17.4, 'G1 — Figs. 2–3', ha='center', fontsize=5.8,
        style='italic', color='#1F4E79')

arrow(31.5, 29.5, 35, 29.5)
ax.text(33.25, 31.1, '512×128', ha='center', fontsize=5.2, color='0.3')

# ---------------- 3) 投影 ----------------
box(35, 24.5, 11, 10, '#F4F4F4', C_GRAY)
ax.text(40.5, 31.1, 'Linear', ha='center', fontsize=6.2)
ax.text(40.5, 28.8, '512 → d = 128', ha='center', fontsize=5.6)

arrow(46, 29.5, 49.5, 29.5)
ax.plot([49.5], [29.5], marker='o', ms=2.2, color='0.15', zorder=4)

# 上路：前端图直连
line([49.5, 49.5, 69], [29.5, 38, 38])
arrow(69, 38, 76.5, 38)
ax.text(63, 39.6, 'frontend map (direct)', ha='center', fontsize=5.8,
        color='0.3')

# 下路：SSM
line([49.5, 49.5], [29.5, 21])
arrow(49.5, 21, 53.5, 21)
box(53.5, 13, 18.5, 16, '#FDE9D6', C_ORANGE)
ax.text(62.75, 26.5, '3 × Selective SSM', ha='center', fontsize=6.6,
        fontweight='bold', color='#9A5200')
ax.text(62.75, 24.2, 'selective scan', ha='center', fontsize=5.6)
ax.text(62.75, 22.0, 'd = 128 · N = 8 · §4.3', ha='center', fontsize=5.6)
ax.text(62.75, 10.4, '+2.69 pts · p = 0.0003 — §6.4', ha='center',
        fontsize=5.8, style='italic', color='#9A5200')

arrow(72, 21, 76.5, 21)

# ---------------- 4) 聚合 ----------------
box(76.5, 15, 14, 27, '#E3F1E3', C_GREEN)
ax.text(83.5, 37.9, 'per-branch', ha='center', fontsize=5.8)
ax.text(83.5, 35.7, 'mean–max pooling', ha='center', fontsize=5.8)
ax.text(83.5, 33.5, '+ concat', ha='center', fontsize=5.8)
ax.text(83.5, 28.2, 'z (512-d)', ha='center', fontsize=6.8,
        fontweight='bold', color='#2E6B2E')

arrow(90.5, 29.5, 94, 29.5)

# ---------------- 5) 分类头 ----------------
box(94, 24, 13.5, 11, '#EFE7F5', '#7D5BA6')
ax.text(100.75, 32.0, 'Head (§4.4)', ha='center', fontsize=6.5,
        fontweight='bold', color='#5B3E82')
ax.text(100.75, 29.4, 'FC-256 → ReLU →', ha='center', fontsize=5.6)
ax.text(100.75, 27.0, 'Dropout 0.3 → FC-C', ha='center', fontsize=5.6)

arrow(107.5, 29.5, 110.5, 29.5)
ax.text(111.2, 29.5, 'logits', ha='left', va='center', fontsize=6.2)

# ---------------- 6) 训练目标（虚线：仅训练期施加） ----------------
box(45.5, 1.5, 58, 6.5, '#EDF3FA', C_BLUE, ls=(0, (3, 2)))
ax.text(74.5, 6.1, 'Training objective (§4.4)', ha='center',
        fontsize=6.2, fontweight='bold', color='#1F4E79')
ax.text(74.5, 3.5, r'$\mathcal{L}=\mathrm{CE}+\lambda_\Delta\|\Delta\|_2$,  '
                   r'$\lambda_\Delta=0.01$ — no alignment heads (§6.4)',
        ha='center', fontsize=6.0, color='#274A63')
arrow(83.5, 15, 83.5, 8.4, color=C_BLUE, ls=(0, (3, 2)))

fig.savefig(os.path.join(FIGD, 'fig4.pdf'))
fig.savefig(os.path.join(FIGD, 'fig4.png'), dpi=300)
print('saved:', os.path.join(FIGD, 'fig4.pdf'), '/ fig4.png')
