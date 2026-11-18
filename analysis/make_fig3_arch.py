# -*- coding: utf-8 -*-
"""Render the delivered v0.3 AG-Mamba architecture (Fig. 3).

The layout follows the supplied reference's framed, multi-panel workflow style.
Outputs: docs/manuscript/figures/fig3_arch.pdf and fig3_arch.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

# A restrained journal palette inspired by the supplied reference figure.
INK = '#252525'
GOLD = '#87621b'
BLUE = '#2e6da4'
BLUE_BG = '#eaf2f9'
BLUE_CELL = '#dce9f5'
GREEN = '#4d8a3c'
GREEN_BG = '#edf5e9'
GREEN_CELL = '#dcebd5'
PURPLE = '#70519a'
PURPLE_BG = '#f2edf7'
PURPLE_CELL = '#e7def0'
ORANGE = '#d97917'
ORANGE_BG = '#fff2e2'
RED = '#b33b32'
RED_BG = '#fff5f3'
GRAY = '#626262'
GRAY_BG = '#f3f3f1'

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 8,
    'mathtext.fontset': 'dejavuserif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig = plt.figure(figsize=(14.0, 8.8), facecolor='white')
ax = fig.add_axes([0.018, 0.025, 0.964, 0.95])
ax.set_xlim(0, 174)
ax.set_ylim(-25.5, 94)
ax.axis('off')


def panel(x, y, w, h, title, edge=INK, title_color=GOLD, lw=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor=edge,
                           linewidth=lw, zorder=0))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='center',
            color=title_color, fontsize=10.0, fontweight='bold', zorder=5)


def block(x, y, w, h, text, face, edge, size=7.0, weight='normal', color=INK,
          radius=0.7, z=2):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f'round,pad=0.12,rounding_size={radius}',
                           facecolor=face, edgecolor=edge, linewidth=0.9, zorder=z)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=size, color=color, fontweight=weight, linespacing=1.15,
            zorder=z + 1)


def arrow(x1, y1, x2, y2, color=INK, lw=1.0, style='-|>', z=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=9), zorder=z)


def line(points, color=INK, lw=1.0, z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=z)


# Figure title and top architecture panel headings.
ax.text(87, 92.0, 'AG-Mamba network architecture', ha='center', va='center',
        fontsize=14, fontweight='bold', color=GOLD)

# Top-row framed modules, arranged as a left-to-right workflow.
panel(2, 36, 19, 43, 'Input', edge=INK)
panel(24, 36, 51, 43, 'Aliased-Gabor (AG) filterbank', edge=BLUE, title_color=GOLD)
panel(78, 49, 15, 20, 'Projection', edge=INK)
panel(97, 36, 31, 43, 'Selective state-space backbone', edge=GREEN, title_color=GOLD)
panel(132, 36, 40, 43, 'Dual readout and classifier', edge=PURPLE, title_color=GOLD)

# Input waveform panel.
ax.text(11.5, 70.0, r'$x\in\mathbb{R}^{1\times 1024}$', ha='center', fontsize=8.4)
ax.text(11.5, 67.2, 'raw acceleration segment', ha='center', fontsize=6.8, color=GRAY)
t = np.linspace(0, 1, 220)
envelope = 0.65 + 0.26 * np.sin(2 * np.pi * 1.1 * t + 0.3)
signal = np.sin(2 * np.pi * 7.0 * t) * envelope + 0.16 * np.sin(2 * np.pi * 18 * t)
ax.plot(4.4 + 14.2 * t, 56.2 + 4.0 * signal, color=BLUE, lw=0.9, zorder=3)
line([(4.1, 56.2), (18.9, 56.2)], color='#b7b7b7', lw=0.45)
ax.text(11.5, 48.4, 'one window', ha='center', fontsize=7, color=GRAY)
ax.text(11.5, 45.6, 'length = 1024 samples', ha='center', fontsize=6.7)
ax.text(11.5, 42.9, 'same on all three benchmarks', ha='center', fontsize=6.0,
        color=GRAY)
ax.text(11.5, 39.2, 'no target-domain input\nat training time', ha='center', fontsize=6.4,
        color=GRAY, linespacing=1.2)

# Four convolutional layers and the two actual downsampling operations.
# Widths correspond to GaborBankFrontend with front_chan=512.
cell_x = [25.4, 33.0, 40.6, 48.2, 55.8, 63.4]
cell_w = 6.6
cell_y = 53.0
cell_h = 13.2
cell_text = [
    'Gabor\nConv 1\n1→64\nk=128, s=2',
    'MaxPool\n×2',
    'Conv 2\n64→128\nk=9',
    'Conv 3\n128→256\nk=7',
    'MaxPool\n×2',
    'Conv 4\n256→512\nk=5',
]
for i, (x, label) in enumerate(zip(cell_x, cell_text)):
    face, edge = (ORANGE_BG, ORANGE) if i == 0 else (BLUE_CELL, BLUE)
    block(x, cell_y, cell_w, cell_h, label, face, edge, size=6.5 if i != 0 else 6.2,
          weight='bold' if i == 0 else 'normal', radius=0.45)
    if i < len(cell_x) - 1:
        arrow(x + cell_w, cell_y + cell_h / 2,
              cell_x[i + 1] - 0.25, cell_y + cell_h / 2, color=BLUE, lw=0.8)

ax.text(49.8, 69.7, 'first-layer atoms: $K=64$ log-spaced Gabor channels',
        ha='center', fontsize=7.2, color=BLUE)
ax.text(49.8, 49.0, r'$1024\;\to\;513\;\to\;256\;\to\;128$ frames  (8$\times$ decimation)',
        ha='center', fontsize=7.0, fontweight='bold', color=BLUE)
block(27.0, 39.0, 45.0, 7.4,
      r'$q=0.5$  ·  $f_k\in[0.0025,0.45]$  ·  phase aliasing $\alpha=2.0$' + '\n'
      'analytic initialization; all frontend layers remain trainable',
      ORANGE_BG, ORANGE, size=7.0, radius=0.5)
arrow(21, 57.0, 24, 57.0, color=INK, lw=1.1)

# Projection: the projected encoder sequence fans out to both readout paths.
block(80.0, 57.4, 11.0, 7.1, 'Linear\n512 → d=128', GRAY_BG, GRAY,
      size=7.0, weight='bold')
block(80.0, 51.0, 11.0, 4.2, 'add learned\npositional embedding', '#fafafa', GRAY,
      size=6.1)
ax.text(85.5, 47.0, r'$U\in\mathbb{R}^{128\times128}$', ha='center', fontsize=6.8,
        color=GRAY)
arrow(75, 57.0, 79.7, 57.0)

# SSM sequence path and its three repeated selective blocks.
ax.text(111.5, 73.0, r'$d=128$, state size $N=8$', ha='center', fontsize=7.4,
        color=GREEN)
ax.text(111.5, 70.5, 'sequence length = 128', ha='center', fontsize=6.7, color=GRAY)
ssm_x, ssm_w, ssm_h = 101.5, 20.0, 6.0
ssm_ys = [61.8, 53.8, 45.8]
for j, y in enumerate(ssm_ys, start=1):
    block(ssm_x, y, ssm_w, ssm_h,
          f'Selective SSM block {j}' + '\ncausal conv  ·  input-dependent ' + r'$\Delta$',
          GREEN_CELL, GREEN, size=6.0, radius=0.45)
    if j < 3:
        arrow(ssm_x + ssm_w / 2, y, ssm_x + ssm_w / 2, y - 1.6,
              color=GREEN, lw=0.85)
arrow(93, 60.5, ssm_x - 0.4, 60.5, color=INK)
ax.text(111.5, 40.0, 'standard selective memory; no $\\tau_0$ modulation',
        ha='center', fontsize=6.4, color=GRAY)

# Dual temporal mean-max pooling and MLP classifier.
block(134.0, 58.0, 14.0, 7.0, 'Projected map $U$\nmean + max over time',
      PURPLE_CELL, PURPLE, size=6.4)
block(134.0, 47.0, 14.0, 7.0, 'SSM output $H$\nmean + max over time',
      PURPLE_CELL, PURPLE, size=6.4)
block(150.2, 52.5, 7.2, 8.0, 'concat\n$z$\n$4d=512$', PURPLE_BG, PURPLE,
      size=6.4, weight='bold')
block(159.0, 51.0, 11.4, 11.0,
      'MLP head\n512→256\nReLU · dropout 0.3\n256→$C$', PURPLE_BG, PURPLE,
      size=6.4, weight='bold')
ax.text(164.7, 43.8, r'$\hat{y}$ (class logits)', ha='center', fontsize=7.0,
        color=PURPLE)
arrow(148, 61.5, 150.0, 58.5, color=PURPLE, lw=0.9)
arrow(148, 50.5, 150.0, 54.5, color=PURPLE, lw=0.9)
arrow(157.5, 56.5, 158.7, 56.5, color=PURPLE, lw=0.9)
arrow(164.7, 50.8, 164.7, 45.2, color=PURPLE, lw=0.9)
# The projected frontend sequence also bypasses the SSM for direct pooling.
line([(93.0, 57.4), (94.5, 57.4), (94.5, 33.0), (141.0, 33.0), (141.0, 46.7)],
     color=PURPLE, lw=0.9)
arrow(141.0, 46.7, 141.0, 47.0, color=PURPLE, lw=0.9)
arrow(128, 48.8, 133.7, 50.0, color=PURPLE, lw=0.9)

# Distinguish training-only objective from inference, matching the reference's lower strip.
line([(2, 29.2), (172, 29.2)], color=RED, lw=0.8, z=1)
for x in np.arange(2, 172, 1.6):
    ax.plot([x, min(x + 0.75, 172)], [29.2, 29.2], color=RED, lw=0.9, zorder=2)

panel(2, 3.8, 85, 22.0, 'Inference panel  ·  unseen target domain', edge=INK)
block(5.0, 9.0, 18.0, 10.0, 'unseen segment\n' + r'$x\in\mathbb{R}^{1\times1024}$',
      ORANGE_BG, ORANGE, size=7.0, weight='bold')
block(29.0, 9.0, 27.0, 10.0, 'AG filterbank\n+ projection\n+ 3 selective SSM blocks',
      BLUE_BG, BLUE, size=7.0, weight='bold')
block(62.0, 9.0, 17.0, 10.0, 'dual pooling\n+ classifier', PURPLE_BG, PURPLE,
      size=7.0, weight='bold')
arrow(23.2, 14.0, 28.7, 14.0, color=ORANGE, lw=1.2)
arrow(56.2, 14.0, 61.7, 14.0, color=ORANGE, lw=1.2)
arrow(79.2, 14.0, 83.5, 14.0, color=ORANGE, lw=1.2)
ax.text(85.0, 14.0, r'$\hat{y}$', ha='center', va='center', fontsize=9,
        color=PURPLE, fontweight='bold')
ax.text(44.5, 6.0, 'No target labels, target statistics, or test-speed metadata are used.',
        ha='center', fontsize=6.7, color=GRAY)

panel(91, 3.8, 81, 22.0, 'Training phase  ·  source domains only', edge=INK)
block(94.0, 10.0, 32.0, 8.3,
      r'$\mathcal{L}_{task}=\mathrm{CE}(\hat{y},y)$' + '\n'
      r'$\mathcal{L}=\mathcal{L}_{task}+0.01\,\overline{\Delta^{2}}$',
      GRAY_BG, GRAY, size=8.1, weight='bold')
ax.text(110.0, 7.2, 'classification plus mean-squared step regularisation',
        ha='center', fontsize=6.7, color=GRAY)
block(130.0, 8.0, 39.0, 12.0,
      'NOT INSTANTIATED\nNo MMD/JMMD, domain discriminator, GRL, or adversarial loss\n'
      'No $\\tau_0$ estimator or liquid-time-constant path',
      RED_BG, RED, size=6.5, weight='normal', color=RED, radius=0.45)

# ---------------------------------------------------------------------------
# Bottom strip: internals of one selective SSM block (the green panel above).
# Mirrors LiquidMambaBlock.forward in models/agmamba_backbone.py.
# ---------------------------------------------------------------------------
panel(2, -21.5, 170, 17.5,
      'Selective SSM block  ·  detail of each green block above', edge=GREEN,
      title_color=GOLD)
ax.text(87, -8.6,
        r'three identical blocks  ·  $d_{\mathrm{model}}=d_{\mathrm{inner}}=128$,'
      r'  state size $N=8$,  step rank 16  ·  residual + branch LayerNorm',
        ha='center', fontsize=6.9, color=GRAY)

byt, bh = -14.0, 4.4
specs = [
    (4.5, 20.0, r'Linear $d\to 2d$' + '\n(in_proj)'),
    (26.5, 9.0, 'split'),
    (37.5, 22.0, 'depthwise causal conv\n$k=4$, left-padded'),
    (61.5, 8.5, 'SiLU'),
    (72.0, 22.0, 'selective scan\n' + r'$\bar{A}=e^{\Delta A}$'),
    (96.0, 9.0, r'$\times$'),
    (108.0, 22.0, 'Linear $2d\to d$' + '\n(out_proj)'),
    (132.0, 30.0, 'add residual  +  LayerNorm'),
]
for i, (bx, bw, txt) in enumerate(specs):
    face, edge = (GREEN_CELL, GREEN) if i in (2, 4) else (GRAY_BG, GRAY)
    block(bx, byt, bw, bh, txt, face, edge, size=6.6, radius=0.4)
    if i < len(specs) - 1:
        arrow(bx + bw + 0.3, byt + bh / 2, specs[i + 1][0] - 0.3, byt + bh / 2,
              color=GRAY, lw=0.8)

# Gate branch: the second half of in_proj, SiLU-activated and multiplied into the scan output.
line([(31.0, byt), (31.0, byt - 2.2), (100.5, byt - 2.2)], color=ORANGE, lw=0.8)
arrow(100.5, byt - 2.2, 100.5, byt - 0.15, color=ORANGE, lw=0.8)
ax.text(66.0, byt - 3.5, r'gate branch:  $\mathrm{SiLU}(z)$', ha='center',
        fontsize=6.5, color=ORANGE)

# Step-size and state-matrix parameterisation, as implemented (Equations (14) and (16)).
ax.text(87.0, -20.6,
        r'$\Delta=\mathrm{clamp}(\mathrm{softplus}(W_\Delta u+b_\Delta),\,10^{-3},\,20)$, '
      r' $B=W_Bu$,  $C=W_Cu$  ·  $A=-\mathrm{diag}(e^{a})$,  $|A|$ linearly spaced on $[0.5,3.0]$',
        ha='center', fontsize=6.5, color=GRAY)

# Small, publication-oriented footer; all absent components are identified in the objective panel.
ax.text(3, -24.4,
        'G1 motivation: Section 3.3  ·  $\\alpha=2.0$, $q=0.5$  ·  SSM and alignment audits: Section 6.4',
        ha='left', fontsize=6.3, color=GRAY)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIGD, 'fig3_arch.' + ext), dpi=350,
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print('[saved] %s' % os.path.join(FIGD, 'fig3_arch.pdf'))
