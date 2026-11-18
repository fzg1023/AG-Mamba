# -*- coding: utf-8 -*-
"""Fig. 1–3 合成总览出图（MSSP 工作稿）。
(a) 膨胀轨道：HUST O 类 20/40/80 Hz 平均幅度谱堆叠 + 谐波刻线 + 扇形连线；
# (b) 核统计 (ζ, Π)：Laplace 参考 / Gabor α 路径 / 随机；
(c) 五机制探针精度（T0/T9，max 口径、5 种子均值±标准差）。
输出: docs/manuscript/figures/fig1-3_composite.pdf / fig1-3_composite.png（三面板合成总览；正式分图为 fig1/fig2/fig3）
"""
import os
import re
import glob
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
sys.path.insert(0, ROOT)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)
torch.manual_seed(0)

# ---------------- (a) HUST spectra ----------------
Z = np.load(os.path.join(ROOT, 'datasets', 'hust_data', 'hust_x1024.npz'))
FS = 25600.0
SPEEDS = [20, 40, 80]
CCLS = {20: '#0072B2', 40: '#009E73', 80: '#D55E00'}
F_LOW, F_HIGH = 20.0, 650.0
OFFS = {20: 80.0, 40: 40.0, 80: 0.0}
DROP = 34.0


def mean_spec(key, nfft=4096, nseg=256):
    """平均功率谱（口径同 make_fig1_panels.py；2026-10-02 去直流 + 功率均值）。"""
    sig = Z[key][:nseg].astype(np.float64)
    sig = sig - sig.mean(axis=1, keepdims=True)
    sig = sig / (sig.std(axis=1, keepdims=True) + 1e-12)
    sig = sig * np.hanning(sig.shape[1])
    S = (np.abs(np.fft.rfft(sig, n=nfft, axis=-1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    return f, S


# ---------------- (b) kernel statistics ----------------
from models.agmamba_backbone import gabor_bank          # noqa: E402
from models.widan_net import Laplace_fast                 # noqa: E402

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402


def kstats(k):
    """返回 (ζ 过零率, Π 谱峰均比)。

    2026-09-30 换口径：原实现用 nfft=1024、按幅度归一算“峰值 bin 占比”，该量在
    2048→65536 点网格上漂移 190×，不满足可报告性；改用 make_fig2_design_space.py
    中的 ζ（过零率）与 Π（谱峰均比，nfft=16384、带 u∈(0,0.45]）。
    """
    a = k.numpy()[:, 0, :]
    zc = float(((a[:, :-1] * a[:, 1:]) < 0).sum(axis=-1).mean()) / (a.shape[-1] - 1)
    sp = np.abs(np.fft.rfft(a, n=16384, axis=-1)) ** 2
    u = np.fft.rfftfreq(16384)
    m = (u > 1e-9) & (u <= 0.45)
    sp = sp[:, m]
    conc = float((sp.max(axis=-1) / np.maximum(sp.mean(axis=-1), 1e-30)).mean())
    return zc, conc


p_laplace = kstats(Laplace_fast(out_channels=64, kernel_size=64, eps=0.1,
                                frequency=64000, mode='sigmoid').forward())
p_rand = kstats(torch.randn(64, 1, 64) * 0.022)
alphas = [0.0, 0.5, 1.0, 2.0, 3.0]
p_gabor = [kstats(gabor_bank(64, 64, 0.005, 0.45, 0.6, 0.3, alias=a))
           for a in alphas]
print('(b) laplace zeta=%.2f chi=%.4f | rand zeta=%.2f chi=%.4f'
      % (*p_laplace, *p_rand))
for a, p in zip(alphas, p_gabor):
    print('    alpha=%.1f -> zeta=%.2f chi=%.4f' % (a, p[0], p[1]))

# ---------------- (c) probe accuracy ----------------
RD = os.path.join(ROOT, 'results')


def parse_max(f):
    s = open(f, errors='ignore').read()
    v = [float(x) for x in re.findall(r'tgt_acc:\s*([0-9.]+)', s)]
    return max(v) * 100 if v else float('nan')


VARIANTS = [('wide_gabor', 'Gabor\n($\\alpha$=0)'), ('mix', 'Mix'),
            ('alias1', '$\\alpha$=1'), ('alias2', '$\\alpha$=2'),
            ('wide_rand', 'Random')]
acc = {}
for v, _ in VARIANTS:
    for t in ['T0', 'T9']:
        vals = [parse_max(f) for f in sorted(glob.glob(_findlog('HUST_%s_AGMamba_fc_gw_%s_seed*.log' % (t, v))))]
        vals = [x for x in vals if not np.isnan(x)]
        acc[(v, t)] = (float(np.mean(vals)), float(np.std(vals)))
for v, _ in VARIANTS:
    print('(c) %-11s T0=%.2f±%.2f  T9=%.2f±%.2f'
          % (v, *acc[(v, 'T0')], *acc[(v, 'T9')]))

# ---------------- figure ----------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 7, 'legend.frameon': False,
    'mathtext.fontset': 'dejavusans',
})

fig = plt.figure(figsize=(7.4, 4.9))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.12, 1.0],
                       hspace=0.62, wspace=0.30)

# ---- (a) stacked dilation orbit ----
axa = fig.add_subplot(gs[0, :])
curves = []
for sp in SPEEDS:
    f, Pw = mean_spec('%d_O' % sp)
    m = (f >= F_LOW) & (f <= F_HIGH)
    db = 10.0 * np.log10(np.maximum(Pw[m], 1e-30))
    db = db - db.max() + OFFS[sp]
    h, = axa.plot(f[m], db, color=CCLS[sp], lw=0.85, zorder=3)
    curves.append(h)
    axa.text(F_HIGH - 3, OFFS[sp] - 1.5, '%d Hz' % sp, color=CCLS[sp],
             fontsize=7.5, ha='right', va='top')
    for k in range(1, 17):
        pos = k * sp
        if pos <= F_HIGH:
            axa.plot([pos], [OFFS[sp] - DROP], marker='|', ms=6.5,
                     color=CCLS[sp], mew=1.0, zorder=4)
rays = None
for k in range(1, 9):
    ln, = axa.plot([k * sp for sp in SPEEDS],
                   [OFFS[sp] - DROP + 3.0 for sp in SPEEDS],
                   color='0.55', lw=0.6, zorder=0)
    if rays is None:
        rays = ln
axa.set_xlim(0, F_HIGH)
axa.set_ylim(-38, 88)
axa.set_yticks([])
axa.set_xlabel('Frequency (Hz)', labelpad=1.5)
axa.set_ylabel('mean power (dB, offset)', labelpad=1.5)
for s in ['left', 'right', 'top']:
    axa.spines[s].set_visible(False)
ruler, = axa.plot([], [], marker='|', ls='none', ms=6.5, mew=1.0,
                  color='0.35')
axa.legend(curves + [ruler, rays],
           ['mean power spectrum, %d Hz' % sp for sp in SPEEDS] +
           ['multiples of $f_r$ (ruler; colour = speed)',
            'the $k$-th multiple tracked across speeds (dilation ray)'],
           loc='upper left', bbox_to_anchor=(0.004, 0.795), fontsize=6.2,
           frameon=True, framealpha=0.93, handlelength=1.5,
           labelspacing=0.32, borderpad=0.4)
axa.text(0.008, 0.97, '(a)', transform=axa.transAxes,
         fontweight='bold', fontsize=9, va='top')
axa.text(0.995, 0.02,
         'each curve normalised to its own maximum in 20-650 Hz; offsets arbitrary',
         transform=axa.transAxes, fontsize=6.0, ha='right', color='0.35')

# ---- (b) kernel statistics plane ----
axb = fig.add_subplot(gs[1, 0])
pathx = [p[0] for p in p_gabor]
pathy = [p[1] for p in p_gabor]
axb.plot(pathx, pathy, '-', color='#d62728', lw=0.9, marker='o', ms=3.2)
lab_off = {0.0: (4, -10), 1.0: (3, 6), 2.0: (2, 6), 3.0: (3, -11)}
for a, (x, y) in zip(alphas, p_gabor):
    if a in lab_off:
        axb.annotate(r'$\alpha$=%.0f' % a, (x, y), textcoords='offset points',
                     xytext=lab_off[a], fontsize=6.5, color='#d62728')
axb.plot([p_laplace[0]], [p_laplace[1]], marker='P', ms=6, color='black')
axb.annotate('wide-kernel ref. [10]', p_laplace, textcoords='offset points',
             xytext=(-5, -12), fontsize=6.5, ha='right')
axb.plot([p_rand[0]], [p_rand[1]], marker='^', ms=6, color='#555555')
axb.annotate('random init', p_rand, textcoords='offset points',
             xytext=(-3, 7), fontsize=6.5, ha='right', color='#333333')
axb.set_xlim(0, 38)
axb.set_ylim(0, 0.031)
axb.set_xlabel(r'zero-crossing rate $\zeta$', labelpad=1.5)
axb.set_ylabel(r'spectral peak-to-mean ratio $\Pi$', labelpad=1.5)
axb.text(0.02, 0.90, '(b)', transform=axb.transAxes,
         fontweight='bold', fontsize=9)

# ---- (c) probe accuracies ----
axc = fig.add_subplot(gs[1, 1])
x = np.arange(len(VARIANTS))
w = 0.36
t0 = [acc[(v, 'T0')][0] for v, _ in VARIANTS]
t0e = [acc[(v, 'T0')][1] for v, _ in VARIANTS]
t9 = [acc[(v, 'T9')][0] for v, _ in VARIANTS]
t9e = [acc[(v, 'T9')][1] for v, _ in VARIANTS]
axc.bar(x - w / 2, t0, w, yerr=t0e, capsize=1.8, color='#4c72b0',
        label='T0 (20 Hz)', error_kw=dict(lw=0.7))
axc.bar(x + w / 2, t9, w, yerr=t9e, capsize=1.8, color='#dd8452',
        label='T9 (80 Hz)', error_kw=dict(lw=0.7))
for xi in range(len(VARIANTS)):
    axc.text(xi - w / 2, t0[xi] + t0e[xi] + 1.0, '%.1f' % t0[xi],
             ha='center', fontsize=5.8)
    axc.text(xi + w / 2, t9[xi] + t9e[xi] + 1.0, '%.1f' % t9[xi],
             ha='center', fontsize=5.8)
axc.set_xticks(x)
axc.set_xticklabels([lab for _, lab in VARIANTS], fontsize=7)
axc.set_ylim(50, 108)
axc.set_yticks([50, 60, 70, 80, 90, 100])
axc.set_ylabel('target accuracy (%)', labelpad=1.5)
axc.legend(loc='upper right', fontsize=6.5, ncol=2, handlelength=1.2,
           columnspacing=1.0)
axc.text(0.02, 0.90, '(c)', transform=axc.transAxes,
         fontweight='bold', fontsize=9)

fig.savefig(os.path.join(FIGD, 'fig1-3_composite.pdf'))
fig.savefig(os.path.join(FIGD, 'fig1-3_composite.png'), dpi=300)
print('saved:', os.path.join(FIGD, 'fig1-3_composite.pdf'), '/ fig1-3_composite.png')
