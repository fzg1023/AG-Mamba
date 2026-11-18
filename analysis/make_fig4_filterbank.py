# -*- coding: utf-8 -*-
"""Fig. 4 —— AG 滤波器组的机制合图（6 面板）。

合并并取代 v0.2 的 Figs 5–10：
  (a) 常 Q 原子波形＋高斯包络        （v0.2 Fig 5）
  (b) 同一批原子的对数轴幅度谱        （v0.2 Fig 6）
  (c) 每倍频程能量覆盖 Ĉ(u)            （v0.2 Fig 7，改为同时画出未混叠与 α=2）
  (d) 三种原子机制对照（α=0 / α=2 / 宽核族）（v0.2 Fig 8）
  (e) 折叠 Bessel 梳验证               （v0.2 Fig 9）
  (f) (ζ, Π) 随 α 迁移                 （取代 v0.2 Fig 10 的 χ 面板）

口径同 make_fig2_design_space.py：交付配置 + ζ = 过零率 + Π = 谱峰均比（nfft=16384）。

输出
----
docs/manuscript/figures/fig4_filterbank.pdf / .png
docs/manuscript/figures/_fig4_values.txt
"""
import math
import os
import sys

import numpy as np
from scipy.special import jv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

# 优先用仓库真实实现；torch 不可用时回退到已数值核验的 numpy 等价版（同 fig 2）。
try:                                                       # pragma: no cover
    import torch                                            # noqa: F401
    from models.agmamba_backbone import gabor_bank as _gb_ref
    from models.widan_net import Laplace_fast as _lap_ref
    HAS_TORCH = True
except Exception:                                          # pragma: no cover
    HAS_TORCH = False

# --------------------------------------------------- 交付配置（同图 2）
KS, C = 128, 64
F_LO, F_HI = 0.0025, 0.45
SIGMA_RATIO, DC_FRAC = 0.5, 0.3
FM_LO, FM_HI = 0.20, 0.48
NFFT, BAND = 16384, 0.45
CF = np.exp(np.linspace(math.log(F_LO), math.log(F_HI), C))     # 中心频率
FM = np.exp(np.linspace(math.log(FM_LO), math.log(FM_HI), C))   # 调制频率


def bank(alias):
    """AG 滤波器组（交付配置）。torch 在则用仓库实现，否则用 numpy 等价版。"""
    if HAS_TORCH:
        return _gb_ref(C, KS, F_LO, F_HI, SIGMA_RATIO, DC_FRAC,
                       float(alias), FM_LO, FM_HI).numpy()[:, 0, :]
    t = np.arange(KS, dtype=np.float64) - (KS - 1) / 2.0
    sg = SIGMA_RATIO / CF[:, None]
    ph = 2.0 * math.pi * CF[:, None] * t[None, :]
    if alias > 0:
        ph = ph + float(alias) * np.sin(2.0 * math.pi * FM[:, None] * t[None, :])
    psi = np.exp(-(t[None, :] ** 2) / (2.0 * sg ** 2)) * np.cos(ph)
    n_dc = int(round(C * DC_FRAC))
    psi[n_dc:] -= psi[n_dc:].mean(axis=-1, keepdims=True)
    return psi / np.clip(np.linalg.norm(psi, axis=-1, keepdims=True), 1e-6, None)


def laplace_bank(ks=64, c=64):
    """宽核 Laplace 参照族 [10]；源码为 exp(sigmoid(arg))（有界于 [1,e]）。"""
    if HAS_TORCH:
        return _lap_ref(out_channels=c, kernel_size=ks, eps=0.1,
                        frequency=64000, mode='sigmoid').forward().numpy()[:, 0, :]
    a_ = np.linspace(0, c, c)[:, None]
    td = np.linspace(0, ks - 1, ks)
    q = 1.0 - 0.03 ** 2
    w = 2.0 * math.pi * 64000.0
    p_ = (td[None, :] - a_) / (a_ + 0.1)
    arg = (-0.03 / math.sqrt(q)) * (w * (p_ - 0.1))
    sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -700.0, 700.0)))
    return (1.0 / math.e) * np.exp(sig) * np.sin(w * (p_ - 0.1))


def env_of(k):
    """第 k 个原子的高斯包络（与 gabor_bank 一致，未做 DC/归一）。"""
    t = np.arange(KS, dtype=np.float64) - (KS - 1) / 2.0
    return np.exp(-(t ** 2) / (2.0 * (SIGMA_RATIO / CF[k]) ** 2))


def zeta_rate(a):
    return float(((a[:, :-1] * a[:, 1:]) < 0).sum(axis=-1).mean()) / (a.shape[-1] - 1)


def peak_to_mean(a, nfft=NFFT, band=BAND):
    sp = np.abs(np.fft.rfft(a, n=nfft, axis=-1)) ** 2
    u = np.fft.rfftfreq(nfft)
    m = (u > 1e-9) & (u <= band)
    sp = sp[:, m]
    return float((sp.max(axis=-1) / np.clip(sp.mean(axis=-1), 1e-30, None)).mean())


LOG = []
B0 = bank(0.0)
B2 = bank(2.0)
LAP = laplace_bank()

# ------------------------------------------------------------------ 通用工具
def norm(a):
    return a / max(np.abs(a).max(), 1e-12)


def peak_freq(kernel, nfft=4096):
    """Dominant frequency of a kernel, in cycles/sample (DC bin excluded)."""
    sp = np.abs(np.fft.rfft(kernel, n=nfft))
    u = np.fft.rfftfreq(nfft)
    return float(u[int(np.argmax(sp[1:])) + 1])


def cap(ax, text, y=-0.30, size=5.6):
    """Panel caption drawn just below the axes (single source of truth)."""
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.25', linespacing=1.35)


def tidy(ax, keep_left=True):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=6.0)


# -------------------------------------------------- 面板共用的选点与预计算
IDX = [int(round(63 * math.log(f / F_LO) / math.log(F_HI / F_LO)))
       for f in (0.005, 0.02, 0.06, 0.15)]
# 选点避开退化区：本方法核组中 sigma_t = 0.5/f_k < 2 个采样点的 8 个通道
# （f_k > 0.253）已不能形成窄带带通，在 (a)/(b) 中作为示例无意义；
# 其存在已在 §4.2.1 正文写明。
IDX = [min(max(i, 0), C - 1) for i in IDX]
KC = IDX[2]

# (d) 的 Laplace 参照核**按主频匹配**选取：写死索引会把两个相差数倍的核画在一起
LAP_PEAKS = np.array([peak_freq(LAP[i]) for i in range(LAP.shape[0])])
JL = int(np.argmin(np.abs(LAP_PEAKS - CF[KC])))

# (e) 的验证原子（与 §4.2.2 正文引用的同一个通道）
KE = int(np.argmin(np.abs(CF - 0.049)))
FK_B, FM_B = CF[KE], FM[KE]
ALPHA_BESSEL = 2.0

CAP_A = ('(a) Constant-Q Gabor atoms at four log-spaced centres\n'
         r'$f_k$ (cycles/sample) of the proposed bank; dashed:' '\n'
         r'Gaussian envelope $\exp(-t^2/2\sigma_k^2)$, $\sigma_k=q/f_k$, $q=0.5$.')
CAP_B = ('(b) Amplitude spectra of the same four atoms (log\n'
         'frequency axis; each normalised to its own maximum\n'
         'and offset). One lobe per atom: constant relative bandwidth.')
CAP_C = ('(c) Energy per octave $\\tilde{C}(u)$ of Eq. (10). Unaliased\n'
         'bank (solid): flat to within 1.17$\\times$ over the shaded\n'
         '4.6 octaves. Dispersed bank at $\\alpha=2$ (dashed): 109$\\times$.')
CAP_D = ('(d) Three atom regimes at matched centre frequency\n'
         '($f_c=%.4f$ cycles/sample): analytic Gabor ($\\alpha=0$),\n'
         'dispersed Gabor ($\\alpha=2$) and a wide Laplace kernel [10];\n'
         'each normalised to unit peak, Laplace at its own length 64.') % CF[KC]
CAP_E = ('(e) Folded Bessel comb at $\\alpha=2$ for the atom at\n'
         '$f_k=%.4f$ ($f_{m,k}=%.4f$): measured spectrum (grey)\n'
         'against the line positions of Eqs. (12)-(13) (red).') % (FK_B, FM_B)
CAP_F = ('(f) Mechanism statistics of Section 3.3 versus $\\alpha$:\n'
         '$\\Pi$ peak-to-mean (left, blue) and $\\zeta$ zero-crossing\n'
         'rate (right, red). The proposed $\\alpha=2$ is dotted.')


# ----------------------------------------------------------------------- (a)
def draw_a(ax):
    off = 2.1
    for j, k in enumerate(IDX):
        ax.plot(B0[k] + j * off, color='#1f77b4', lw=0.9)
        ax.plot(env_of(k) + j * off, 'k--', lw=0.5, alpha=0.7)
        ax.text(KS * 0.985, j * off, r'$f_k=%.4f$' % CF[k], fontsize=5.0,
                ha='right', va='bottom', color='0.3')
    ax.set_xlabel('sample index', fontsize=6.5)
    ax.set_yticks([])
    tidy(ax, keep_left=False)


# ----------------------------------------------------------------------- (b)
def draw_b(ax):
    for j, k in enumerate(IDX):
        sp = np.abs(np.fft.rfft(B0[k], n=NFFT))
        u = np.fft.rfftfreq(NFFT)
        m = (u > 1e-5) & (u <= 0.5)
        ax.plot(u[m], sp[m] / sp[m].max() + j * 0.55, color='#1f77b4', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 0.5)
    ax.set_ylim(-0.1, len(IDX) * 0.55 + 0.5)
    ax.set_xlabel('normalised frequency $u$ (cycles/sample)', fontsize=6.5)
    ax.set_yticks([])
    tidy(ax, keep_left=False)

# ----------------------------------------------------------------------- (c)
def draw_c(ax):
    u = np.fft.rfftfreq(NFFT)
    for B, lab, st, cl in ((B0, 'unaliased (Eq. 9)', '-', '#1f77b4'),
                           (B2, r'$\alpha=2$ (proposed)', '--', '#d62728')):
        F = np.abs(np.fft.rfft(B, n=NFFT, axis=-1)) ** 2
        Cov = u * F.sum(axis=0)                   # energy per octave
        m = (u >= 1e-4) & (u <= 0.5)
        ax.plot(u[m], Cov[m] / Cov[m].max(), st, color=cl, lw=1.0, label=lab)
        sel = (u >= 0.015) & (u <= 0.375)
        LOG.append('coverage max/min over u in [0.015,0.375]: %s -> %.2fx'
                   % (lab, Cov[sel].max() / max(Cov[sel].min(), 1e-30)))
    ax.axvspan(0.015, 0.375, color='0.85', alpha=0.6, lw=0, zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel('$u$ (cycles/sample)', fontsize=6.5)
    ax.set_ylabel(r'$\tilde{C}(u)$ (norm.)', fontsize=6.5)
    ax.legend(fontsize=5.4, frameon=False, loc='lower center')
    tidy(ax)


# ----------------------------------------------------------------------- (d)
def draw_d(ax):
    """Three regimes at a *matched* centre frequency.

    The Laplace kernel is picked by matching its measured dominant frequency
    (cycles/sample) to that of the AG atoms, so the three curves really are
    comparable; a hard-coded index can put them an octave apart.
    """
    for a, nm, cl in ((B0[KC], r'AG $\alpha=0$', '#1f77b4'),
                      (B2[KC], r'AG $\alpha=2$', '#d62728'),
                      (LAP[JL], 'wide kernel [10]', '#2ca02c')):
        ax.plot(norm(a), color=cl, lw=0.9, label=nm)
    ax.set_xlabel('sample index', fontsize=6.5)
    ax.set_ylabel('amplitude (norm.)', fontsize=6.5)
    ax.set_ylim(-1.18, 1.55)
    ax.legend(fontsize=5.0, frameon=True, framealpha=0.95, edgecolor='none',
              ncol=3, loc='upper center')
    LOG.append('panel (d): AG f_c=%.4f (idx %d); Laplace idx %d peak %.4f'
               % (CF[KC], KC, JL, LAP_PEAKS[JL]))
    tidy(ax)

# ----------------------------------------------------------------------- (e)
def draw_e(ax):
    sp = np.abs(np.fft.rfft(B2[KE], n=NFFT))
    uu = np.fft.rfftfreq(NFFT)
    keep = (uu > 0) & (uu <= 0.5)
    ax.plot(uu[keep], sp[keep] / sp[keep].max(), color='0.35', lw=0.8,
            label='measured')
    lines_at = {}
    for n in np.arange(-14, 15):
        raw = FK_B + n * FM_B
        fr = raw - np.floor(raw)
        fu = min(fr, 1.0 - fr)                     # discrete folding F(u)
        w = abs(jv(n, ALPHA_BESSEL))
        if w < 1e-4 or fu <= 1e-6 or fu > 0.5:
            continue
        lines_at[round(fu, 6)] = lines_at.get(round(fu, 6), 0.0) + w
    xs = sorted(lines_at)
    ys = np.array([lines_at[x] for x in xs])
    ys = ys / max(np.max(ys), 1e-12)
    for x, y in zip(xs, ys):
        ax.plot([x, x], [0, y], color='#d62728', lw=0.9)
    ax.plot(xs, ys, 'r.', ms=1.6)
    ax.plot([], [], color='#d62728', lw=0.9, label=r'folded lines, $|J_n(2)|$')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 0.5)
    ax.set_xlabel('$u$ (cycles/sample)', fontsize=6.5)
    ax.set_ylabel('normalised amplitude', fontsize=6.5)
    ax.legend(fontsize=5.4, frameon=False)
    tidy(ax)
    LOG.append('Bessel check: f_k=%.4f  f_m=%.4f  J0=%.3f  J1=%.3f  J2=%.3f'
               % (FK_B, FM_B, jv(0, ALPHA_BESSEL), jv(1, ALPHA_BESSEL),
                  jv(2, ALPHA_BESSEL)))


# ----------------------------------------------------------------------- (f)
AL = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
zs = [zeta_rate(bank(a)) for a in AL]
ps = [peak_to_mean(bank(a)) for a in AL]


def draw_f(ax):
    ax.plot(AL, ps, '-o', color='#1f77b4', ms=3.0, lw=1.0)
    ax.set_xlabel(r'aliasing strength $\alpha$', fontsize=6.5)
    ax.set_ylabel(r'$\Pi$ (peak-to-mean)', fontsize=6.5, color='#1f77b4')
    ax.tick_params(axis='y', labelsize=6.0, colors='#1f77b4')
    ax.tick_params(axis='x', labelsize=6.0)
    ax2 = ax.twinx()
    ax2.plot(AL, zs, '--s', color='#d62728', ms=3.0, lw=1.0)
    ax2.set_ylabel(r'$\zeta$ (zero-crossing rate)', fontsize=6.5, color='#d62728')
    ax2.tick_params(axis='y', labelsize=6.0, colors='#d62728')
    ax.axvline(2.0, color='0.6', lw=0.7, ls=':')
    ax.spines['top'].set_visible(False)


# =========================== 输出 1：3x2 合成图（图注置于各面板下方）========
PANELS = [('a', draw_a, CAP_A), ('b', draw_b, CAP_B), ('c', draw_c, CAP_C),
          ('d', draw_d, CAP_D), ('e', draw_e, CAP_E), ('f', draw_f, CAP_F)]

fig = plt.figure(figsize=(7.4, 8.6))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=1.30, wspace=0.34)
for i, (key, fn, caption) in enumerate(PANELS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    fn(ax)
    cap(ax, caption)
fig.suptitle('AG filterbank: atoms, dispersion and coverage', fontsize=8.5,
             y=0.995)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIGD, 'fig4_filterbank.' + ext), dpi=400,
                bbox_inches='tight')
plt.close(fig)

# =========================== 输出 2：单面板图（便于后续排版拼版）============
for key, fn, caption in PANELS:
    f1 = plt.figure(figsize=(3.6, 3.3))
    ax = f1.add_axes([0.17, 0.42, 0.79, 0.53])
    fn(ax)
    ax.text(0.0, -0.28, caption, transform=ax.transAxes, ha='left', va='top',
            fontsize=6.3, color='0.25', linespacing=1.35)
    for ext in ('pdf', 'png'):
        f1.savefig(os.path.join(FIGD, 'fig4' + key + '_panel.' + ext), dpi=400,
                   bbox_inches='tight')
    plt.close(f1)

with open(os.path.join(FIGD, '_fig4_values.txt'), 'w', encoding='utf-8') as fh:
    fh.write('# Fig. 4 数值\n')
    for a, z, p in zip(AL, zs, ps):
        fh.write('alpha=%.1f  zeta=%.4f  Pi=%8.3f\n' % (a, z, p))
    fh.write('Laplace ref: zeta=%.4f  Pi=%8.3f\n' % (zeta_rate(LAP), peak_to_mean(LAP)))
    for ln in dict.fromkeys(LOG):
        fh.write(ln + '\n')
print('[saved] %s' % os.path.join(FIGD, 'fig4_filterbank.pdf'))
for key, _, _ in PANELS:
    print('[saved] %s' % os.path.join(FIGD, 'fig4%s_panel.pdf' % key))
for ln in LOG:
    print(ln)
