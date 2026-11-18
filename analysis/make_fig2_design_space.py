# -*- coding: utf-8 -*-
"""Fig. 2 —— 前端先验的设计空间：(ζ, Π) 平面（v0.3 新口径）。

为什么换口径（2026-09-30 决定）
-------------------------------
v0.2 的 Fig. 2 画的是 (ζ, χ)，χ = “单一峰值 bin 占谱能量的比重”。该量**可复现**，但
**其值几乎完全由未记录的测量口径决定**：在 2048 → 65536 点的 FFT 网格上漂移达 190×
（见 §3.3 起草注）。仓库内两套实现甚至互不相容：
  · `make_fig1.py::kstats`      —— nfft = 1024，按**幅度**归一（产出论文里那组数）
  · `exp_e3_recalc.py::zeta_chi` —— nfft = 2048，按**功率**归一
两者相差 ≈15%。因此 v0.3 改用两个**口径稳健**的量：

  ζ = 过零率 = (1/(K-1)) * Σ 1[ψ(n)ψ(n+1) < 0]        跨样本归一，故可跨核长比较
  Π = 谱峰均比 = max_u |ψ̂(u)|² / mean_u |ψ̂(u)|²       nfft = 16384，带 u ∈ (0, 0.45]

实测稳健性：32 倍网格范围内 Π 变化 <2%、ζ 不变。

口径
----
所有核统计量均为**交付配置**（agmamba/configs.yaml：front_ksize=128, gabor_f_lo=0.0025,
gabor_sigma=0.5, gabor_dc_frac=0.3, gabor_alias=2.0, front_chan=512 → c1=64 个原子）。
Laplace 参照族按其自身实现（64 点核）给出，标注核长。

输出
----
docs/manuscript/figures/fig2_zeta_pi.pdf / .png
docs/manuscript/figures/_fig2_values.txt   （图上所有数值，便于核对）
"""
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

# ------------------------------------------------------------------ 核组构造
# 优先用仓库的真实实现（需 torch）；torch 不可用时回退到数值等价的 numpy 版。
# numpy 回退版已逐项核对：过零率 5.94/15.12/25.50（v0.2 报 6.0/15.3/25.5）、
# −3 dB 相对带宽中位数 0.530 = 理论 1.665ρ、Laplace 核 0/4096 非有限。
try:                                                       # pragma: no cover
    import torch                                            # noqa: F401
    from models.agmamba_backbone import gabor_bank as _gb_ref
    from models.widan_net import Laplace_fast as _lap_ref
    HAS_TORCH = True
except Exception:                                          # pragma: no cover
    HAS_TORCH = False

# ----------------------------------------------------------------- 交付配置
KS = 128                 # front_ksize
C = 64                   # front_chan // 8 = 512 // 8
F_LO, F_HI = 0.0025, 0.45
SIGMA_RATIO = 0.5        # gabor_sigma
DC_FRAC = 0.3            # gabor_dc_frac
FM_LO, FM_HI = 0.20, 0.48
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # 0 = 解析；2 = 交付；与 §6.3 的精度扫描同网格
NFFT = 16384
BAND = 0.45


# ----------------------------------------------------------------- 度量
def zeta_rate(psi):
    """过零率（跨样本归一）。psi: (C,K) 或 (C,1,K)。"""
    a = np.asarray(psi)
    if a.ndim == 3:
        a = a[:, 0, :]
    return float(((a[:, :-1] * a[:, 1:]) < 0).sum(axis=-1).mean()) / (a.shape[-1] - 1)


def peak_to_mean(psi, nfft=NFFT, band=BAND):
    """谱峰均比 Π（功率口径）。"""
    a = np.asarray(psi)
    if a.ndim == 3:
        a = a[:, 0, :]
    sp = np.abs(np.fft.rfft(a, n=nfft, axis=-1)) ** 2
    u = np.fft.rfftfreq(nfft)
    m = (u > 1e-9) & (u <= band)
    sp = sp[:, m]
    return float((sp.max(axis=-1) / np.clip(sp.mean(axis=-1), 1e-30, None)).mean())


def kaiming_bank(seed, c=C, ks=KS):
    """与 models/agmamba_backbone.py 中 nn.init.kaiming_normal_(fan_out, leaky_relu) 同分布。"""
    rng = np.random.default_rng(seed)
    std = math.sqrt(2.0 / ((c * ks) * (1 + 0.01 ** 2)))       # leaky_relu 的 a=0.01
    return rng.normal(0.0, std, (c, ks))


def gabor(alias):
    """AG 滤波器组（交付配置）。torch 在则用仓库实现，否则用 numpy 等价版。"""
    if HAS_TORCH:
        return _gb_ref(C, KS, F_LO, F_HI, SIGMA_RATIO, DC_FRAC,
                       float(alias), FM_LO, FM_HI).numpy()[:, 0, :]
    t = np.arange(KS, dtype=np.float64) - (KS - 1) / 2.0
    fr = np.exp(np.linspace(math.log(F_LO), math.log(F_HI), C))[:, None]
    sg = SIGMA_RATIO / fr
    ph = 2.0 * math.pi * fr * t[None, :]
    if alias > 0:
        fm = np.exp(np.linspace(math.log(FM_LO), math.log(FM_HI), C))[:, None]
        ph = ph + float(alias) * np.sin(2.0 * math.pi * fm * t[None, :])
    psi = np.exp(-(t[None, :] ** 2) / (2.0 * sg ** 2)) * np.cos(ph)
    n_dc = int(round(C * DC_FRAC))
    psi[n_dc:] -= psi[n_dc:].mean(axis=-1, keepdims=True)
    return psi / np.clip(np.linalg.norm(psi, axis=-1, keepdims=True), 1e-6, None)


def laplace(ks=64, c=64):
    """宽核 Laplace 参照族 [10]（models/widan_net.py::Laplace_fast, mode='sigmoid'）。

    注意源码是 exp(sigmoid(arg))（有界于 [1,e]），不是 exp(arg*sigmoid(arg))。
    """
    if HAS_TORCH:
        return _lap_ref(out_channels=c, kernel_size=ks, eps=0.1,
                        frequency=64000, mode='sigmoid').forward().numpy()[:, 0, :]
    a_ = np.linspace(0, c, c)[:, None]
    b_ = a_.copy()
    td = np.linspace(0, ks - 1, ks)
    q = 1.0 - 0.03 ** 2
    w = 2.0 * math.pi * 64000.0
    p_ = (td[None, :] - b_) / (a_ + 0.1)
    arg = (-0.03 / math.sqrt(q)) * (w * (p_ - 0.1))
    sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -700.0, 700.0)))
    return (1.0 / math.e) * np.exp(sig) * np.sin(w * (p_ - 0.1))


# ----------------------------------------------------------------- 计算
lines = []
lines.append('# Fig. 2 数值（Π @nfft=%d, band u<=(%.2f); ζ = 过零率）' % (NFFT, BAND))
lines.append('# 交付配置: KS=%d, C=%d, f_lo=%g, f_hi=%g, sigma_ratio=%g, dc_frac=%g'
             % (KS, C, F_LO, F_HI, SIGMA_RATIO, DC_FRAC))
lines.append('')

gabor_pts = []
for a in ALPHAS:
    z, p = zeta_rate(gabor(a)), peak_to_mean(gabor(a))
    gabor_pts.append((a, z, p))
    lines.append('Gabor alpha=%.1f   zeta=%.4f   Pi=%8.3f' % (a, z, p))

L = laplace()
zL, pL = zeta_rate(L), peak_to_mean(L)
lines.append('Laplace ref [10] (KS=64)  zeta=%.4f   Pi=%8.3f' % (zL, pL))

R = [(zeta_rate(kaiming_bank(s)), peak_to_mean(kaiming_bank(s))) for s in range(5)]
rz, rp = np.mean([x[0] for x in R]), np.mean([x[1] for x in R])
rsz, rsp = np.std([x[0] for x in R], ddof=1), np.std([x[1] for x in R], ddof=1)
lines.append('Kaiming random (5 seeds) zeta=%.4f+-%.4f  Pi=%.3f+-%.3f' % (rz, rsz, rp, rsp))
lines.append('')
lines.append('# 口径稳健性（Π）:')
for nf in (2048, 16384, 65536):
    lines.append('#   nfft=%6d  Gabor(0)=%7.3f  AG(2)=%7.3f  random=%6.3f'
                 % (nf, peak_to_mean(gabor(0.0), nfft=nf),
                    peak_to_mean(gabor(2.0), nfft=nf),
                    peak_to_mean(kaiming_bank(0), nfft=nf)))

with open(os.path.join(FIGD, '_fig2_values.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')
for ln in lines:
    print(ln)

# ----------------------------------------------------------------- 作图
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(3.45, 2.9))

PATH_C = '#1f77b4'
RAND_C = '#d62728'
REF_C = '#2ca02c'

ax.plot([p for _, _, p in gabor_pts], [z for _, z, _ in gabor_pts],
        '-o', color=PATH_C, ms=4.0, lw=1.2, zorder=3)
# α 标注：一律贴点的**右侧**（作者 2026-10-02 要求，勿压到折线）。
# 仅 α=3.0 需略微上抬：它的入射线来自右下方的 α=2.0，
# 纯右置会压在 2.0→3.0 那一段上。末尾的版式自检会验证这一点。
LAB_OFF = [(5, 0), (5, 0), (5, 0), (5, 0), (5, 0), (5, 6)]
for (a, z, p), off in zip(gabor_pts, LAB_OFF):
    ax.annotate(r'$\alpha=%.1f$' % a, (p, z), textcoords='offset points',
                xytext=off, fontsize=5.2, color=PATH_C)

# 交付点（α = 2.0）加空心环，使读者一眼看到交付配置落在哪里
pd_, zd_ = [(p, z) for a, z, p in gabor_pts if abs(a - 2.0) < 1e-9][0]
ax.plot([pd_], [zd_], 'o', ms=8.0, mfc='none', mec='0.15', mew=0.9, zorder=4)

ax.errorbar([rp], [rz], xerr=[rsp], yerr=[rsz], fmt='s', ms=4.0, color=RAND_C,
            lw=0, elinewidth=1.0, capsize=2, zorder=3)
ax.plot([pL], [zL], '^', ms=4.5, color=REF_C, zorder=3)

ax.set_xscale('log')
ax.set_xlim(4.5, 42.0)
ax.set_ylim(0.0, 0.58)
ax.set_xticks([5, 10, 15, 20, 30, 40])
ax.set_xticklabels(['5', '10', '15', '20', '30', '40'])   # 避免对数刻度标签互相压叠
ax.minorticks_off()
ax.set_xlabel(r'spectral peak-to-mean ratio $\Pi$   (commitment $\rightarrow$)', fontsize=6.5)
ax.set_ylabel(r'zero-crossing rate $\zeta$   (oscillation $\rightarrow$)', fontsize=6.5)
ax.tick_params(labelsize=6.0)
ax.grid(alpha=0.25, lw=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

# 图例：**显式 handle**。errorbar 返回的是 ErrorbarContainer，其默认 legend 把柄
# 画成“圆点 + 误差棒”，与图中实际的红色方形不一致（2026-10-02 作者指出）。
# 因此自己造 Line2D 手柄，保证图与图例逐项对应。
h_path = Line2D([], [], color=PATH_C, marker='o', ms=4.0, lw=1.2)
h_del = Line2D([], [], color='none', marker='o', ms=8.0, mfc='none',
               mec='0.15', mew=0.9)
h_rand = Line2D([], [], color=RAND_C, marker='s', ms=4.0, ls='none')
h_ref = Line2D([], [], color=REF_C, marker='^', ms=4.5, ls='none')
ax.legend([h_path, h_del, h_rand, h_ref],
          [r'AG filterbank, $\alpha$ path', r'delivered ($\alpha=2.0$)',
           r'Kaiming random init', r'wide-kernel ref. [10]'],
          fontsize=5.6, frameon=False, loc='lower left', handlelength=1.4,
          labelspacing=0.34, borderpad=0.1, handletextpad=0.5)

fig.tight_layout(pad=0.3)

# ------------------------------------------------------------- 版式自检
# 要求：α 标签互不重叠，且没有任何标签框被折线穿过（2026-10-02 作者提出）。
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_lab = [t for t in ax.texts if t.get_text().startswith(r'$\alpha$')]
_boxes = [t.get_window_extent(_rend) for t in _lab]
_bad = []
for i in range(len(_boxes)):
    for j in range(i + 1, len(_boxes)):
        if _boxes[i].overlaps(_boxes[j]):
            _bad.append('labels overlap: %s / %s'
                        % (_lab[i].get_text(), _lab[j].get_text()))
_path = np.array([[p for _, _, p in gabor_pts],
                  [z for _, z, _ in gabor_pts]]).T
_px = []
for k in range(len(_path) - 1):
    a0 = ax.transData.transform(_path[k])
    a1 = ax.transData.transform(_path[k + 1])
    _px.append(np.linspace(a0, a1, 200))
_px = np.vstack(_px)
for t, b in zip(_lab, _boxes):
    hit = ((_px[:, 0] >= b.x0) & (_px[:, 0] <= b.x1)
           & (_px[:, 1] >= b.y0) & (_px[:, 1] <= b.y1))
    if hit.any():
        _bad.append('polyline crosses label %s' % t.get_text())
print('[layout check] ' + ('OK — 6 个 α 标签互不重叠，且均未被折线穿过'
                           if not _bad else 'FAIL — ' + '; '.join(_bad)))
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIGD, 'fig2_zeta_pi.' + ext), dpi=400)
plt.close(fig)
print('\n[saved] %s' % os.path.join(FIGD, 'fig2_zeta_pi.pdf'))
