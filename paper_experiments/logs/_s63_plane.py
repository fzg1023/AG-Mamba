"""§6.3 用：把前端消融各臂的首层核组放到 (zeta, Pi) 平面上。

复用 analysis/make_fig2_design_space.py 的度量与交付配置常数，
保证与 §3.3 / Fig. 2 的数值同源（同一个 gabor_bank 实现、同一 nfft/band）。
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))

import make_fig2_design_space as F  # noqa: E402

print('# zeta/Pi per arm  (nfft=%d, band u<=%.2f, torch=%s)'
      % (F.NFFT, F.BAND, F.HAS_TORCH))

# 1) 交付配置 alpha = 2.0
g2 = F.gabor(2.0)
print('delivered  alpha=2.0        KS=128 C=64   zeta=%.4f  Pi=%8.3f'
      % (F.zeta_rate(g2), F.peak_to_mean(g2)))

# 2) 解析端点 alpha = 0
g0 = F.gabor(0.0)
print('analytic   alpha=0.0        KS=128 C=64   zeta=%.4f  Pi=%8.3f'
      % (F.zeta_rate(g0), F.peak_to_mean(g0)))

# 3) alpha = 3.0（路径另一端）
g3 = F.gabor(3.0)
print('path end   alpha=3.0        KS=128 C=64   zeta=%.4f  Pi=%8.3f'
      % (F.zeta_rate(g3), F.peak_to_mean(g3)))

# 4) fr_narrow: f_lo 0.005 + KS 64（低频扩展反向消融）
gn = F.gabor_bank_like = None
if F.HAS_TORCH:
    gn = F._gb_ref(64, 64, 0.005, F.F_HI, F.SIGMA_RATIO, F.DC_FRAC,
                   2.0, F.FM_LO, F.FM_HI).numpy()[:, 0, :]
else:
    t = np.arange(64, dtype=np.float64) - (64 - 1) / 2.0
    fr = np.exp(np.linspace(math.log(0.005), math.log(F.F_HI), 64))[:, None]
    sg = F.SIGMA_RATIO / fr
    ph = 2.0 * math.pi * fr * t[None, :]
    fm = np.exp(np.linspace(math.log(F.FM_LO), math.log(F.FM_HI), 64))[:, None]
    ph = ph + 2.0 * np.sin(2.0 * math.pi * fm * t[None, :])
    psi = np.exp(-(t[None, :] ** 2) / (2.0 * sg ** 2)) * np.cos(ph)
    n_dc = int(round(64 * F.DC_FRAC))
    psi[n_dc:] -= psi[n_dc:].mean(axis=-1, keepdims=True)
    gn = psi / np.clip(np.linalg.norm(psi, axis=-1, keepdims=True), 1e-6, None)
print('fr_narrow  f_lo=0.005       KS=64  C=64    zeta=%.4f  Pi=%8.3f'
      % (F.zeta_rate(gn), F.peak_to_mean(gn)))

# 5) fr_mix: 32 Gabor(alpha=2) + 32 Kaiming(KS=128)
mix = np.concatenate([g2[:32], F.kaiming_bank(0, c=32, ks=128)], axis=0)
print('fr_mix     50/50            KS=128 C=64   zeta=%.4f  Pi=%8.3f'
      % (F.zeta_rate(mix), F.peak_to_mean(mix)))

# 6) fr_rand: 5 种子 Kaiming（与 §3.3 同分布）
zs, ps = [], []
for s in range(5):
    k = F.kaiming_bank(s)
    zs.append(F.zeta_rate(k))
    ps.append(F.peak_to_mean(k))
print('fr_rand    Kaiming (5 seeds) KS=128 C=64  zeta=%.4f+-%.4f  Pi=%.3f+-%.3f'
      % (np.mean(zs), np.std(zs), np.mean(ps), np.std(ps)))
