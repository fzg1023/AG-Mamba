#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.5 E3：用存下的权重按 §3.3 口径复算核统计（ζ = 64 点窗内过零数；χ 用补零 FFT）。

§3.3 报告的参照值：窄带解析 Gabor ζ=6.0 / χ=0.026；宽核 Laplace ζ=33.9 / χ=0.023；
随机 ζ≈32 / χ=0.005。本脚本对同一批核组复算，确认口径一致后再用于表 17 / Fig. 13。
"""
import math
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
FIGDIR = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
KS = 64
NFFT = 2048


def zeta_chi(p, nfft=NFFT):
    """p: (C,K)。ζ 返回“64 点窗内的过零次数”（与 §3.3 同尺度）；χ 用补零到 nfft 的 FFT。"""
    p = torch.as_tensor(p, dtype=torch.float32)
    if p.dim() == 3:
        p = p.squeeze(1)
    zc = ((p[:, :-1] * p[:, 1:]) < 0).float().sum(dim=1)      # 次数
    sp = torch.fft.rfft(p, n=nfft, dim=1).abs() ** 2
    chi = sp.max(dim=1).values / sp.sum(dim=1).clamp_min(1e-12)
    return float(zc.mean()), float(chi.mean()), float(zc.std()), float(chi.std())


def gabor_bank(alias, dc_frac=0.3, f_lo=0.005, f_hi=0.45, sigma_ratio=0.6):
    t = torch.arange(KS, dtype=torch.float32) - (KS - 1) / 2.0
    freqs = torch.exp(torch.linspace(math.log(f_lo), math.log(f_hi), 64)).view(-1, 1)
    sigma = sigma_ratio / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    phase = 2.0 * math.pi * freqs * t.unsqueeze(0)
    if alias > 0:
        fms = torch.exp(torch.linspace(math.log(0.20), math.log(0.48), 64)).view(-1, 1)
        phase = phase + float(alias) * torch.sin(2.0 * math.pi * fms * t.unsqueeze(0))
    psi = env * torch.cos(phase)
    n_dc = int(round(64 * dc_frac))
    psi[n_dc:] = psi[n_dc:] - psi[n_dc:].mean(dim=-1, keepdim=True)
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def narrowband_analytic():
    t = torch.arange(KS, dtype=torch.float32) - (KS - 1) / 2.0
    freqs = torch.exp(torch.linspace(math.log(0.01), math.log(0.08), 64)).view(-1, 1)
    sigma = 0.6 / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    psi = env * torch.cos(2.0 * math.pi * freqs * t.unsqueeze(0))
    psi = psi - psi.mean(dim=-1, keepdim=True)
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def laplace_bank(frequency=64000.0):
    eps = 0.1
    td = torch.linspace(0, 63, steps=64)
    idx = torch.linspace(0, 64, 64).view(-1, 1)
    w = 2 * math.pi * frequency
    q = torch.tensor(1 - 0.03 ** 2)
    p = (td.unsqueeze(0) - idx) / (idx + eps)
    psi = (1 / math.e) * torch.exp(((-0.03 / torch.sqrt(q)) * (w * (p - 0.1))).sigmoid()) \
        * torch.sin(w * (p - 0.1))
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def random_bank(seed=0):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(64, KS, generator=g)
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def eta_key(psi, s=80.0 / 75.0):
    """式 (14) 的膨胀闭包。与 exp_e1_closure.py 完全同一探测集（240 条谐波梳）。"""
    import torch.nn.functional as F
    SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
    KAPPAS = [3, 5, 8]
    NHARM, L, FS = 12, 8192, 25600.0
    rng = np.random.RandomState(0)
    xs = []
    for fr in SPEEDS:
        for kap in KAPPAS:
            fc = kap * fr
            for _ in range(8):
                ph = rng.uniform(0, 2 * np.pi, NHARM)
                t = np.arange(L) / FS
                x = np.zeros(L)
                for h in range(1, NHARM + 1):
                    x += np.cos(2 * np.pi * fc * h * t + ph[h - 1]) / h
                xs.append(((x - x.mean()) / (x.std() + 1e-12)).astype(np.float32))
    XS = np.stack(xs)
    p = torch.as_tensor(psi, dtype=torch.float32)
    if p.dim() == 3:
        p = p.squeeze(1)
    Kc = p.shape[0]
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w = p.unsqueeze(1)
    with torch.no_grad():
        y = F.conv1d(torch.from_numpy(XS).unsqueeze(1), w, stride=2,
                     padding=KS // 2)[..., KS // 2:-(KS // 2)]
    r1 = y.flatten(2).norm(dim=2).numpy()
    n = np.arange(L, dtype=np.float64) / s
    dil = np.stack([np.interp(n, np.arange(L), x.astype(np.float64)) for x in XS])
    dil = np.stack([(q - q.mean()) / (q.std() + 1e-12) for q in dil]).astype(np.float32)
    with torch.no_grad():
        y2 = F.conv1d(torch.from_numpy(dil).unsqueeze(1), w, stride=2,
                      padding=KS // 2)[..., KS // 2:-(KS // 2)]
    r2 = y2.flatten(2).norm(dim=2).numpy()
    d = math.log(s) / math.log(0.45 / 0.005) * (Kc - 1)
    idx = np.arange(Kc, dtype=np.float64)
    src = np.clip(idx - d, 0, Kc - 1.0001)
    lo = np.floor(src).astype(int)
    wq = src - lo
    rs = (1 - wq) * r1[:, lo] + wq * r1[:, np.clip(lo + 1, 0, Kc - 1)]
    return float(np.mean(np.sum(r2 * rs, 1)
                         / (np.linalg.norm(r2, axis=1) * np.linalg.norm(rs, axis=1) + 1e-12)))


L = []
P = L.append
P('# §6.5 E3 核统计（按 §3.3 口径复算：ζ = 64 点窗内过零数；χ 用 2048 点补零 FFT）')
P('')
P('## 口径核对（对照 §3.3 报告值）')
P('')
P('| 核组 | ζ（过零数/64 点） | χ | §3.3 报告 |')
P('|---|---|---|---|')
ref = [('窄带解析 Gabor（f∈[0.01,0.08]、去直流）', narrowband_analytic(), '6.0 / 0.026'),
       ('宽核 Laplace（参照设计，64000 Hz 参数）', laplace_bank(), '33.9 / 0.023'),
       ('随机初始化（Kaiming 同形状）', random_bank(), '≈32 / 0.005')]
for nm, psi, exp in ref:
    z, c, _, _ = zeta_chi(psi)
    P('| {} | {:.1f} | {:.4f} | {} |'.format(nm, z, c, exp))
P('')

W = np.load(os.path.join(ROOT, 'docs', 'manuscript', 'figures', 'e3_weights.npz'))
ACC = {'alpha=0|T0|0': 95.33, 'alpha=2|T0|0': 95.11,
       'alpha=0|T9|0': 70.89, 'alpha=2|T9|0': 84.44}   # 来自 _s65_e3.md（同一次运行）
print('npz keys:', sorted(W.files), flush=True)
P('## 训练前后（480 步探针预算，T0/T9 各 1 种子；权重已存盘复算）')
P('')
P('| 配置 | 任务 | ζ 初始化 → 训练后 | χ 初始化 → 训练后（×10³） | η(1.067) 初始化 → 训练后 | best tgt-acc |')
P('|---|---|---|---|---|---|')
STATS = {}
CELLS = ['alpha=0|T0|0', 'alpha=2|T0|0', 'alpha=0|T9|0', 'alpha=2|T9|0']
for key in CELLS:
    w0 = W[key + '|init']
    w1 = W[key + '|trained']
    z0, c0, zs0, cs0 = zeta_chi(w0)
    z1, c1, zs1, cs1 = zeta_chi(w1)
    e0 = eta_key(w0)
    e1 = eta_key(w1)
    acc = ACC.get(key, float('nan'))
    STATS[key] = dict(zeta0=z0, zeta1=z1, chi0=c0, chi1=c1,
                      eta0=e0, eta1=e1, acc=acc)
    nm, task, seed = key.split('|')
    P('| {} | {} | {:.1f} → {:.1f} | {:.1f} → {:.1f} | {:.4f} → {:.4f} | {:.2f} |'.format(
        nm, task, z0, z1, c0 * 1e3, c1 * 1e3, e0, e1, acc))
    print('{} {}: zeta {:.1f}->{:.1f}, chi(1e-3) {:.1f}->{:.1f}, eta {:.4f}->{:.4f}, acc {:.2f}'.format(
        nm, task, z0, z1, c0 * 1e3, c1 * 1e3, e0, e1, acc), flush=True)
P('')
REF = {}
for nm, psi in [('nb_analytic', narrowband_analytic()), ('laplace', laplace_bank()),
                ('random', random_bank())]:
    z, c, _, _ = zeta_chi(psi)
    REF[nm] = dict(zeta=z, chi=c, eta=eta_key(psi))
P('  参照核组（与表 15 同实现）：窄带解析 ζ={:.1f}/χ={:.4f}；宽核 Laplace ζ={:.1f}/χ={:.4f}/η={:.4f}；随机 ζ={:.1f}/χ={:.4f}/η={:.4f}'
  .format(REF['nb_analytic']['zeta'], REF['nb_analytic']['chi'],
          REF['laplace']['zeta'], REF['laplace']['chi'], REF['laplace']['eta'],
          REF['random']['zeta'], REF['random']['chi'], REF['random']['eta']))
np.savez(os.path.join(FIGDIR, 'e3_stats.npz'),
         cells=np.array(CELLS), refs=np.array(sorted(REF)),
         **{k + '|' + kk: vv for k, d in STATS.items() for kk, vv in d.items()},
         **{'ref:' + k + '|' + kk: vv for k, d in REF.items() for kk, vv in d.items()})
P('  交付核组（α=2, dc 0.3）在交付配置下的统计（供对照）：ζ = {:.1f}、χ = {:.4f}'
  .format(*zeta_chi(gabor_bank(2.0))[:2]))
P('  解析核组（α=0, dc 0.3）：ζ = {:.1f}、χ = {:.4f}'.format(
    *zeta_chi(gabor_bank(0.0))[:2]))
P('')

with open(os.path.join(ROOT, 'docs/manuscript', '_s65_e3_recalc.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
