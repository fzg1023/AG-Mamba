#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.5 E2：膨胀的非线性成分——固定结构谐振 + 冲击重叠。

物理合成（不依赖训练）：
    x(t) = sum_i  exp(-(t - i/f_p)/tau) * sin(2*pi*f_n*(t - i/f_p))   (t >= i/f_p)
其中 f_p = kappa * f_r 为冲击重复率（随转速成比例），而 **tau 与 f_n 是结构常数、不随转速变**。
这正是 §3.2 所说“谐振包络近似不变”的部分：转速变化时梳齿成比例移动，而包络不移动，
故实际变化**不是**纯膨胀。

对每个核组比较三件事（同一窗长 1024、按流水线做零均值/单位方差）：
    cos(x_slow, D_s x_slow)          纯膨胀理想化的响应相似度（应接近 1）
    cos(x_slow, x_fast_phys)         物理快转速信号的响应相似度
    max_d cos(S_d r_slow, r_phys)    用“通道平移模型”解释物理变化的最佳相似度
                                   （越高说明该核组的响应变化越接近一次纯膨胀）

输出：docs/manuscript/_s65_e2.md + docs/manuscript/figures/e2.npz
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
K, KS = 64, 64
F_LO, F_HI, SIGMA_RATIO = 0.005, 0.45, 0.6
FS = 25600.0
L = 1024                      # 与数据流水线一致的窗长（40 ms）
TAU = 0.001                   # 1 ms：结构衰減常数（不随转速变）
F_N = 5000.0                  # 结构谐振（不随转速变）
KAPPAS = [3.05, 5.42]
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
STEPS = [(25, 30), (40, 60), (75, 80)]        # 范围内 / 跨带 / 极端外推
RATIO = (F_HI / F_LO) ** (1.0 / (K - 1))
rng = np.random.RandomState(0)


def gabor_bank(alias, dc_frac=0.3, fm_lo=0.20, fm_hi=0.48):
    t = torch.arange(KS, dtype=torch.float32) - (KS - 1) / 2.0
    freqs = torch.exp(torch.linspace(math.log(F_LO), math.log(F_HI), K)).view(-1, 1)
    sigma = SIGMA_RATIO / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    phase = 2.0 * math.pi * freqs * t.unsqueeze(0)
    if alias > 0:
        fms = torch.exp(torch.linspace(math.log(fm_lo), math.log(fm_hi), K)).view(-1, 1)
        phase = phase + float(alias) * torch.sin(2.0 * math.pi * fms * t.unsqueeze(0))
    psi = env * torch.cos(phase)
    n_dc = int(round(K * dc_frac))
    psi[n_dc:] = psi[n_dc:] - psi[n_dc:].mean(dim=-1, keepdim=True)
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def laplace_bank(frequency):
    eps = 0.1
    td = torch.linspace(0, 63, steps=64)
    idx = torch.linspace(0, K, K).view(-1, 1)
    w = 2 * math.pi * frequency
    q = torch.tensor(1 - 0.03 ** 2)
    p = (td.unsqueeze(0) - idx) / (idx + eps)
    psi = (1 / math.e) * torch.exp(((-0.03 / torch.sqrt(q)) * (w * (p - 0.1))).sigmoid()) \
        * torch.sin(w * (p - 0.1))
    return psi


BANKS = {}
for a in (0.0, 1.0, 2.0, 3.0):
    BANKS['alpha={}'.format(a)] = gabor_bank(a)
BANKS['random'] = torch.stack([
    torch.randn(KS, generator=torch.Generator().manual_seed(k)) for k in range(K)])
BANKS['ref_Laplace'] = laplace_bank(64000.0)
for k in BANKS:
    BANKS[k] = BANKS[k] / BANKS[k].norm(dim=-1, keepdim=True).clamp_min(1e-6)


def synth(f_c_hz, phase, f_n=F_N, tau=TAU, L=L):
    """冲击序列 + 固定结构谐振响应，窗内截断（与真实窗一致的有限观测）。"""
    t = np.arange(L) / FS
    x = np.zeros(L)
    p = 1.0 / f_c_hz
    i = 0
    while i * p < t[-1]:
        t0 = i * p
        idx = t >= t0
        dt = t[idx] - t0
        x[idx] += np.exp(-dt / tau) * np.sin(2 * np.pi * f_n * dt + phase) * (tau * FS)
        i += 1
    return x


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def dilate(x, s):
    n = np.arange(L, dtype=np.float64) / s
    y = np.interp(n, np.arange(L, dtype=np.float64), x.astype(np.float64))
    return standardize(y)


def responses(psi, xs, chunk=64):
    w = psi.unsqueeze(1)
    out = []
    for i in range(0, len(xs), chunk):
        xb = torch.from_numpy(xs[i:i + chunk]).unsqueeze(1)
        y = F.conv1d(xb, w, stride=2, padding=KS // 2)
        y = y[..., KS // 2:-(KS // 2)]
        out.append(y.flatten(2).norm(dim=2).numpy())
    return np.concatenate(out, 0)


def shift(r, d):
    idx = np.arange(K, dtype=np.float64)
    src = np.clip(idx - d, 0, K - 1.0001)
    lo = np.floor(src).astype(int)
    w = src - lo
    return (1 - w) * r[:, lo] + w * r[:, np.clip(lo + 1, 0, K - 1)]


def cos(a, b):
    num = np.sum(a * b, 1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def build_probe_set():
    """每个 (kappa, f_target) 生成一条物理信号与随机相位。"""
    slow, fast = [], []
    for kap in KAPPAS:
        for fr in sorted(set([s for pr in STEPS for s in pr])):
            ph = rng.uniform(0, 2 * np.pi, 8)
            for p in ph:
                slow.append(standardize(synth(kap * fr, p)))
                fast.append(standardize(synth(kap * fr, p)))     # 占位，稍后按目标替换
    return np.array(slow), np.array(fast)


rows = []
L_OUT = []
P = L_OUT.append
P('# §6.5 E2 数值底稿（物理非膨胀成分：固定结构谐振 + 冲击重叠）')
P('')
P('模型：x(t) = Σ_i exp(−(t−i/f_p)/τ)·sin(2π f_n(t−i/f_p))，f_p = κ·f_r（随转速），'
  'τ = 1 ms 与 f_n = 5 kHz 为**结构常数**（不随转速）；窗长 1024（40 ms），逐窗标准化。')
P('')

for (fr_a, fr_b) in STEPS:
    s = fr_b / fr_a
    slow_xs, dil_xs, phys_xs = [], [], []
    for kap in KAPPAS:
        for p in rng.uniform(0, 2 * np.pi, 8):
            xs_ = standardize(synth(kap * fr_a, p))
            slow_xs.append(xs_)
            dil_xs.append(dilate(xs_, s))
            phys_xs.append(standardize(synth(kap * fr_b, p)))
    slow_xs = np.array(slow_xs, dtype=np.float32)
    dil_xs = np.array(dil_xs, dtype=np.float32)
    phys_xs = np.array(phys_xs, dtype=np.float32)
    d0 = math.log(s) / math.log(RATIO)
    shifts = np.arange(-6, 6.01, 0.25)
    P('### {} → {} Hz  (s = {:.3f}，通道平移 {:.2f})'.format(fr_a, fr_b, s, d0))
    P('')
    P('| 核组 | cos(纯膨胀) | cos(物理) | max_d cos(平移模型) | 最优平移 d* |')
    P('|---|---|---|---|---|')
    for nm, psi in BANKS.items():
        r_slow = responses(psi, slow_xs)
        r_dil = responses(psi, dil_xs)
        r_phys = responses(psi, phys_xs)
        c_dil = float(np.mean(cos(r_dil, shift(r_slow, d0))))
        c_phys = float(np.mean(cos(r_phys, r_slow)))
        best_c, best_d = -1.0, np.nan
        for d in shifts:
            c = float(np.mean(cos(r_phys, shift(r_slow, d))))
            if c > best_c:
                best_c, best_d = c, d
        rows.append((nm, fr_a, fr_b, c_dil, c_phys, best_c, best_d))
        P('| {} | {:.4f} | {:.4f} | {:.4f} | {:+.2f} |'.format(
            nm, c_dil, c_phys, best_c, best_d))
    P('')
    print('{}->{} 完成'.format(fr_a, fr_b), flush=True)

P('说明：`cos(纯膨胀)` 是把慢转速信号做频率膨胀 s 倍后与慢转速响应按通道平移对齐的相似度；'
  '`cos(物理)` 是真实快转速物理信号（冲击更密、谐振不移动）与慢转速响应的相似度；'
  '`max_d cos(平移模型)` 是允许任意通道平移时能达到的最佳相似度——越高表示该核组的响应变化'
  '越能被“一次纯膨胀”解释。')
P('')

with open(os.path.join(ROOT, 'docs/manuscript', '_s65_e2.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L_OUT) + '\n')
np.savez(os.path.join(ROOT, 'docs', 'manuscript', 'figures', 'e2.npz'),
         rows=np.array(rows, dtype=object))
print('\n'.join(L_OUT))
