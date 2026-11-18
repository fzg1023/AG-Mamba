#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.5 E1/E2：膨胀闭包 eta(s) 与合成膨胀实验（全部在冻结的核组上、训练前计算）。

严格实现手稿式 (14)：
    r_k(x) = || psi_k * x ||_2                     （逐通道输出能量，核先 L2 归一化）
    eta(s) = E_x[ < r(D_s x), S_{ln s} r(x) > ] / E_x[ ||r(D_s x)|| ||r(x)|| ]
其中 D_s 为频率膨胀 s 倍（y(t) = x(t/s)），S 为通道索引平移 ln s / ln r（线性插值），
r = 相邻中心频率比（对数网格间距）。

核组：
  · 本方法 Gabor 组（探针配置 K=64、K_s=64、f_lo=0.005、f_hi=0.45、sigma_ratio=0.6），
    alpha ∈ {0, 0.5, 1, 1.5, 2, 3}（dc_frac=0.3，与 α 扫描一致）
  · 随机初始化（kaiming，同形状）
  · 参照设计 [10] 的 Laplace 小波首层（WIDAN 发布实现，frequency=64000；另给 25600 版本）
所有核逐通道 L2 归一化后再算指标（只比较分析剖面的形状，不比较通道增益）。

探测集：谐波梳——故障基频 f_c = kappa * f_r（kappa ∈ {3,5,8}），f_r 取 HUST 十个转速，
12 次谐波（幅度 1/k），随机相位；信号长 8192，按流水线做零均值/单位方差标准化。

输出：docs/manuscript/_s65_data.md + figures/e1e2.npz（供画图）
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
sys.path.insert(0, ROOT)

K = 64            # 首层通道数
KS = 64           # 核长
F_LO, F_HI = 0.005, 0.45
SIGMA_RATIO = 0.6
FS = 25600.0      # HUST 采样率
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]     # 转速工况
KAPPAS = [3, 5, 8]                                     # 故障频率乘子（典型轴承值）
NHARM = 12
L = 8192
RATIO = (F_HI / F_LO) ** (1.0 / (K - 1))               # 相邻中心比 = 1.0739
S_LIST = [1.0 + 0.02 * i for i in range(21)]           # 1.00 .. 1.40 步长 0.02
S_KEY = 80.0 / 75.0                                    # 1.0667：T9 相对最近源域的膨胀
rng = np.random.RandomState(0)


# ----------------------------------------------------------------- 核组构造
def gabor_bank(alias, dc_frac=0.3, fm_lo=0.20, fm_hi=0.48):
    """复刻 models/agmamba_backbone.gabor_bank 的探针配置。"""
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
    """复刻 models/widan_net.Laplace_fast(eps=0.1, mode='sigmoid').forward()。"""
    eps = 0.1
    time_disc = torch.linspace(0, 63, steps=64)
    idx = torch.linspace(0, K, K).view(-1, 1)
    w = 2 * math.pi * frequency
    q = torch.tensor(1 - 0.03 ** 2)
    p = (time_disc.unsqueeze(0) - idx) / (idx + eps)
    psi = (1 / math.e) * torch.exp(((-0.03 / torch.sqrt(q)) * (w * (p - 0.1))).sigmoid()) \
        * torch.sin(w * (p - 0.1))
    return psi


def random_bank():
    g = torch.Generator().manual_seed(0)
    psi = torch.empty(K, KS)
    for k in range(K):
        nn_ = torch.empty(1, 1, KS)
        nn_.normal_(generator=g)
        psi[k] = nn_.flatten()
    return psi


BANKS = {}
for a in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    BANKS['alpha={}'.format(a)] = gabor_bank(a)
BANKS['random'] = random_bank()
BANKS['ref_Laplace_64k'] = laplace_bank(64000.0)
BANKS['ref_Laplace_25k6'] = laplace_bank(FS)

# 逐通道 L2 归一化（统一口径）
for k in BANKS:
    BANKS[k] = BANKS[k] / BANKS[k].norm(dim=-1, keepdim=True).clamp_min(1e-6)


# --------------------------------------------------------------- 探测信号
def make_probes():
    xs = []
    for fr in SPEEDS:
        for kap in KAPPAS:
            fc = kap * fr / FS                      # 归一化故障基频
            for _ in range(8):
                ph = rng.uniform(0, 2 * np.pi, NHARM)
                t = np.arange(L) / FS
                x = np.zeros(L)
                for h in range(1, NHARM + 1):
                    x += np.cos(2 * np.pi * fc * FS * h * t + ph[h - 1]) / h
                x = (x - x.mean()) / (x.std() + 1e-12)
                xs.append(x.astype(np.float32))
    return np.stack(xs)


def dilate(x, s):
    """频率膨胀 s 倍：y[n] = x(n/s)，线性插值，然后按流水线再标准化。"""
    n = np.arange(L, dtype=np.float64) / s
    y = np.interp(n, np.arange(L, dtype=np.float64), x.astype(np.float64))
    y = (y - y.mean()) / (y.std() + 1e-12)
    return y.astype(np.float32)


def responses(psi, xs, chunk=40):
    """逐通道输出能量 r(x) ∈ R^{n x K}。psi: (K,KS)"""
    w = psi.unsqueeze(1)                     # (K,1,KS)
    out = []
    for i in range(0, len(xs), chunk):
        xb = torch.from_numpy(xs[i:i + chunk]).unsqueeze(1)   # (B,1,L)
        y = F.conv1d(xb, w, stride=2, padding=KS // 2)        # (B,K,L')
        edge = KS // 2
        y = y[..., edge:-edge]
        out.append(y.flatten(2).norm(dim=2).numpy())          # (B,K)
    return np.concatenate(out, 0)


def shift(r, dfrac):
    """通道索引平移 dfrac（线性插值，环形外推用端点值）。"""
    idx = np.arange(K, dtype=np.float64)
    src = idx - dfrac
    lo = np.floor(src).astype(int)
    w = src - lo
    lo_c = np.clip(lo, 0, K - 1)
    hi_c = np.clip(lo + 1, 0, K - 1)
    return (1 - w) * r[:, lo_c] + w * r[:, hi_c]


def eta(psi, xs, s):
    r1 = responses(psi, xs)
    r2 = responses(psi, np.stack([dilate(x, s) for x in xs]))
    d = math.log(s) / math.log(RATIO)
    rs = shift(r1, d)
    num = np.sum(r2 * rs, 1)
    den = np.linalg.norm(r2, axis=1) * np.linalg.norm(rs, axis=1) + 1e-12
    return float(np.mean(num / den)), float(np.std(num / den))


def unshifted(psi, xs, s):
    """不做通道平移的相似度：1 - 它是 E2 的“特征漂移” D(s)。"""
    r1 = responses(psi, xs)
    r2 = responses(psi, np.stack([dilate(x, s) for x in xs]))
    num = np.sum(r2 * r1, 1)
    den = np.linalg.norm(r2, axis=1) * np.linalg.norm(r1, axis=1) + 1e-12
    return float(np.mean(num / den))


xs = make_probes()
print('探测信号 {} 条'.format(len(xs)))

L_OUT = []
P = L_OUT.append
P('# §6.5 E1/E2 数值底稿（由 `exp_e1_closure.py` 在冻结核组上计算）')
P('')
P('式 (14) 口径：r_k = ||psi_k * x||_2（核逐通道 L2 归一化）；eta(s) = 平均余弦相似度；'
  '通道平移量 = ln s / ln r，r = {:.4f}。探测集：{} 条谐波梳（kappa ∈ {}, '
  '转速 {} Hz，{} 次谐波，随机相位）。'.format(RATIO, len(xs), KAPPAS, SPEEDS, NHARM))
P('')

P('## E1. 膨胀闭包 eta(1.067)（T9 相对最近源域 75→80 Hz 的膨胀）')
P('')
P('| 核组 | eta(1.067) | eta(1.20) | eta(1.40) | 无平移相似度@1.067 |')
P('|---|---|---|---|---|')
eta_key = {}
for nm in BANKS:
    e1, _ = eta(BANKS[nm], xs, S_KEY)
    e12, _ = eta(BANKS[nm], xs, 1.20)
    e14, _ = eta(BANKS[nm], xs, 1.40)
    u1 = unshifted(BANKS[nm], xs, S_KEY)
    eta_key[nm] = e1
    P('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(nm, e1, e12, e14, u1))
P('')

P('## E2. 曲线：eta(s) 与无平移相似度（s = 1.00…1.40）')
P('')
P('| s | ' + ' | '.join(BANKS.keys()) + ' |')
P('|' + '---|' * (len(BANKS) + 1))
curve = {nm: [] for nm in BANKS}
for s in S_LIST:
    row = []
    for nm in BANKS:
        e, _ = eta(BANKS[nm], xs, s)
        curve[nm].append(e)
        row.append('{:.4f}'.format(e))
    P('| {:.2f} | {} |'.format(s, ' | '.join(row)))
P('')

P('## E3. 与实测极端转速精度对照（α 扫描 6 点，T9 best-epoch，§6.3 表 6）')
P('')
T9 = {'alpha=0.0': 63.33, 'alpha=0.5': 68.58, 'alpha=1.0': 63.07,
      'alpha=1.5': 73.60, 'alpha=2.0': 79.56, 'alpha=3.0': 82.09}
P('| α | eta(1.067) | T9 best-epoch |')
P('|---|---|---|')
xs_ = []
ys_ = []
for a in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    nm = 'alpha={}'.format(a)
    P('| {} | {:.4f} | {:.2f} |'.format(a, eta_key[nm], T9[nm]))
    xs_.append(eta_key[nm])
    ys_.append(T9[nm])
from scipy import stats as st
r_, p_ = st.pearsonr(xs_, ys_)
rho, prho = st.spearmanr(xs_, ys_)
P('')
P('  Pearson r = {:.3f} (p = {:.3f})；Spearman rho = {:.3f} (p = {:.3f})'.format(
    r_, p_, rho, prho))
P('')

with open(os.path.join(ROOT, 'docs/manuscript', '_s65_data.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L_OUT) + '\n')
np.savez(os.path.join(ROOT, 'docs', 'manuscript', 'figures', 'e1e2.npz'),
         s=np.array(S_LIST), eta_key=eta_key, T9=T9,
         **{'curve_' + nm: np.array(curve[nm]) for nm in BANKS})
print('\n'.join(L_OUT))
