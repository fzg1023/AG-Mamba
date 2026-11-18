#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.5 膨胀鲁棒性（受控合成）—— v0.3 清洁协议版。

**取代 v0.2 的 `exp_e1_closure.py` / `exp_e2_morph.py`**：旧版以退役指标 η(s)（式 14 的
"闭包度"）为核心，该指标与精度**反相关**（r = −0.795，见附录 C.4），已退役。
本脚本改用两个**无需训练、可审计**的量：

    r_k(x) = || psi_k * x ||_2                     逐通道响应能量（64 维）
    D(a, b) = 1 - cos( mean_x r(x)/||r(x)|| )      两组窗的平均响应之间的"响应漂移"

回答 §3.2 提出的两个竞争假说所需的**数据侧**证据：

    (i)  学习导致的锐化      → 需要"训练前后核组审计"（另由
                              `logs/run_s65_trained.sh` 的短训练审计给出）
    (ii) 形态学变化          → 冻结分析器即可复现：**真实变速** 与
                              **精确合成膨胀** 在同一条 s 阶梯上应给出**不同**的 D(s)

三组对照（全部用**交付配置的冻结核组**，从 `models/agmamba_backbone.gabor_bank` 取）：

  A. 真实数据：HUST T0（目标 20 Hz）vs 九个源域（25…80 Hz）→ D_phys(s)
  B. 合成膨胀：把 20 Hz 的同一批窗按 s 时间压缩（`dilate`）→ D_synth(s)
  C. 模型隔离：冲击序列模型下，(a) **纯膨胀**（重复率、衰减、谐振**全部**随 s 缩放）
     vs (b) **物理变速**（只缩放重复率，衰减 τ 与谐振 f_n 固定在物理单位）

  ⚠ B 与 C(a) 是"纯膨胀"；C(b) 才是真实转速变化。若真实变速的 D 远大于纯膨胀，
  则"变速 ≠ 膨胀"（支持假说 ii）。

输出：docs/manuscript/_s65_dilation.md（数值底稿）+ figures/_s65_dilation.npz
用法：python analysis/exp_s65_dilation.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from models.agmamba_backbone import gabor_bank          # noqa: E402

OUT_MD = os.path.join(ROOT, 'docs', 'manuscript', '_s65_dilation.md')
OUT_NPZ = os.path.join(ROOT, 'docs', 'manuscript', 'figures', '_s65_dilation.npz')

# --------------------------------------------------------------- 交付配置
C, KS = 64, 128                 # 首层 64 原子 × 核长 128（与 §4.2.1 一致）
F_LO, F_HI = 0.0025, 0.45
SIGMA, DC_FRAC, ALIAS = 0.5, 0.3, 2.0
FM_LO, FM_HI = 0.20, 0.48
L = 1024                        # 窗长
FS = 25600.0                    # HUST 采样率（Hz）
STRIDE, PAD = 2, KS // 2        # 与 GaborBankFrontend.conv1 一致

SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]


def bank():
    p = gabor_bank(C, KS, F_LO, F_HI, SIGMA, DC_FRAC, ALIAS, FM_LO, FM_HI)
    return p.squeeze(1)                       # (C, KS)


# --------------------------------------------------------------- 基本算子
def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def dilate(x, s):
    """时间压缩 s 倍（重复率 ×s）：y(t) = x(t/s)，离散上用线性插值实现。"""
    n = np.arange(L, dtype=np.float64) / s
    y = np.interp(n, np.arange(L, dtype=np.float64), x.astype(np.float64))
    return standardize(y)


def responses(psi, xs, chunk=64):
    """r_k(x) = || psi_k * x ||_2，stride 2、half-kernel padding（与源码一致）。"""
    w = psi.unsqueeze(1)
    out = []
    for i in range(0, len(xs), chunk):
        xb = torch.from_numpy(np.asarray(xs[i:i + chunk], dtype=np.float32)).unsqueeze(1)
        y = F.conv1d(xb, w, stride=STRIDE, padding=PAD)
        y = y[..., PAD:y.shape[-1] - PAD] if PAD > 0 else y
        out.append(y.flatten(2).norm(dim=2).numpy())
    return np.concatenate(out, 0)             # (N, C)


def mean_r(psi, xs):
    """窗平均的**单位化**响应向量（抗窗间能量波动）。"""
    r = responses(psi, xs)
    r = r / np.clip(np.linalg.norm(r, axis=1, keepdims=True), 1e-12, None)
    return r.mean(0), r


def shift_chan(r, d):
    """把响应剖面沿**通道轴**（对数频率）平移 d 个通道，端点做线性插值。"""
    idx = np.arange(len(r), dtype=np.float64)
    src = np.clip(idx - d, 0, len(r) - 1.0001)
    lo = np.floor(src).astype(int)
    w = src - lo
    return (1 - w) * r[lo] + w * r[np.clip(lo + 1, 0, len(r) - 1)]


def best_match(ra, rb, dmax=28):
    """式 (5) 的协变性检验：M = max_d cos(ra, S_d rb)，返回 (M, d*)。"""
    a = ra / max(np.linalg.norm(ra), 1e-12)
    b = rb / max(np.linalg.norm(rb), 1e-12)
    best = (-1.0, 0)
    for d in range(-dmax, dmax + 1):
        sd = shift_chan(b, d)
        sd = sd / max(np.linalg.norm(sd), 1e-12)      # ⚠ 平移后必须重新单位化，否则不是余弦
        c = float(np.dot(a, sd))
        if c > best[0]:
            best = (c, d)
    return best


def expected_shift(s, n=C, span=F_HI / F_LO):
    """对数均匀格里，s 倍膨胀对应的通道位移（正 = 向高频）。"""
    return n * math.log(s) / math.log(span)


def drift(ra, rb):
    """响应漂移 D = 1 - cos（0 = 完全一致，1 = 正交，2 = 反向）。"""
    a = ra / max(np.linalg.norm(ra), 1e-12)
    b = rb / max(np.linalg.norm(rb), 1e-12)
    return float(1.0 - np.dot(a, b))


# --------------------------------------------------------------- 合成数据
def synth(f_p_hz, phase=0.0, f_n=5000.0, tau=1e-3, length=L, fs=FS):
    """冲击序列 + **固定**结构谐振/衰减（物理转速变化：只改重复率）。"""
    t = np.arange(length) / fs
    x = np.zeros(length)
    p = 1.0 / f_p_hz
    i = 0
    while i * p < t[-1]:
        t0 = i * p
        idx = t >= t0
        dt = t[idx] - t0
        x[idx] += np.exp(-dt / tau) * np.sin(2 * np.pi * f_n * dt + phase) * (tau * fs)
        i += 1
    return standardize(x)


def main():
    psi = bank()
    LOG = []
    log = LOG.append

    log('# §6.5 膨胀鲁棒性（受控合成）—— v0.3 清洁协议数值底稿')
    log('')
    log('由 `analysis/exp_s65_dilation.py` 生成（**取代** v0.2 的 `exp_e1_closure.py` / `exp_e2_morph.py`）。')
    log('- 分析器：**冻结的交付配置核组**（64 原子 × 核长 128，$f_{\\rm lo}$=0.0025、$f_{\\rm hi}$=0.45、'
        'σ-ratio 0.5、dc 0.3、α=2.0），直接取自 `models/agmamba_backbone.gabor_bank`')
    log('- 响应量：$r_k(x)=\\|\\psi_k*x\\|_2$（stride 2 + half-pad，与源码一致）；'
        '漂移量 $D(a,b)=1-\\cos(\\overline{r},\\overline{r})$，两组窗各自单位化后取均值')
    log('- **不再使用 η(s)**：该闭包指标与精度反相关（r = −0.795），附录 C.4 已退役')
    log('')

    # ---------------- A/B：真实数据（HUST T0 = 目标 20 Hz，源域 25…80 Hz）
    import run_cell as RC
    srcs, tar, ncls = RC.load_data('HUST', RC.TASKS['HUST']['T0'])   # 目标域 = 域 0（20 Hz）

    def all_windows(ds):
        xs = []
        for i in range(len(ds)):
            seq, _ = ds[i]
            xs.append(np.asarray(seq, dtype=np.float64).reshape(-1)[:L])
        return np.stack(xs)

    W = {'20': all_windows(tar)}
    for name, ds in zip(SPEEDS[1:], srcs):
        W[str(name)] = all_windows(ds)
    log('## (A/B) 数据规模')
    log('')
    log('| 转速 (Hz) | 窗数 |')
    log('|---|---|')
    for k, v in W.items():
        log('| %s | %d |' % (k, len(v)))
    log('')
    log('（HUST 每域每类 50 训练 + 50 测试窗，9 类 → 目标域 450 窗；源域同规模。）')

    r20, _ = mean_r(psi, W['20'])
    # 自基线：把 20 Hz 的窗分两半，量一次"同一转速内的窗间漂移"
    h = len(W['20']) // 2
    ra, _ = mean_r(psi, W['20'][:h])
    rb, _ = mean_r(psi, W['20'][h:])
    d_self = drift(ra, rb)

    rows = []
    M = []
    chan = {}
    for k in SPEEDS[1:]:
        s = k / 20.0
        r_phys, _ = mean_r(psi, W[str(k)])
        r_syn, _ = mean_r(psi, dilate_batch(W['20'], s))
        d_phys = drift(r20, r_phys)
        d_syn = drift(r20, r_syn)
        m_phys, dp_ = best_match(r20, r_phys)
        m_syn, ds_ = best_match(r20, r_syn)
        M.append((s, m_phys, dp_, m_syn, ds_, expected_shift(s)))
        # 逐通道比值（物理 vs 合成），单位化后取 log2
        cp = np.log2(np.clip(r_phys, 1e-12, None) / np.clip(r20, 1e-12, None))
        cs = np.log2(np.clip(r_syn, 1e-12, None) / np.clip(r20, 1e-12, None))
        chan[str(k)] = (cp, cs)
        rows.append((k, s, d_phys, d_syn))
        log('')

    log('## (A/B) 响应漂移 $D$ 随转速比 $s$')
    log('')
    log('| s | 真实转速对 | 真实变速 $D_{\\rm phys}$ | 合成膨胀 $D_{\\rm synth}$ | 比值 |')
    log('|---|---|---|---|---|')
    for k, s, dp, ds_ in rows:
        log('| %.2f | 20 → %d Hz | %.4f | %.4f | %.2fx |' % (s, k, dp, ds_, dp / max(ds_, 1e-12)))
    log('')
    log('- **同一转速内的窗间自基线** $D_{\\rm self}$ = %.4f（20 Hz 的一半窗 vs 另一半）' % d_self)
    log('')
    log('## (A/B) 协变性检验 $M=\\max_d \\cos(\\bar r_{20}, S_d\\bar r_{s})$')
    log('')
    log('| s | 期望通道位移 | $M_{\\rm phys}$ | $d^*_{\\rm phys}$ | $M_{\\rm synth}$ | $d^*_{\\rm synth}$ |')
    log('|---|---|---|---|---|---|')
    for s, mp, dp_, ms, ds_, e in M:
        log('| %.2f | %+.1f | %.4f | %+d | %.4f | %+d |' % (s, e, mp, dp_, ms, ds_))
    log('')

    # ---------------- C：模型隔离（纯膨胀 vs 物理变速）
    KC = [3.05, 5.42]      # 冲击/轴转比（与 v0.2 同取值）
    log('## (C) 模型隔离：纯膨胀 vs 物理变速')
    log('')
    log('冲击序列模型：$f_p=\\kappa f_r$，结构衰减 $\\tau$ = 1 ms、谐振 $f_n$ = 5 kHz **固定在物理单位**；'
        '窗长 1024 点 @ 25.6 kHz（= 40 ms）。')
    log('')
    log('| κ | s | 纯膨胀 $D_{\\rm dil}$ | 物理变速 $D_{\\rm phys}$ | 比值 |')
    log('|---|---|---|---|---|')
    model_rows = []
    for kap in KC:
        x0 = synth(kap * 20.0)
        r0, _ = mean_r(psi, x0[None, :])
        for k in SPEEDS[1:]:
            s = k / 20.0
            d_dil = drift(r0, mean_r(psi, dilate_batch(x0[None, :], s))[0])
            d_ph = drift(r0, mean_r(psi, synth(kap * 20.0 * s)[None, :])[0])
            model_rows.append((kap, s, d_dil, d_ph))
            log('| %.2f | %.2f | %.4f | %.4f | %.2fx |' % (kap, s, d_dil, d_ph,
                                                           d_ph / max(d_dil, 1e-12)))
    log('')

    np.savez(OUT_NPZ,
             s=np.array([r[1] for r in rows]),
             d_phys=np.array([r[2] for r in rows]),
             d_synth=np.array([r[3] for r in rows]),
             d_self=np.array([d_self]),
             speeds=np.array(SPEEDS[1:]),
             chan_keys=np.array(list(chan)),
             chan_phys=np.stack([chan[k][0] for k in chan]),
             chan_synth=np.stack([chan[k][1] for k in chan]),
             mdl_kappa=np.array([r[0] for r in model_rows]),
             mdl_s=np.array([r[1] for r in model_rows]),
             mdl_dil=np.array([r[2] for r in model_rows]),
             mdl_phys=np.array([r[3] for r in model_rows]),
             m_phys=np.array([m[1] for m in M]),
             m_syn=np.array([m[3] for m in M]),
             m_shift_phys=np.array([m[2] for m in M]),
             m_shift_syn=np.array([m[4] for m in M]),
             m_expect=np.array([m[5] for m in M]))
    with open(OUT_MD, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    print('wrote %s and %s' % (OUT_MD, OUT_NPZ))
    for r in rows:
        print('  s=%.2f  D_phys=%.4f  D_synth=%.4f' % (r[1], r[2], r[3]))
    print('  D_self=%.4f' % d_self)


def dilate_batch(xs, s):
    return np.stack([dilate(x, s) for x in xs])


if __name__ == '__main__':
    main()
