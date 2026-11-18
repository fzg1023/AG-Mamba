#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.5 E3：训练是否“锐化”首层核（§3.2 假说 (i) 的直接检验）。

做法：用交付管线训练 480 步（探针预算），在**训练前/后**分别测量首层 64 个核的
式 (8) 统计（过零率 ζ、谱集中度 χ）与式 (14) 的膨胀闭包 η(1.067)——
后者把“训练后的核组”当成冻结分析器，用与 E1 相同的谐波梳探测集计算。

变体：α = 0（解析核组）与 α = 2（交付核组），任务 T0/T9。
输出：docs/manuscript/_s65_e3.md + docs/manuscript/figures/e3.npz
"""
import importlib
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
sys.path.insert(0, ROOT)

from agmamba import method as M                     # noqa: E402  触发 configs
import run_dgfdb as R                    # noqa: E402
import run_final_eval as FE              # noqa: E402

KS = 64
FS = 25600.0
RATIO = (0.45 / 0.005) ** (1.0 / 63.0)
S_KEY = 80.0 / 75.0
SPEEDS = [20, 25, 30, 40, 60, 70, 75, 80]
KAPPAS = [3.05, 5.42]
rng = np.random.RandomState(0)

VARIANTS = {
    'alpha=0': {'front_mode': 'gaborwide', 'front_init': 'gabor', 'front_chan': 512,
                'gabor_dc_frac': 0.3, 'gabor_alias': 0.0, 'use_ssm': True,
                'tau_mode': 'zero', 'steps_override': 480, 'use_scheduler': False,
                'gabor_f_lo': 0.005, 'front_ksize': 64},
    'alpha=2': {'front_mode': 'gaborwide', 'front_init': 'gabor', 'front_chan': 512,
                'gabor_dc_frac': 0.3, 'gabor_alias': 2.0, 'use_ssm': True,
                'tau_mode': 'zero', 'steps_override': 480, 'use_scheduler': False,
                'gabor_f_lo': 0.005, 'front_ksize': 64},
}
CELLS = [('alpha=0', 'T0', 0), ('alpha=2', 'T0', 0),
         ('alpha=0', 'T9', 0), ('alpha=2', 'T9', 0)]


# --------------------------------------------------------------- 核组统计
def zeta_chi(psi):
    """式 (8) 的 ζ（过零率）与 χ（谱集中度）。psi 可为 (C,K) 或 (C,1,K)。"""
    p = psi.detach().cpu().float()
    if p.dim() == 3:
        p = p.squeeze(1)
    zc = ((p[:, :-1] * p[:, 1:]) < 0).float().mean(dim=1)
    sp = torch.fft.rfft(p, dim=1).abs() ** 2
    chi = sp.max(dim=1).values / sp.sum(dim=1).clamp_min(1e-12)
    return float(zc.mean()), float(chi.mean()), float(zc.std()), float(chi.std())


def narrowband_reference():
    """§3.3 的窄带解析 Gabor 组（f∈[0.01,0.08]、去直流）——用于核对测量代码。"""
    t = torch.arange(KS, dtype=torch.float32) - (KS - 1) / 2.0
    freqs = torch.exp(torch.linspace(math.log(0.01), math.log(0.08), 64)).view(-1, 1)
    sigma = 0.6 / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    psi = env * torch.cos(2.0 * math.pi * freqs * t.unsqueeze(0))
    psi = psi - psi.mean(dim=-1, keepdim=True)
    return psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def make_probes():
    xs = []
    for fr in SPEEDS:
        for kap in KAPPAS:
            fc = kap * fr
            for _ in range(4):
                ph = rng.uniform(0, 2 * np.pi, 12)
                t = np.arange(8192) / FS
                x = np.zeros(8192)
                for h in range(1, 13):
                    x += np.cos(2 * np.pi * fc * h * t + ph[h - 1]) / h
                xs.append(((x - x.mean()) / (x.std() + 1e-12)).astype(np.float32))
    return np.stack(xs)


XS = make_probes()


def responses(psi, xs, chunk=64):
    w = psi.detach().cpu().float().unsqueeze(1)
    out = []
    for i in range(0, len(xs), chunk):
        xb = torch.from_numpy(xs[i:i + chunk]).unsqueeze(1)
        y = F.conv1d(xb, w, stride=2, padding=KS // 2)
        y = y[..., KS // 2:-(KS // 2)]
        out.append(y.flatten(2).norm(dim=2).numpy())
    return np.concatenate(out, 0)


def eta_key(psi):
    p = psi.detach().cpu().float()
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    K = p.shape[0]
    n = np.arange(8192, dtype=np.float64) / S_KEY
    dil = np.stack([np.interp(n, np.arange(8192), x.astype(np.float64))
                    for x in XS])
    dil = np.stack([(y - y.mean()) / (y.std() + 1e-12) for y in dil]).astype(np.float32)
    r1 = responses(p, XS)
    r2 = responses(p, dil)
    d = math.log(S_KEY) / math.log(RATIO)
    idx = np.arange(K, dtype=np.float64)
    src = np.clip(idx - d, 0, K - 1.0001)
    lo = np.floor(src).astype(int)
    w = src - lo
    rs = (1 - w) * r1[:, lo] + w * r1[:, np.clip(lo + 1, 0, K - 1)]
    num = np.sum(r2 * rs, 1)
    den = np.linalg.norm(r2, axis=1) * np.linalg.norm(rs, axis=1) + 1e-12
    return float(np.mean(num / den))


def train_cell(variant, task, seed):
    for k, v in FE.DEFAULTS.items():
        setattr(M.configs, k, v)
    for k, v in VARIANTS[variant].items():
        setattr(M.configs, k, v)
    R.setup_seed(seed)
    task_obj = R.TASKS['HUST'][task]
    srcs, tar, num_cls = R.load_data('HUST', task_obj)
    srcs = [R.ResampleDataset(s) for s in srcs]
    tar = R.ResampleDataset(tar)
    iters = max(1, (len(srcs[0]) + R.BATCH_SIZE - 1) // R.BATCH_SIZE)
    steps = R.EPOCHS * iters
    cfg = R.make_config(M, 'AGMamba', num_cls, steps, iters, 'cuda',
                        n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = M.AGMamba(cfg)
    logger = logging.getLogger('e3_{}_{}_{}'.format(variant, task, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    path = '/tmp/e3_{}_{}_seed{}.log'.format(variant.replace('=', ''), task, seed)
    logger.addHandler(logging.FileHandler(path, mode='w'))
    train_loaders = [R.InfiniteDataLoader(s, R.BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=R.BATCH_SIZE, shuffle=False,
                             num_workers=0)
    w0 = model.model.front.conv1.weight.detach().cpu().clone()
    res = model.train_model(zip(*train_loaders), [test_loader], logger)
    acc = float(np.array(res['acces'])[:, 0].max())
    w1 = model.model.front.conv1.weight.detach().cpu().clone()
    return w0, w1, acc


L = []
P = L.append
P('# §6.5 E3 数值底稿（训练前后首层核统计，480 步探针预算）')
P('')
P('ζ：核的过零率；χ：谱集中度（式 8）。η(1.067)：把该核组当作冻结分析器，'
  '在与 E1 相同的谐波梳探测集上按式 (14) 计算。')
P('')
P('| 变体 | 任务 | 种子 | ζ(初始化) | ζ(训练后) | χ(初始化) | χ(训练后) | best tgt-acc |')
P('|---|---|---|---|---|---|---|---|')

out = {}
W = {}
for variant, task, seed in CELLS:
    w0, w1, acc = train_cell(variant, task, seed)
    z0, c0, zs0, cs0 = zeta_chi(w0)
    z1, c1, zs1, cs1 = zeta_chi(w1)
    out[(variant, task, seed)] = (z0, z1, c0, c1, acc)
    W['{}|{}|{}|init'.format(variant, task, seed)] = w0.squeeze(1).numpy()
    W['{}|{}|{}|trained'.format(variant, task, seed)] = w1.squeeze(1).numpy()
    P('| {} | {} | {} | {:.1f} | {:.1f} | {:.4f} | {:.4f} | {:.2f} |'.format(
        variant, task, seed, z0, z1, c0, c1, acc * 100))
    print('{} {} seed{}: zeta {:.1f}->{:.1f} (sd {:.1f}->{:.1f}), '
          'chi {:.4f}->{:.4f} (sd {:.4f}->{:.4f}), acc {:.2f}'.format(
              variant, task, seed, z0, z1, zs0, zs1, c0, c1, cs0, cs1, acc * 100),
          flush=True)
P('')

zb = narrowband_reference()
zr, cr, _, _ = zeta_chi(zb)
nb = {}
for a in (0.0, 2.0):
    t = torch.arange(KS, dtype=torch.float32) - (KS - 1) / 2.0
    freqs = torch.exp(torch.linspace(math.log(0.005), math.log(0.45), 64)).view(-1, 1)
    sigma = 0.6 / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    phase = 2.0 * math.pi * freqs * t.unsqueeze(0)
    if a > 0:
        fms = torch.exp(torch.linspace(math.log(0.20), math.log(0.48), 64)).view(-1, 1)
        phase = phase + a * torch.sin(2.0 * math.pi * fms * t.unsqueeze(0))
    psi = env * torch.cos(phase)
    n_dc = int(round(64 * 0.1))
    psi[n_dc:] = psi[n_dc:] - psi[n_dc:].mean(dim=-1, keepdim=True)
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nb[a] = zeta_chi(psi)[:2]
P('## 核统计的参照核对（式 8）')
P('')
P('| 核组 | ζ | χ |')
P('|---|---|---|')
nb[0.0]
zz = narrowband_reference()
P('| 窄带解析 Gabor（f∈[0.01,0.08]、去直流；§3.3 报告 6.0 / 0.026） | {:.1f} | {:.4f} |'
  .format(*zeta_chi(zz)[:2]))
P('| 交付宽频段核组 α=2（dc 0.3，第 10% 以上通道去直流） | {:.1f} | {:.4f} |'.format(
    *nb[2.0]))
P('| 宽频段解析核组 α=0（dc 0.3） | {:.1f} | {:.4f} |'.format(*nb[0.0]))
P('')
with open(os.path.join(ROOT, 'docs', 'manuscript', 'figures', 'e3_weights.npz'), 'wb') as f:
    np.savez(f, **W)

with open(os.path.join(ROOT, 'docs/manuscript', '_s65_e3.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
np.savez(os.path.join(ROOT, 'docs', 'manuscript', 'figures', 'e3.npz'),
         keys=np.array(['|'.join(map(str, k)) for k in out]),
         vals=np.array([out[k] for k in out]))
print('\n'.join(L))
