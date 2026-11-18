#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现性审计（P-统一协议的前置检查）。

回答两个问题：
  1. **主干层面**：同一输入/权重下 forward+backward 两遍，梯度是否逐位相同？
     并对比官方 mamba CUDA 扫描内核 vs 纯 PyTorch 分块扫描。
  2. **速度代价**：两种扫描实现在交付配置下的单步耗时与单单元（1500 步）估计。

用法
----
    python tools/check_determinism.py            # 跑全部检查
    python tools/check_determinism.py --timing   # 只跑速度

背景（2026-09-28 实测结论）
--------------------------
`mamba_ssm` 选择性扫描的**反传含原子累加**，梯度不逐位可复现
（max|Δgrad| ≈ 4e-8）。经 1500 步训练放大后，同一 (method, task, seed)
的 last10 可漂移 ~1.3 点。纯 PyTorch 分块扫描完全确定，代价约 5.4×。

→ 结论：公平对比应设 `AGM_FORCE_PY_SCAN=1`；且该代价**只影响 AG-Mamba**
  （基线是普通 CNN，不走 SSM 扫描）。
"""
import argparse
import os
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models import agmamba_backbone as B            # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def build():
    """交付配置主干（gaborwide 512ch, 3 blocks, d_model 128, tau_mode=zero）。"""
    torch.manual_seed(0)
    return B.LiquidMambaDGNet(
        num_classes=14, front_mode='gaborwide', front_init='gabor',
        front_chan=512, kernel_size=128, gabor_f_lo=0.0025,
        gabor_dc_frac=0.3, gabor_alias=2.0, use_ssm=True,
        tau_mode='zero', n_blocks=3, d_model=128, d_inner=128,
        n_state=8).to(DEV)


def _one_step(m, x, y, backward=True):
    lg, _, dn, _ = m(x)
    loss = F.cross_entropy(lg, y) + dn
    if not backward:
        return float(loss)
    m.zero_grad()
    loss.backward()
    g = torch.cat([p.grad.detach().flatten()
                   for p in m.parameters() if p.grad is not None])
    return float(loss), g


def check_grad(label, n=2):
    torch.manual_seed(1)
    x = torch.randn(64, 1, 1024, device=DEV)
    y = torch.randint(0, 14, (64,), device=DEV)
    res = []
    for _ in range(n):
        res.append(_one_step(build(), x, y))
    same = all(torch.equal(res[0][1], r[1]) for r in res[1:])
    md = max(float((res[0][1] - r[1]).abs().max()) for r in res[1:])
    print('  %-16s loss=%.6f  grad_identical=%-5s  max|Δgrad|=%.3e'
          % (label, res[0][0], same, md))
    return same


def timing(label, n=20):
    m = build()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    torch.manual_seed(1)
    x = torch.randn(64, 1, 1024, device=DEV)
    y = torch.randint(0, 14, (64,), device=DEV)

    def step():
        lg, _, dn, _ = m(x)
        loss = F.cross_entropy(lg, y) + dn
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    if DEV == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        step()
    if DEV == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print('  %-16s %7.1f ms/step   1500 步 ≈ %5.1f min/单元'
          % (label, dt * 1000, dt * 1500 / 60))
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', action='store_true', help='只跑速度测量')
    a = ap.parse_args()
    print('device = %s' % DEV)
    saved = B._OFFICIAL_SCAN
    print('官方 CUDA 扫描内核可用:', saved is not None)

    if not a.timing:
        print('\n[1/2] 梯度可复现性（同输入同权重跑两遍）')
        if saved is not None:
            check_grad('official-CUDA')
        else:
            print('  official-CUDA    不可用')
        B._OFFICIAL_SCAN = None
        check_grad('python-scan')

    print('\n[2/2] 速度')
    B._OFFICIAL_SCAN = saved
    t_off = timing('official-CUDA') if saved is not None else None
    B._OFFICIAL_SCAN = None
    t_py = timing('python-scan')
    if t_off:
        print('  → 纯 PyTorch 扫描慢 %.1fx（只影响 AG-Mamba 的单元）'
              % (t_py / t_off))


if __name__ == '__main__':
    main()
