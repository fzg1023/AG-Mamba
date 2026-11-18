#!/usr/bin/python
# -*- coding:utf-8 -*-
"""`fr_lap`（Laplace 首层）起飞前自检：

1. **随机数消耗中性检查** —— `fr_lap` 分支只做 `copy_` 写入（不像 `fr_rand`/`fr_mix`
   要额外调 `kaiming_normal_`），因此理论上它与交付配置的 **RNG 消耗量相同**，
   下游各层的随机权重应逐位一致。若成立，则该臂属于**严格配对**（与阈值法同口径），
   判定依据可从"2.55 点下限"升级为配对检验。
2. 打印 Laplace 核组的 (ζ, Π) 与有限性检查；并与参考点（KS=64: ζ=0.5382, Π=26.902）对照。
3. 真前向一次，确认输出形状与有限性。

用法：python logs/_check_fr_lap_pairing.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import numpy as np
import torch
import yaml

from utils.DictObj import DictObj
from models.agmamba_backbone import LiquidMambaDGNet, laplace_bank

CFG = os.path.join(_ROOT, 'agmamba', 'configs.yaml')
KW = dict(d_model=128, d_inner=128, n_blocks=3, n_state=8, stride=4,
          front_chan=512, kernel_size=128, wave_init=False, detach_tau=False,
          gamma_init=0.1, a_lo=0.5, a_hi=3.0, tau_min_lag=8, tau_max_lag=300,
          tau_temp=0.05, use_pos_embed=True, head_dropout=0.3,
          tau_estimator='autocorr', tau_mode='zero', tau_proj=False,
          ssm_mode='single', ms_d_inner=64, front_mode='gaborwide',
          front_norm='bn', gabor_sigma=0.5, gabor_f_lo=0.0025,
          gabor_f_hi=0.45, gabor_dc_frac=0.3, gabor_alias=2.0,
          freeze_init=False, film=False, use_ssm=True)


def build(init, seed=0):
    cfg = DictObj(yaml.load(open(CFG, encoding='utf-8'), Loader=yaml.FullLoader))
    n_cls = int(getattr(cfg, 'num_classes', 14))
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = LiquidMambaDGNet(num_classes=n_cls, front_init=init, **KW)
    return m


def main():
    print('=== 1. RNG 消耗中性：gabor vs laplace 除 conv1.weight 外的参数是否逐位一致 ===')
    a = build('gabor', seed=0)
    b = build('laplace', seed=0)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    diff = [(k, float((sa[k].float() - sb[k].float()).abs().max()))
            for k in sa if sa[k].dtype.is_floating_point]
    conv1 = [d for k, d in diff if k.endswith('conv1.weight')]
    other = [(k, d) for k, d in diff if not k.endswith('conv1.weight')]
    worst = max(other, key=lambda t: t[1])
    print('  n_float_tensors=%d | conv1.weight max|Δ|=%.6g (应为大值)'
          % (len(diff), conv1[0] if conv1 else float('nan')))
    print('  其余张量 max|Δ| = %.6g  (键: %s)' % (worst[1], worst[0]))
    print('  非零差异张量数 = %d' % sum(1 for _, d in other if d > 0))
    print('  => RNG 消耗%s' % ('中性：该臂属于**严格配对**' if worst[1] == 0 else
                               '**不同**：该臂只能按 (task,seed) 配对的弱口径判读'))

    print()
    print('=== 2. Laplace 核组统计 (ζ, Π) ===')
    import make_fig2_design_space as F
    for ks in (64, 128):
        bank = laplace_bank(64, ks).numpy()[:, 0, :]
        print('  KS=%3d  finite=%s  max|psi|=%.4f  ζ=%.4f  Π=%8.3f'
              % (ks, np.isfinite(bank).all(), np.abs(bank).max(),
                 F.zeta_rate(bank), F.peek_to_mean(bank) if hasattr(F, 'peek_to_mean')
                 else F.peak_to_mean(bank)))
    ref = laplace_bank(64, 64).numpy()[:, 0, :]
    print('  参考点（§3.3：KS=64  ζ=0.5382  Π=26.902）→ 复算 ζ=%.4f Π=%.3f'
          % (F.zeta_rate(ref), F.peak_to_mean(ref)))

    print()
    print('=== 3. 前向 ===')
    m = build('laplace', seed=0)
    with torch.no_grad():
        out = m(torch.randn(2, 1, 1024))
    if isinstance(out, (tuple, list)):
        print('  out[0].shape=%s  finite=%s' % (tuple(out[0].shape),
                                                bool(torch.isfinite(out[0]).all())))
    else:
        print('  out.shape=%s finite=%s' % (tuple(out.shape), bool(torch.isfinite(out).all())))
    print('  可训练参数量 = %d' % sum(p.numel() for p in m.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()
