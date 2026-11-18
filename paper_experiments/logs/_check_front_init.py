#!/usr/bin/python
# -*- coding:utf-8 -*-
"""核验 `front_init` 四个取值是否**真正**生效于模型内部（而非只写在配置里）。

背景：`LiquidMambaDGNet` 曾把 `front_init` 白名单限定为 ('gabor','random','mix')，
于是配置写 'laplace' 时模型会**静默回退成 'gabor'**。本脚本从**模型自身**读
`front.init_mode` 与首层权重，直接与两个解析核组比对。

用法：python logs/_check_front_init.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

from models.agmamba_backbone import LiquidMambaDGNet, gabor_bank, laplace_bank

KW = dict(num_classes=14, d_model=128, d_inner=128, n_blocks=3, n_state=8,
          stride=4, front_chan=512, kernel_size=128, front_mode='gaborwide',
          gabor_f_lo=0.0025, gabor_f_hi=0.45, gabor_sigma=0.5,
          gabor_dc_frac=0.3, gabor_alias=2.0, tau_mode='zero', use_ssm=True)

G = gabor_bank(64, 128, 0.0025, 0.45, 0.5, 0.3, 2.0, 0.20, 0.48).squeeze(1).unsqueeze(1)
L = laplace_bank(64, 128)


def main():
    print('%-10s %-10s %-14s %-12s %-12s' % ('cfg init', 'model init', 'requires_grad',
                                             '|w-gabor|', '|w-laplace|'))
    for init in ('gabor', 'laplace', 'random', 'mix'):
        m = LiquidMambaDGNet(front_init=init, **KW)
        w = m.front.conv1.weight
        print('%-10s %-10s %-14s %-12.4g %-12.4g' % (
            init, m.front.init_mode, str(w.requires_grad),
            float((w - G).abs().max()), float((w - L).abs().max())))
    # 冻结检查
    m = LiquidMambaDGNet(front_init='gabor', freeze_init=True, **KW)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print('\nfreeze_init=True -> conv1.requires_grad = %s | trainable %d / total %d'
          % (m.front.conv1.weight.requires_grad, n_train, n_all))
    m2 = LiquidMambaDGNet(front_init='gabor', freeze_init=False, **KW)
    print('freeze_init=False -> conv1.requires_grad = %s | trainable %d / total %d'
          % (m2.front.conv1.weight.requires_grad,
             sum(p.numel() for p in m2.parameters() if p.requires_grad),
             sum(p.numel() for p in m2.parameters())))


if __name__ == '__main__':
    main()
