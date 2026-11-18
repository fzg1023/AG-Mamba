#!/usr/bin/python
# -*- coding:utf-8 -*-
"""§6.3 前端设计消融（frontend_ablate_clean）**起飞前自检**。

对 5 个新臂 + 交付参照，逐一：
  1. 用交付配置（agmamba/configs.yaml）构造 LiquidMambaDGNet；
  2. 打印可训练参数量；
  3. 真前向一次 x=(2,1,1024)，打印前端图与最终输出形状。

不训练、不写 results/，只验证 5 个臂的配置都能建图、且时间长度仍为 128。
用法：python logs/_preflight_front_arms.py > logs/_preflight_front_arms.out 2>&1
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import yaml
import torch

from utils.DictObj import DictObj
from models.agmamba_backbone import LiquidMambaDGNet

CFG = os.path.join(_ROOT, 'agmamba', 'configs.yaml')

ARMS = [
    ('fr_ref(delivered)', {}),
    ('fr_rand', {'front_init': 'random'}),
    ('fr_mix', {'front_init': 'mix'}),
    ('fr_lap', {'front_init': 'laplace'}),
    ('fr_narrow', {'gabor_f_lo': 0.005, 'front_ksize': 64}),
    ('fr_cap768', {'front_chan': 768}),
    ('fr_cap960', {'front_chan': 960}),
]

KW_KEYS = [
    ('d_model', 128), ('d_inner', 128), ('n_blocks', 3), ('n_state', 8),
    ('front_stride', 4), ('front_chan', 128), ('front_ksize', 64),
    ('wave_init', False), ('detach_tau', False), ('gamma_init', 0.1),
    ('a_lo', 0.5), ('a_hi', 3.0), ('tau_min_lag', 8), ('tau_max_lag', 300),
    ('tau_temp', 0.05), ('tau_f_lo', 3), ('tau_f_hi', 120),
    ('use_pos_embed', True), ('head_dropout', 0.3),
    ('tau_estimator', 'autocorr'), ('tau_mode', 'normal'),
    ('tau_proj', False), ('ssm_mode', 'single'), ('ms_d_inner', 64),
    ('front_mode', 'cnn'), ('front_norm', 'bn'), ('front_init', 'none'),
    ('gabor_sigma', 0.5), ('gabor_f_lo', 0.005), ('gabor_f_hi', 0.45),
    ('gabor_dc_frac', 0.5), ('gabor_alias', 0.0), ('freeze_init', False),
    ('film', False), ('use_ssm', True),
]


def load_cfg():
    with open(CFG, encoding='utf-8') as f:
        return DictObj(yaml.load(f, Loader=yaml.FullLoader))


def build(over):
    c = load_cfg()
    for k, v in over.items():
        setattr(c, k, v)
    kw = {}
    for key, dflt in KW_KEYS:
        val = getattr(c, key, dflt)
        # 与 agmamba/method.py 的取值方式一致（int/bool/float/str 归一）
        if key in ('d_model', 'd_inner', 'n_blocks', 'n_state', 'stride',
                   'front_chan', 'kernel_size', 'tau_min_lag', 'tau_max_lag',
                   'tau_f_lo', 'tau_f_hi', 'ms_d_inner'):
            val = int(val)
        elif key in ('wave_init', 'detach_tau', 'use_pos_embed', 'tau_proj',
                     'freeze_init', 'film', 'use_ssm'):
            val = bool(val)
        elif key in ('gamma_init', 'a_lo', 'a_hi', 'tau_temp', 'head_dropout',
                     'gabor_sigma', 'gabor_f_lo', 'gabor_f_hi',
                     'gabor_dc_frac', 'gabor_alias'):
            val = float(val)
        elif key in ('tau_estimator', 'tau_mode', 'ssm_mode', 'front_mode',
                     'front_norm', 'front_init'):
            val = str(val)
        # 配置名 -> 构造函数形参名
        kw[{'front_ksize': 'kernel_size', 'front_stride': 'stride'}.get(
            key, key)] = val
    kw['num_classes'] = 14
    return LiquidMambaDGNet(**kw), c


def main():
    x = torch.randn(2, 1, 1024)
    print('%-18s %10s %14s %14s' % ('arm', 'params', 'front out', 'model out'))
    print('-' * 64)
    for name, over in ARMS:
        try:
            net, c = build(over)
            n = sum(p.numel() for p in net.parameters() if p.requires_grad)
            net.eval()
            with torch.no_grad():
                f = net.front(x)
                try:
                    y = net(x)
                    ys = str(tuple(y.shape)) if torch.is_tensor(y) else str(type(y))
                except Exception as e:
                    ys = 'N/A: ' + str(e)[:40]
            cfgstr = 'chan=%s ks=%s f_lo=%s init=%s' % (
                c.front_chan, c.front_ksize, c.gabor_f_lo, c.front_init)
            print('%-18s %10d %14s %14s   [%s]' % (
                name, n, str(tuple(f.shape)), ys, cfgstr))
        except Exception as e:
            print('%-18s  BUILD FAIL: %s' % (name, str(e)[:90]))


if __name__ == '__main__':
    main()
