#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 数据集 -> DG 任务加载器（**域 = 不同转速，载荷固定**）。

只服务于"附加试验"：不参与论文的任何数字，也不进 `docs/manuscript/`。

数据来源: Politecnico di Torino 滚动轴承台架（Daga et al., MSSP 120 (2019) 252-273）
  说明书 PDF 在 `data/DIRG_bearing/`；预处理见 `tools/prepare_dirg.py`。
  - 7 类: 0A 无缺陷；1A/2A/3A 内圈压痕 450/250/150 um；4A/5A/6A 滚动体压痕 450/250/150 um
  - 域 = 名义转速 100/200/300/400/500 Hz（100 Hz = 6000 rpm）
  - **载荷固定**在一个档位（0 / ~1000 / ~1400 / ~1800 N），故域间只变转速
  - 实测载荷档覆盖（见 prepare_dirg 的 meta.csv）：
        L1000 -> 5 个转速（100..500），每域每类 500 窗
        L1400 -> 4 个转速（100..400）
        L1800 -> 3 个转速（100..300）
  - 采样率 51,200 Hz，每文件 10 s；单通道（prepare 时选定，默认 Ch2 = Radial y）

DG 协议: leave-one-domain-out（LODO），其余转速为源域、留一转速为目标域。
  源域训练取每类前 num_train_samples 窗，目标域测试取每类**末尾**窗（与源域不重叠）。

用法:
  from datasets.DIRG_bearing import DIRG_DG
  srcs, tar, num_cls = DIRG_DG([[1,2,3,4],[0]], 1000, 'mean-std', 50, 50)\\
      .train_test_data_split()
"""
import os

import numpy as np
import pandas as pd

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *          # noqa: F401,F403

num_cls = 7
sig_size = 1024

CLASSES = ['0A', '1A', '2A', '3A', '4A', '5A', '6A']
CLASS_DESC = {
    '0A': 'no defect',
    '1A': 'inner ring dent 450 um',
    '2A': 'inner ring dent 250 um',
    '3A': 'inner ring dent 150 um',
    '4A': 'roller dent 450 um',
    '5A': 'roller dent 250 um',
    '6A': 'roller dent 150 um',
}
SPEEDS = [100, 200, 300, 400, 500]
DOMAINS = {i: '%dHz' % s for i, s in enumerate(SPEEDS)}
LOAD_TAG = {0: 'L0', 1000: 'L1000', 1400: 'L1400', 1800: 'L1800'}
# 每个载荷档实际可用的**域索引**（域索引恒对应 SPEEDS 的下标）
DOMAIN_IDS_BY_LOAD = {
    1000: [0, 1, 2, 3, 4],
    1400: [0, 1, 2, 3, 4],      # 400 档: 见下方 SPEED_SET_BY_LOAD 的过滤
    1800: [0, 1, 2, 3, 4],
}
SPEED_SET_BY_LOAD = {           # 该档实际存在的转速
    1000: [100, 200, 300, 400, 500],
    1400: [100, 200, 300, 400],
    1800: [100, 200, 300],
}
FS = 51200.0

_CACHE = {}
_LP_CACHE = {}


def _npz_path(ch, fs=51200):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dirg_data')
    if int(fs) == 51200:
        return os.path.join(d, 'dirg_x1024_ch%d.npz' % ch)
    return os.path.join(d, 'dirg_x1024_ch%d_fs%d.npz' % (ch, int(fs)))


def _load_all(ch, fs=51200):
    if (ch, int(fs)) not in _CACHE:
        p = _npz_path(ch, fs)
        if not os.path.exists(p):
            raise FileNotFoundError(
                '{} 不存在，请先运行 tools/prepare_dirg.py --ch {}'.format(p, ch))
        _CACHE[(ch, int(fs))] = np.load(p)
    return _CACHE[(ch, int(fs))]


def available_domains(load, ch=1, fs=51200):
    """返回 (域索引列表, 对应转速列表)：该载荷档在 npz 里真正齐全的转速。"""
    z = _load_all(ch, fs)
    tag = LOAD_TAG[int(load)]
    ids, sps = [], []
    for i, s in enumerate(SPEEDS):
        if all('%d_%s_%s' % (s, c, tag) in z.files for c in CLASSES):
            ids.append(i)
            sps.append(s)
    return ids, sps


def data_load(domain_id, load, num_samples, mode='head', ch=1, fs=51200,
              sig=1024, lp_hz=0):
    """取某域某载荷档每类 num_samples 条 1024 点样本。

    mode='head' 取每类前 num_samples 窗（源域训练）；
    mode='tail' 取每类末尾 num_samples 窗（目标域测试，与源域不重叠）。
    """
    z = _load_all(ch, fs)
    speed = SPEEDS[domain_id]
    tag = LOAD_TAG[int(load)]
    data, label = [], []
    for lab, cls in enumerate(CLASSES):
        key = '%d_%s_%s' % (speed, cls, tag)
        if key not in z.files:
            raise KeyError('npz 缺少 %s（该转速/载荷档不存在或未 prepare）' % key)
        arr = z[key]                                   # (n, 1024)
        # lp_hz>0：保持 fs 不变，把该频率以上滤掉。
        # 关键：在**整段连续记录**上做零相位 rFFT 掩码，而不是逐窗 FFT
        # （逐窗 FFT 等于循环卷积，砖墙截止会在窗内产生 Gibbs 振铃）。
        # 掩码带升余弦过渡，避免激进的砖墙。
        if lp_hz:
            ck = (int(ch), int(fs), float(lp_hz), key)
            if ck not in _LP_CACHE:
                x = np.asarray(arr, dtype=np.float64).reshape(-1)
                n = x.size
                f = np.fft.rfftfreq(n, d=1.0 / float(fs))
                m = np.ones_like(f)
                lo = float(lp_hz) * 0.85
                m[f >= float(lp_hz)] = 0.0
                tr = (f > lo) & (f < float(lp_hz))
                m[tr] = 0.5 * (1.0 + np.cos(
                    np.pi * (f[tr] - lo) / (float(lp_hz) - lo)))
                y = np.fft.irfft(np.fft.rfft(x) * m, n=n)
                _LP_CACHE[ck] = y.reshape(arr.shape).astype(np.float32)
            arr = _LP_CACHE[ck]
        # sig>1024：把相邻的 1024 点窗**拼回连续时序**（npz 里窗是连续不重叠的），
        # 于是得到真正的更长窗（2048 点 @51200 = 40 ms），而不是插值放大。
        if int(sig) % 1024:
            raise ValueError('sig 必须是 1024 的整数倍（当前 %s）' % sig)
        m = max(1, int(sig) // 1024)
        if m > 1:
            n_use = (len(arr) // m) * m
            arr = arr[:n_use].reshape(n_use // m, 1024 * m)
        seg = arr[:num_samples] if mode == 'head' else arr[-num_samples:]
        if len(seg) < num_samples:
            raise ValueError('DIRG %s 类%s 样本不足: 需要 %d, 实际 %d'
                             % (DOMAINS[domain_id], cls, num_samples, len(seg)))
        for s in seg:
            data.append(np.asarray(s, dtype=np.float32).reshape(-1, 1))
            label.append(lab)
    return [data, label]


class DIRG_DG(object):
    """DIRG 同载荷、异转速 LODO DG 数据集（最多 5 域，7 类）。"""

    def __init__(self, transfer_task, load=1000, norm_type='mean-std',
                 num_train_samples=50, num_test_samples=50, ch=1, fs=51200,
                 sig=1024, lp_hz=0):
        self.source_N = list(transfer_task[0])
        self.target_N = transfer_task[1]
        if isinstance(self.target_N, (list, tuple)):
            self.target_N = self.target_N[0]
        self.load = int(load)
        self.norm_type = norm_type
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.ch = ch
        self.fs = int(fs)
        self.sig = int(sig)
        self.lp_hz = float(lp_hz)

        self.data_transforms = {
            'train': Compose([Reshape(), Normalize(self.norm_type), Retype()]),
            'test': Compose([Reshape(), Normalize(self.norm_type), Retype()]),
        }

    def train_test_data_split(self):
        srcs = []
        for i, d in enumerate(self.source_N):
            ld = data_load(d, self.load, self.num_train_samples,
                           mode='head', ch=self.ch, fs=self.fs, sig=self.sig, lp_hz=self.lp_hz)
            df = pd.DataFrame({'data': ld[0], 'label': ld[1]})
            print('source_{} ({}): {} 条'.format(
                i + 1, DOMAINS[d], len(df)), flush=True)
            srcs.append(dataset(list_data=df,
                                transform=self.data_transforms['train']))

        ld = data_load(self.target_N, self.load, self.num_test_samples,
                       mode='tail', ch=self.ch, fs=self.fs, sig=self.sig, lp_hz=self.lp_hz)
        df = pd.DataFrame({'data': ld[0], 'label': ld[1]})
        print('target ({}): {} 条'.format(DOMAINS[self.target_N], len(df)),
              flush=True)
        tar = dataset(list_data=df, transform=self.data_transforms['test'])
        return srcs, tar, num_cls
