#!/usr/bin/python
# -*- coding:utf-8 -*-
"""HUSTbearing 数据集 -> DG 任务加载器（10 个恒定转速域，LODO）。

数据来源: 华中科技大学 HUSTbearing（Zhao et al., RESS 2024）
  - 采样率 25.6 kHz，每文件 262144 点 (10.2 s)，取 X 轴加速度
  - 10 个恒定转速域: 20/25/30/35/40/60/65/70/75/80 Hz
  - 9 类健康状态: H(健康)；I/O/B/C(严重 内圈/外圈/滚动体/复合)；
                  0.5X_I/0.5X_O/0.5X_B/0.5X_C(中等程度)
  - 每类每域 256 条 1024 点无重叠样本（由 prepare_hust.py 预先切分）

DG 协议: leave-one-domain-out (LODO)，9 源域 -> 1 目标域，共 10 个任务。
  域间同为恒定转速，但转速跨度 20~80 Hz（4 倍），构成真实的域偏移。

用法:
  from datasets.HUST_bearing import HUST_DG
  srcs, tar, num_cls = HUST_DG([[0,1,2,3,4,5,6,7,8], [9]], 'mean-std', 50, 50)\\
      .train_test_data_split()
"""
import os

import numpy as np
import pandas as pd

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from datasets.paths import HUST_NPZ

num_cls = 9
sig_size = 1024

DOMAINS = {0: '20Hz', 1: '25Hz', 2: '30Hz', 3: '35Hz', 4: '40Hz',
           5: '60Hz', 6: '65Hz', 7: '70Hz', 8: '75Hz', 9: '80Hz'}
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
CLASSES = ['H', 'I', 'O', 'B', 'C', '0.5X_I', '0.5X_O', '0.5X_B', '0.5X_C']
CLASS_NAMES = {i: c for i, c in enumerate(CLASSES)}

NPZ = HUST_NPZ

_CACHE = {}


def _load_all():
    if 'npz' not in _CACHE:
        if not os.path.exists(NPZ):
            raise FileNotFoundError(
                '{} 不存在，请先运行 prepare_hust.py'.format(NPZ))
        _CACHE['npz'] = np.load(NPZ)
    return _CACHE['npz']


def data_load(domain_id, num_samples, mode='head'):
    """取 domain_id 域每类 num_samples 条 1024 点样本。

    mode='head' 取前 num_samples 条（源域训练用）；
    mode='tail' 取最后 num_samples 条（目标域测试用，与源域不重叠）。
    """
    z = _load_all()
    speed = SPEEDS[domain_id]
    data, label = [], []
    for lab, cls in enumerate(CLASSES):
        arr = z['{}_{}'.format(speed, cls)]        # (256, 1024)
        seg = arr[:num_samples] if mode == 'head' else arr[-num_samples:]
        if len(seg) < num_samples:
            raise ValueError('HUST 域{} 类{} 样本不足: 需要 {}, 实际 {}'.format(
                DOMAINS[domain_id], cls, num_samples, len(seg)))
        for s in seg:
            data.append(s.reshape(-1, 1).astype(np.float32))
            label.append(lab)
    return [data, label]


class HUST_DG(object):
    """HUSTbearing LODO DG 数据集（10 域，9 类）。"""

    def __init__(self, transfer_task, norm_type='mean-std',
                 num_train_samples=50, num_test_samples=50):
        self.source_N = list(transfer_task[0])
        self.target_N = transfer_task[1]
        if isinstance(self.target_N, (list, tuple)):
            self.target_N = self.target_N[0]
        self.norm_type = norm_type
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.data_transforms = {
            'train': Compose([Reshape(), Normalize(self.norm_type), Retype()]),
            'test': Compose([Reshape(), Normalize(self.norm_type), Retype()]),
        }

    def train_test_data_split(self):
        srcs = []
        for i, d in enumerate(self.source_N):
            ld = data_load(d, self.num_train_samples, mode='head')
            df = pd.DataFrame({'data': ld[0], 'label': ld[1]})
            print('source_{} ({}): {} 条'.format(
                i + 1, DOMAINS[d], len(df)), flush=True)
            srcs.append(dataset(list_data=df,
                                transform=self.data_transforms['train']))

        ld = data_load(self.target_N, self.num_test_samples, mode='tail')
        df = pd.DataFrame({'data': ld[0], 'label': ld[1]})
        print('target ({}): {} 条'.format(DOMAINS[self.target_N], len(df)),
              flush=True)
        tar = dataset(list_data=df, transform=self.data_transforms['test'])

        return srcs, tar, num_cls
