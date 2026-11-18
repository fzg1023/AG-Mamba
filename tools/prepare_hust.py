#!/usr/bin/python
# -*- coding:utf-8 -*-
"""HUSTbearing 数据集准备。

10 个恒定转速域 x 9 类健康状态，提取 X 轴加速度，切成 1024 点样本。
- 采样率 25.6 kHz，每文件 262144 点 -> 256 个无重叠样本
- 9 类: H(健康), I/O/B/C(严重), 0.5X_I/O/B/C(中等)
- 10 域: 20/25/30/35/40/60/65/70/75/80 Hz

输出: datasets/hust_data/hust_x1024.npz  (key = '{speed}_{cls}')
"""
import os

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from datasets.paths import HUST_DIR
RAW = _os.path.join(HUST_DIR, 'raw data')
OUT = os.environ.get(
    'AGM_HUST_NPZ',
    os.path.join(os.environ.get('AGM_DATA_ROOT',
                                os.path.join(os.path.dirname(os.path.dirname(
                                    os.path.abspath(__file__))), 'data')),
                 'HUST', 'hust_x1024.npz'))
OUT_DIR = os.path.dirname(OUT)

SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
CLASSES = ['H', 'I', 'O', 'B', 'C', '0.5X_I', '0.5X_O', '0.5X_B', '0.5X_C']
WIN = 1024
SKIPROWS = 22      # 第 1-22 行为 DAQ 头部
X_COL = 2          # 列: Time, Speed, X, Y, Z


def find_file(cls, speed):
    for suffix in ['Hz', 'hz', 'HZ', 'HZ']:
        p = os.path.join(RAW, '{}_{}{}.xls'.format(cls, speed, suffix))
        if os.path.exists(p):
            return p
    raise FileNotFoundError('{}_{}Hz.xls not found'.format(cls, speed))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {}
    for speed in SPEEDS:
        for cls in CLASSES:
            f = find_file(cls, speed)
            sig = pd.read_csv(f, sep=r'\s+', skiprows=SKIPROWS, header=None,
                              usecols=[X_COL], dtype=np.float32)
            s = sig.values[:, 0]
            n = len(s) // WIN
            arr = s[:n * WIN].reshape(n, WIN).astype(np.float32)
            key = '{}_{}'.format(speed, cls)
            data[key] = arr
            print('{:>3d}Hz {:>7s} -> {} samples  ({})'.format(
                speed, cls, n, os.path.basename(f)), flush=True)

    np.savez_compressed(OUT, **data)
    print('\nsaved {} arrays to {}'.format(len(data), OUT))
    print('total samples per domain:', sum(
        data['{}_{}'.format(SPEEDS[0], c)].shape[0] for c in CLASSES))


if __name__ == '__main__':
    main()
