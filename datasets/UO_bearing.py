#!/usr/bin/python
# -*- coding:utf-8 -*-
"""UO (University of Ottawa) 时变转速轴承数据集 -> DG 任务加载器。

数据来源: DIS-main 项目已用 scripts/prepare_uo.py 从原始 .mat 预处理好的
  {UO_x.npy, UO_y.npy, UO_rpm.npy} (可选数据集，默认未接入；路径见 datasets/paths.py 的 UO_DIR)。
  - UO_x.npy: (12000, 4096) float32, 振动信号分段 (200kHz 采样, 20.48ms/段)
  - UO_y.npy: (12000, 2) int64, [class_id, file_id]; file_id = orig_domain*5+class_id
              orig_domain 0..3 对应原始转速曲线 A(升速)/B(降速)/C(先升后降)/D(先降后升)
  - UO_rpm.npy: (12000, 4096) float32, 每点瞬时轴转速 (Hz, 编码器脉冲率/CPR)

关键设计决策 (见与用户的分域方案讨论):
  原始 A/B/C/D 四种转速曲线的转速覆盖区间几乎完全重叠 (经
  DIS-main/scripts/analyze_uo_domains.py 验证: 同转速下 A-B 频谱残差远小于
  同一域内低速-高速的频谱差异)，直接以 A/B/C/D 作为域标签会退化为插值任务，
  达不到验证"跨转速偏移"泛化能力的目的。

  因此本加载器废弃原始 A/B/C/D 标签，改为将全部 12000 段样本按其"瞬时平均
  转速" (rpm.mean(axis=-1)) 池化后按分位数重新切分为 4 个转速域 (与 PU/PHM2009
  的 4-domain leave-one-out 协议对齐):
    domain 0 (Low):      ~33-57 Hz
    domain 1 (MidLow):   ~57-66 Hz
    domain 2 (MidHigh):  ~66-76 Hz
    domain 3 (High):     ~76-102 Hz
  分位数边界 SPEED_EDGES 由 inspect_uo_speed_bins.py 预先计算并固定，保证跨
  任务/种子的域划分完全可复现。每个(域,类别)组合样本量最少 450 条，train/test
  各取 200/50 条时样本充足 (见 inspect_uo_speed_bins.py 输出)。
"""

import os
import numpy as np
import pandas as pd

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

num_cls = 5
sig_size = 4096
cls_names = {0: 'H', 1: 'B', 2: 'I', 3: 'O', 4: 'C'}

# 域名仅用于日志展示，实际划分依据 SPEED_EDGES 对瞬时转速分位数切分
total_conditions = {0: 'Low_33-57Hz_', 1: 'MidLow_57-66Hz_',
                     2: 'MidHigh_66-76Hz_', 3: 'High_76-102Hz_'}

# 全量数据集(12000段)瞬时转速的 25/50/75 分位数边界 (Hz)，固定值见
# inspect_uo_speed_bins.py 的输出，保证可复现。
SPEED_EDGES = [57.22, 66.29, 75.67]

_CACHE = {}


def _load_all(root):
    if root in _CACHE:
        return _CACHE[root]
    x = np.load(os.path.join(root, 'UO_x.npy'))
    y = np.load(os.path.join(root, 'UO_y.npy'))
    rpm = np.load(os.path.join(root, 'UO_rpm.npy'))
    labels = y[:, 0]
    shaft_hz = rpm.mean(axis=-1)
    domain_id = np.digitize(shaft_hz, np.asarray(SPEED_EDGES, dtype=np.float64))
    _CACHE[root] = (x, labels, domain_id)
    return _CACHE[root]


def data_load(root, N, num_samples):
    x, labels, domain_id = _load_all(root)
    data = []
    label = []
    for lab in range(num_cls):
        idx = np.where((domain_id == N) & (labels == lab))[0]
        if len(idx) < num_samples:
            raise ValueError(
                'UO domain {} class {} ({}) 样本不足: 需要 {}，实际 {}'.format(
                    N, lab, cls_names[lab], num_samples, len(idx)))
        idx = idx[:num_samples]
        for i in idx:
            seg = x[i].reshape(-1, 1).astype(np.float32)
            data.append(seg)
            label.append(lab)
    return [data, label]


class UO_DG(object):
    def __init__(self, transfer_task, normalizetype, num_train_samples, num_test_samples):
        from datasets.paths import UO_DIR as _UO_DIR
        self.data_dir = os.environ.get('AGM_UO_DIR', _UO_DIR)
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        if isinstance(self.target_N, (list, tuple)):
            self.target_N = self.target_N[0]
        self.normalizetype = normalizetype
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normalizetype),
                Retype(),
            ]),
            'test': Compose([
                Reshape(),
                Normalize(self.normalizetype),
                Retype(),
            ])
        }

    def train_test_data_split(self):
        # get source_1
        list_data = data_load(self.data_dir, self.source_N[0], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_1:\n', data_pd)
        source_1 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get source_2
        list_data = data_load(self.data_dir, self.source_N[1], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_2:\n', data_pd)
        source_2 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get source_3
        list_data = data_load(self.data_dir, self.source_N[2], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_3:\n', data_pd)
        source_3 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get target
        list_data = data_load(self.data_dir, self.target_N, self.num_test_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('target:\n', data_pd)
        target = dataset(list_data=data_pd, transform=self.data_transforms['test'])

        return source_1, source_2, source_3, target, num_cls
