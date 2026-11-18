#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
MRA-SNN 方法模块（移植自 other/MRA-SNN-main，SIGKDD'25）。
对齐 run_dgfdb.py 的统一协议（(B,1,1024) 输入 / 4 任务 / 5 种子）。

训练目标：纯 CE（忠实原版设置），作为 DG benchmark 的非域对齐 baseline。

TimeStep 机制：网络按 TimeStep=4 做时间步展开，输入需复制 4 次后送入
（原版 train.py 的做法），输出对 4 个时间步的平均 logits 求 CE。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.DictObj import DictObj
from utils.CalIndex import cal_index
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset

from models.mra_snn_net import MRASNN, TimeStep

configs = DictObj({
    'dataset_type': 'bearing',
    'use_cuda': True,
    'device': 'cpu',
    'data_length': 1024,
    'batch_size': 64,
    'num_classes': 14,
    'steps': 600,
    'checkpoint_freq': 1,
    'lr': 0.005,
})


class MRA_SNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.current_step = 0

        self.net = MRASNN(num_classes=int(configs.num_classes)).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_spike_input(x):
        """(B,1,L) -> (TimeStep*B, L)：每个时间步复制同一份输入。

        对齐原版 train.py 的 data[t*bs:(t+1)*bs] = ori_data。
        """
        if x.dim() == 3:
            x = x.squeeze(1)                 # (B, L)
        return x.repeat(TimeStep, 1)         # (TimeStep*B, L)

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device).long()

        data = self._to_spike_input(x)
        logits = self.net(data)
        loss = F.cross_entropy(logits, labels)

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = losses['loss_total']
        return losses

    # ------------------------------------------------------------------
    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        self.train()

        loss_acc_result = {'loss_total': [], 'loss_ce': [], 'acces': []}

        for step in range(1, self.steps + 1):
            self.current_step = step
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_total'].append(losses['loss_total'])
            loss_acc_result['loss_ce'].append(losses['loss_ce'])

            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                # 与 run_dgfdb 统一日志格式（便于统一解析/统计）
                self.logger.info('================Step {}================'.format(step))
                self.logger.info('loss_total: {:.4f} | ce: {:.4f}'.format(
                    float(losses['loss_total']), float(losses['loss_ce'])))
                self.logger.info('tgt_acc: {:.4f}'.format(
                    float(acc_results[0])))

        return loss_acc_result

    def test_model(self, loaders):
        self.eval()
        acc_results = []
        for loader in loaders:
            y_pred_lst = []
            y_true_lst = []
            for batched_data in loader:
                x, label_fault = batched_data
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)
                label_pred = self.predict(x)
                y_pred_lst.extend(label_pred.detach().cpu().data.numpy())
                y_true_lst.extend(label_fault.cpu().numpy())
            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst)
            acc_results.append(acc_i)
        self.train()
        return acc_results

    def predict(self, x):
        with torch.no_grad():
            data = self._to_spike_input(x)
            logits = self.net(data)
            return torch.max(logits, dim=1)[1]


def main(idx, configs):
    raise NotImplementedError('MRA_SNN 由 run_dgfdb.py 驱动，不单独 main 运行')
