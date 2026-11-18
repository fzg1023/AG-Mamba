#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
WIDAN（Interpretable Physics-informed wavelet Domain Adaptation Network）方法模块。
移植自 other/WIDAN-main，对齐 run_dgfdb.py 统一协议。

- 骨干：宽核 64 Laplace 小波初始化 ResNet18-1D + bottleneck(256)
- 对齐：JMMD（特征层 + softmax 层联合核），DG 协议下仅在 3 个源域间 pairwise
- lam 退火：2/(1+exp(-10*(epoch-middle)/(max-middle))) - 1
- middle_epoch 按原版 10/300 比例 -> 60 epoch 的 2
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.DictObj import DictObj
from utils.CalIndex import cal_index
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset

from models.widan_net import ResNet18Features, jmmd_loss

configs = DictObj({
    'dataset_type': 'bearing',
    'use_cuda': True,
    'device': 'cpu',
    'data_length': 1024,
    'batch_size': 64,
    'num_classes': 14,
    'steps': 600,
    'checkpoint_freq': 1,
    'lr': 0.001,
    'dim_feature': 448,
    'use_domain_weight': False,
    'weight_decay': 1e-5,
    'bottleneck_num': 256,
    'middle_epoch': 2,   # 原版 10/300 -> 60 epoch 的 2
    'max_epoch': 60,
    'wavelet_frequency': 64000,
    'wavelet_init': True,   # False -> 第一层用 kaiming 随机初始化（消融用）
    'widen': 1.0,           # ResNet 通道缩放（容量消融用）
})


class WIDAN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.middle_epoch = int(getattr(configs, 'middle_epoch', 2))
        self.max_epoch = int(getattr(configs, 'max_epoch', 60))
        bottleneck_num = int(getattr(configs, 'bottleneck_num', 256))

        self.features = ResNet18Features(
            wavelet_frequency=float(getattr(configs, 'wavelet_frequency', 64000)),
            wavelet_init=bool(getattr(configs, 'wavelet_init', True)),
            widen=float(getattr(configs, 'widen', 1.0)),
        ).to(self.device)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.features.output_num(), bottleneck_num),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ).to(self.device)
        self.classifier = nn.Linear(bottleneck_num,
                                    configs.num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.features.parameters())
            + list(self.bottleneck.parameters())
            + list(self.classifier.parameters()),
            lr=self.lr, weight_decay=float(getattr(configs, 'weight_decay', 1e-5)))
        self.current_step = 0

    # ------------------------------------------------------------------
    def _forward(self, x):
        feats = self.bottleneck(self.features(x))
        logits = self.classifier(feats)
        return feats, logits

    def _lam(self, epoch):
        if epoch < self.middle_epoch:
            return 0.0
        return 2.0 / (1.0 + math.exp(
            -10.0 * (epoch - self.middle_epoch)
            / max(self.max_epoch - self.middle_epoch, 1))) - 1.0

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device).long()

        feats, logits = self._forward(x)
        ce_loss = F.cross_entropy(logits, labels)

        epoch = (self.current_step - 1) // self.checkpoint_freq
        lam = self._lam(epoch)
        mmd_loss = torch.tensor(0.0, device=self.device)
        if lam > 0:
            B = self.batch_size
            feats_s = torch.split(feats, B, dim=0)
            softmax_s = torch.split(F.softmax(logits, dim=1), B, dim=0)
            n_dom = len(feats_s)
            pairs = [(i, j) for i in range(n_dom) for j in range(i + 1, n_dom)]
            total = 0.0
            for i, j in pairs:
                total = total + jmmd_loss(
                    [feats_s[i], softmax_s[i]], [feats_s[j], softmax_s[j]])
            mmd_loss = total / max(len(pairs), 1)

        loss = ce_loss + lam * mmd_loss

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_jmmd'] = mmd_loss.detach().cpu().data.numpy()
        losses['lam'] = lam
        return losses

    # ------------------------------------------------------------------
    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        self.train()

        loss_acc_result = {'loss_total': [], 'loss_ce': [], 'loss_jmmd': [],
                           'acces': []}

        for step in range(1, self.steps + 1):
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            for k in ['loss_total', 'loss_ce', 'loss_jmmd']:
                loss_acc_result[k].append(losses[k])

            self.logger.info('loss_total: {:.4f} | ce: {:.4f} | jmmd: {:.4f} | lam: {:.3f}'.format(
                float(losses['loss_total']), float(losses['loss_ce']),
                float(losses['loss_jmmd']), float(losses['lam'])))

            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                self.logger.info('tgt_acc: {:.4f}'.format(acc_results[0]))
                for i in range(1, len(acc_results)):
                    self.logger.info('src_acc: {:.4f}'.format(acc_results[i]))

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
            _, logits = self._forward(x)
            return torch.max(logits, dim=1)[1]


def main(idx, configs):
    raise NotImplementedError('WIDAN 由 run_dgfdb.py 驱动，不单独 main 运行')
