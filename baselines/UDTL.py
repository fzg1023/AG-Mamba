#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
UDTL（Unsupervised Deep Transfer Learning，ResNet18-1D + 多源域对抗）方法模块。
移植自 other/UDTL_multi_domain-main，对齐 run_dgfdb.py 的统一协议
（1024 长度、60 epoch、5 种子、leave-one-condition-out）。

DG 协议下目标域不可见，域对抗仅对齐 3 个源域之间；
middle_epoch 按原版比例（50/300）映射到 60 epoch → 10。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.DictObj import DictObj
from utils.CalIndex import cal_index
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset

from models.udtl_net import UDTLNet

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
    'hidden_size': 1024,
    'middle_epoch': 10,  # 原版 50/300 比例 -> 60 epoch 的 10
    'trade_off_adversarial': 'Step',
    'lam_adversarial': 1.0,
})


class UDTL(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.middle_epoch = int(getattr(configs, 'middle_epoch', 10))

        iters_per_epoch = max(1, configs.steps // 60)
        # GRL 系数退火步数：域对抗阶段的迭代总数（原版 max_iter 语义）
        adv_iters = max(1, iters_per_epoch * (60 - self.middle_epoch))

        self.net = UDTLNet(
            num_classes=configs.num_classes,
            bottleneck_num=int(getattr(configs, 'bottleneck_num', 256)),
            # 源域个数随数据集变化（PHM2009/PU=3，HUST=9），由 run_dgfdb 写入
            n_domains=int(getattr(configs, 'n_domains', 3)),
            hidden_size=int(getattr(configs, 'hidden_size', 1024)),
            max_iter=float(adv_iters),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr,
            weight_decay=float(getattr(configs, 'weight_decay', 1e-5)))
        self.current_step = 0

    # ------------------------------------------------------------------
    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device).long()

        doms = []
        for i, (_, y) in enumerate(minibatches):
            doms.append(torch.full((len(y),), i, dtype=torch.long,
                                   device=self.device))
        domains = torch.cat(doms)

        feats, logits = self.net(x)
        ce_loss = F.cross_entropy(logits, labels)

        epoch = (self.current_step - 1) // self.checkpoint_freq
        if epoch < self.middle_epoch:
            adv_loss = torch.tensor(0.0, device=self.device)
        else:
            adv_out = self.net.adversarial(feats)
            adv_loss = F.cross_entropy(adv_out, domains)

        loss = ce_loss + adv_loss

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_adv'] = adv_loss.detach().cpu().data.numpy()
        return losses

    # ------------------------------------------------------------------
    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        self.train()

        loss_acc_result = {'loss_total': [], 'loss_ce': [], 'loss_adv': [],
                           'acces': []}

        for step in range(1, self.steps + 1):
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            for k in ['loss_total', 'loss_ce', 'loss_adv']:
                loss_acc_result[k].append(losses[k])

            self.logger.info('loss_total: {:.4f} | ce: {:.4f} | adv: {:.4f}'.format(
                float(losses['loss_total']), float(losses['loss_ce']),
                float(losses['loss_adv'])))

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
            _, logits = self.net(x)
            return torch.max(logits, dim=1)[1]


def main(idx, configs):
    raise NotImplementedError('UDTL 由 run_dgfdb.py 驱动，不单独 main 运行')
