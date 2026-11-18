#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DANN（Domain-Adversarial Neural Network）方法模块。

移植自 DG-for-RMFD 原始实现（models/extractor.py + models/classifier.py +
models/AdversarialNet.py），对齐 run_dgfdb.py 的统一协议，使其可在
PHM2009 / PU / UO / HUST（含 9 源域）上运行。

构成：
- 骨干   Feature_extractor : 1D CNN k=25/15/5/5 -> 512 维
- 分类器 Classifier        : 512 -> 256 -> 256 -> num_classes
- 判别器 Discriminator     : 512 -> 1024 -> 1024 -> n_domains，GRL 反转 -1
- 损失   CE(src) + trade_off * CE(domain_pred)

超参与原版 DG_main.py 保持一致：lr=1e-3, trade_off=0.01, Adam(wd=1e-5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.DictObj import DictObj
from utils.CalIndex import cal_index

from models import Feature_extractor, Classifier, Discriminator

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
    'trade_off': 0.01,   # 原版 DG_main.py 默认值
})


class DANN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.trade_off = float(getattr(configs, 'trade_off', 0.01))
        n_src = int(getattr(configs, 'n_domains', 3))

        self.features = Feature_extractor().to(self.device)
        self.classifier = Classifier(configs.num_classes).to(self.device)
        self.discriminator = Discriminator(output_size=n_src).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.features.parameters())
            + list(self.classifier.parameters())
            + list(self.discriminator.parameters()),
            lr=self.lr, weight_decay=float(getattr(configs, 'weight_decay', 1e-5)))

    # ------------------------------------------------------------------
    def _forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return feats, logits

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

        feats, logits = self._forward(x)
        ce_loss = F.cross_entropy(logits, labels)

        # GRL 在 Discriminator.forward 内部通过 register_hook 反转梯度
        adv_logits = self.discriminator(feats)
        adv_loss = F.cross_entropy(adv_logits, domains)

        loss = ce_loss + self.trade_off * adv_loss

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
    raise NotImplementedError('DANN 由 run_dgfdb.py 驱动，不单独 main 运行')
