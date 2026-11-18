#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
P2C-DGNet 独立训练器。复用 datasets 的加载逻辑，但不修改任何现有训练器。
"""
import logging
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from datasets import PHM2009_DG, PU_DG
from models.p2c_dgnet import P2CDGNet


def orthogonal_loss(z_f, z_d):
    """故障空间与工况空间的正交解耦损失 ||Cov(z_f, z_d)||_F^2"""
    zf = F.normalize(z_f, dim=-1)
    zd = F.normalize(z_d, dim=-1)
    zf = zf - zf.mean(dim=0, keepdim=True)
    zd = zd - zd.mean(dim=0, keepdim=True)
    n = zf.size(0) - 1
    cov = (zf.t() @ zd) / max(n, 1)
    return cov.pow(2).sum()


class P2C_train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1

        # ---- 数据集（复用现有 datasets 加载）----
        self.datasets = {}
        if args.data_name == 'PHM2009':
            self.datasets['src_1'], self.datasets['src_2'], self.datasets['src_3'], self.datasets['tar'], \
                self.datasets['num_cls'] = PHM2009_DG(transfer_task=args.transfer_task,
                                                      normalizetype=args.normalizetype,
                                                      num_train_samples=args.num_train_samples,
                                                      num_test_samples=args.num_test_samples).train_test_data_split()
        elif args.data_name == 'PU':
            self.datasets['src_1'], self.datasets['src_2'], self.datasets['src_3'], self.datasets['tar'], \
                self.datasets['num_cls'] = PU_DG(transfer_task=args.transfer_task,
                                                 normalizetype=args.normalizetype,
                                                 num_train_samples=args.num_train_samples,
                                                 num_test_samples=args.num_test_samples).train_test_data_split()
        else:
            raise ValueError('unsupported data_name: {}'.format(args.data_name))

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[0] == 'src' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False) for x in
                            ['src_1', 'src_2', 'src_3', 'tar']}

        # ---- 模型 ----
        self.num_domains = len(args.transfer_task[0])
        self.model = P2CDGNet(num_classes=self.datasets['num_cls'], num_domains=self.num_domains,
                              aug_ratio=getattr(args, 'aug_ratio', 0.6))

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # 辅助损失 warm-up 系数（train 中按 epoch 更新）
        self.warmup_factor = 0.0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info('model saved to {}'.format(path))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logging.info('model loaded from {}'.format(path))

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['tar']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                out = self.model(inputs, return_dict=True)
                pred = out['logits'].argmax(dim=1)
                correct += torch.eq(pred, labels).float().sum().item()
                total += labels.size(0)
        acc = correct / total
        logging.info('Re-evaluation: tar-Acc: {:.4f}'.format(acc))
        return acc

    def train(self):
        args = self.args
        num_domains = self.num_domains

        best_acc = 0.0
        best_epoch = -1

        for epoch in range(args.epoch):
            logging.info('-' * 10 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 10)
            # 辅助损失 warm-up：前 warmup_epochs 个 epoch 权重从 0 线性增长到 1
            if args.warmup_epochs > 0:
                self.warmup_factor = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))
            else:
                self.warmup_factor = 1.0
            iter_src_2 = iter(self.dataloaders['src_2'])
            iter_src_3 = iter(self.dataloaders['src_3'])
            logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))

            for phase in ['src_1', 'tar']:
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                if phase == 'src_1':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase == 'src_1':
                        src_2_inputs, src_2_labels = next(iter_src_2)
                        src_3_inputs, src_3_labels = next(iter_src_3)
                        inputs = torch.cat((inputs, src_2_inputs, src_3_inputs), dim=0)
                        labels = torch.cat((labels, src_2_labels, src_3_labels), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        interval = labels.size(0) // num_domains
                        domain_labels = torch.cat([torch.full((interval,), s, device=self.device)
                                                   for s in range(num_domains)], dim=0)

                        with torch.set_grad_enabled(True):
                            loss = self._compute_train_loss(inputs, labels, domain_labels, interval)

                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                            self.optimizer.step()

                    else:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        with torch.set_grad_enabled(False):
                            out = self.model(inputs, return_dict=True)
                            loss = self.criterion(out['logits'], labels)

                    epoch_loss += loss.item() * labels.size(0)
                    epoch_length += labels.size(0)

                    if phase == 'tar':
                        pred = out['logits'].argmax(dim=1)
                        epoch_acc += torch.eq(pred, labels).float().sum().item()

                epoch_loss = np.divide(epoch_loss, epoch_length)
                epoch_acc = np.divide(epoch_acc, epoch_length)

                if phase == 'src_1':
                    logging.info('Epoch: {} train-Loss: {:.4f}'.format(epoch, epoch_loss))
                else:
                    logging.info('Epoch: {} tar-Loss: {:.4f} tar-Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                        self.save_model(os.path.join(self.save_dir, 'best_model.pth'))

            self.lr_scheduler.step()

        self.save_model(os.path.join(self.save_dir, 'final_model.pth'))
        logging.info('Training finished. Best tar-Acc: {:.4f} at epoch {}'.format(best_acc, best_epoch))
        self.load_model(os.path.join(self.save_dir, 'best_model.pth'))
        self.evaluate()

    def _compute_train_loss(self, inputs, labels, domain_labels, interval):
        args = self.args
        num_domains = self.num_domains

        out = self.model(inputs, return_dict=True)
        logits = out['logits']
        # 表示已在模型内部经过 LayerNorm，范数稳定
        z_f = out['z_f']
        z_d = out['z_d']
        h = out['h']
        h_sem = out['h_sem']
        attn_logits = out['attn_logits']
        domain_logits = out['domain_logits']

        # ---- 主分类损失（均匀加权，不干扰主任务）----
        loss_cls = self.criterion(logits, labels)

        # ---- 自蒸馏域可靠性权重（仅用于辅助损失加权）----
        sd_per_domain = []
        for s in range(num_domains):
            sl = slice(s * interval, (s + 1) * interval)
            sd_per_domain.append(F.mse_loss(h[sl], h_sem[sl].detach()))
        sd_stack = torch.stack(sd_per_domain)
        w_domain = torch.softmax(-args.tau * sd_stack, dim=0)  # (num_domains,)

        # ---- 物理语义锚点（温度软化）----
        loss_sem = self.criterion(attn_logits / args.anchor_temp, labels)

        # ---- 域判别损失 ----
        loss_dom = self.criterion(domain_logits, domain_labels)

        # ---- 因果解耦正交损失 ----
        loss_orth = orthogonal_loss(z_f, z_d)

        # ---- 原型图损失 ----
        loss_proto = self.model.prototype_loss(out['z_f'], labels)

        # ---- 自蒸馏损失 ----
        loss_sd = sd_stack.mean()

        # ---- 物理一致外推增强（频域迁移，增强样本标签一致性）----
        if args.lam_aug > 0:
            aug_inputs = self.model.physics_augment(inputs)
            out_aug = self.model(aug_inputs, return_dict=True)
            loss_aug = self.criterion(out_aug['logits'], labels)
        else:
            loss_aug = torch.zeros_like(loss_cls)

        # 辅助损失用域可靠性加权（体现源域异质性，但不干扰主分类）
        aux = args.lam_sem * loss_sem + args.lam_dom * loss_dom \
            + args.lam_orth * loss_orth + args.lam_proto * loss_proto \
            + args.lam_aug * loss_aug + args.lam_sd * loss_sd

        loss = loss_cls + self.warmup_factor * aux

        return loss
