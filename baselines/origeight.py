#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""origeight 家族（原仓库 8 个基线中的 7 个）的统一实现。

来源与保真度
------------
本文件是**对原始实现的移植**，不是重新发明：

* `utils/train.py`（原 8 方法的训练器）给出了 AGG / DANN / MMD / CORAL /
  Triplet_loss / Mixup 的精确损失结构；
* `utils/MLDG_train.py` 给出 MLDG 的内环/外环元更新；
* `utils/DAEL_train.py` 给出 DAEL 的「三分类器 + 跨域集成一致性」。

`docs/legacy/方法对比分析报告.md` §2 明确：这 8 个方法「共享同一套轻量主干：
4 层 1D CNN 特征提取器（卷积核 25/15/5/5，输出 512 维）+ MLP 分类器
（512→256→256→类别数）」——即 `models/extractor.py::Feature_extractor` +
`models/classifier.py::Classifier`。因此这里沿用同一主干，属于**复现原架构**。

原样保留的细节
--------------
* 6 个方法共用**同一个** `trade_off`（原代码里就是单一 `args.trade_off`），
  取 0.01，与已移植的 `baselines/DANN.py` 一致；
* `TripletLoss(margin=2)`（原代码硬编码 margin=2，非 `models/loss.py` 的默认 0.3）；
* CORAL / MMD 对 3 个源域取**全部 3 个配对**再乘 `trade_off/3`；
* Mixup 用 `models/Mixup.py::mixup_augmentation`（Dirichlet λ，跨域插值后按 λ 加权 CE）。

必要的一般化（n_src ≠ 3 时，如 HUST 的 9 源域）
-----------------------------------------------
* CORAL/MMD：改为**全部配对** `i<j` 的均值（n_src=3 时与原来的 3 对逐位等价）；
* Mixup：Dirichlet(1,...,1) 泛化为 n_src 个分量（n_src=3 时与原来一致）；
* DAEL：分类器数与源域数一致（n_src=3 时与原来的 3 个一致）；
* MLDG：随机留 1 域作 meta-test、其余作 meta-train（n_src=3 时即原逻辑）。

日志格式
--------
统一输出 `tgt_acc: {:.4f}`（与 DANN/MRA_SNN/UDTL/WIDAN 一致），
供 `experiments/run_compare.py::parse_accs` 解析。
注：ADN/CCDG/CCN/CDDG/CNNC/DDGFD/DGNIS/IEDGNet/WhiteningNet 这 9 个移植方法用的
是历史标签 `tgt_train_acc:`；`parse_accs` 已同时兼容两种。
"""
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.DictObj import DictObj
from utils.CalIndex import cal_index
from models import (Feature_extractor, Classifier, Discriminator, MMD, CORAL,
                    TripletLoss, mixup_augmentation)

configs = DictObj({
    'dataset_type': 'bearing',
    'use_cuda': True,
    'device': 'cpu',
    'data_length': 1024,
    'batch_size': 64,
    'num_classes': 14,
    'steps': 1500,
    'checkpoint_freq': 1,
    'lr': 0.001,
    'dim_feature': 448,
    'use_domain_weight': False,
    'weight_decay': 1e-5,
    'trade_off': 0.01,        # 原 utils/train.py 的单一超参；与 DANN 移植值一致
    'use_learning_rate_sheduler': True,
    'cos': False,
    'schedule': [500, 1000],
})


class _OrigeightBase(nn.Module):
    """origeight 家族的统一骨架：共享主干 + 训练/评估循环。

    子类只需实现 `extra_loss(...)`（返回一个标量，加在 CE 上）。
    需要多分类器的（DAEL）或元学习的（MLDG）重写 `update`。
    """

    n_cls_heads = 1          # 分类器个数（DAEL 用 n_src 个）

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.weight_decay = float(getattr(configs, 'weight_decay', 1e-5))
        self.use_learning_rate_sheduler = bool(
            getattr(configs, 'use_learning_rate_sheduler', True))
        self.n_src = int(getattr(configs, 'n_domains', 3))
        # 6 个方法共用同一 trade_off（原代码即单一 args.trade_off）
        self.trade_off = float(getattr(configs, 'trade_off', 0.01))

        self.Gf = Feature_extractor().to(self.device)
        self.Gc = Classifier(configs.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.Gf.parameters()) + list(self.Gc.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def adjust_learning_rate(self, step):
        """统一协议：cos -> 余弦退火；否则阶梯衰减（milestone 处 ×0.5）。"""
        lr = self.lr
        if getattr(self.configs, 'cos', False):
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:
            for milestone in (getattr(self.configs, 'schedule', None) or []):
                lr *= 0.5 if step >= milestone else 1.0
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    # ------------------------------------------------------------------
    def _split(self, minibatches):
        """把各域 minibatch 拼成 (x, labels, interval, feats 用的域列表)。"""
        xs = [x for x, _ in minibatches]
        ys = [y for _, y in minibatches]
        interval = int(xs[0].size(0))
        x = torch.cat(xs).to(self.device)
        labels = torch.cat(ys).to(self.device).long()
        return x, labels, interval, xs, ys

    def extra_loss(self, features, logits, labels, interval, n_src):
        """子类重写：返回附加损失（标量）。默认 0。"""
        return torch.zeros((), device=self.device, dtype=logits.dtype)

    # ------------------------------------------------------------------
    def update(self, minibatches):
        x, labels, interval, _, _ = self._split(minibatches)
        n_src = len(minibatches)

        features = self.Gf(x)
        logits = self.Gc(features)
        ce_loss = F.cross_entropy(logits, labels)
        aux = self.extra_loss(features, logits, labels, interval, n_src)
        loss = ce_loss + aux

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        return {'loss_total': float(loss.detach()),
                'loss_ce': float(ce_loss.detach()),
                'loss_aux': float(aux.detach())}

    # ------------------------------------------------------------------
    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        self.train()

        res = {'loss_total': [], 'loss_ce': [], 'loss_aux': [], 'acces': []}
        for step in range(1, self.steps + 1):
            self.train()
            self.logger.info("================Step {}================".format(step))
            losses = self.update(next(train_minibatches_iterator))
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(step)

            for k in ('loss_total', 'loss_ce', 'loss_aux'):
                res[k].append(losses[k])
            self.logger.info(
                'loss_total: {:.4f} | ce: {:.4f} | aux: {:.4f}'.format(
                    losses['loss_total'], losses['loss_ce'], losses['loss_aux']))

            if step % self.checkpoint_freq == 0 or step == self.steps:
                accs = self.test_model(test_loaders)
                res['acces'].append(accs)
                self.logger.info('tgt_acc: {:.4f}'.format(accs[0]))
        return res

    def test_model(self, loaders):
        self.eval()
        accs = []
        for loader in loaders:
            y_pred, y_true = [], []
            for x, y in loader:
                x = x.to(self.device)
                y_pred.extend(self.predict(x).detach().cpu().numpy())
                y_true.extend(y.numpy())
            acc_i, _, _, _ = cal_index(y_true, y_pred)
            accs.append(acc_i)
        self.train()
        return accs

    def predict(self, x):
        with torch.no_grad():
            return torch.max(self.Gc(self.Gf(x)), dim=1)[1]


# ---------------------------------------------------------------------------
# 1. AGG —— 纯 ERM（无任何域泛化模块）
# ---------------------------------------------------------------------------
class AGG(_OrigeightBase):
    """朴素基线：只最小化源域交叉熵（对应 utils/train.py 的 else 分支）。"""


# ---------------------------------------------------------------------------
# 2. MMD —— 源域间多核 MMD 分布对齐
# ---------------------------------------------------------------------------
class MMD_(_OrigeightBase):
    """CE + trade_off * mean_{i<j} MMD(f_i, f_j)。"""

    def extra_loss(self, features, logits, labels, interval, n_src):
        fs = list(torch.split(features, interval, dim=0))
        pairs = [(i, j) for i in range(n_src) for j in range(i + 1, n_src)]
        if not pairs:
            return torch.zeros((), device=self.device, dtype=logits.dtype)
        total = sum(MMD(fs[i], fs[j]) for i, j in pairs) / len(pairs)
        return self.trade_off * total


# ---------------------------------------------------------------------------
# 3. CORAL —— 源域间二阶统计（协方差）对齐
# ---------------------------------------------------------------------------
class CORAL_(_OrigeightBase):
    """CE + trade_off * mean_{i<j} CORAL(f_i, f_j)。"""

    def extra_loss(self, features, logits, labels, interval, n_src):
        fs = list(torch.split(features, interval, dim=0))
        pairs = [(i, j) for i in range(n_src) for j in range(i + 1, n_src)]
        if not pairs:
            return torch.zeros((), device=self.device, dtype=logits.dtype)
        total = sum(CORAL(fs[i], fs[j]) for i, j in pairs) / len(pairs)
        return self.trade_off * total


# ---------------------------------------------------------------------------
# 4. Triplet_loss —— 难样本三元组度量学习
# ---------------------------------------------------------------------------
class Triplet_(_OrigeightBase):
    """CE + trade_off * TripletLoss(margin=2)(features, labels)。

    margin=2 是原 utils/train.py 的硬编码值（非 models/loss.py 的默认 0.3）。
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.triplet = TripletLoss(margin=2)

    def extra_loss(self, features, logits, labels, interval, n_src):
        return self.trade_off * self.triplet(features, labels)


# ---------------------------------------------------------------------------
# 5. Mixup —— 跨源域插值 + 按 λ 加权分类
# ---------------------------------------------------------------------------
class Mixup_(_OrigeightBase):
    """CE + [λ1·CE(aug, y1) + λ2·CE(aug, y2) + λ3·CE(aug, y3)]。

    原实现用 models/Mixup.py::mixup_augmentation（Dirichlet(1,1,1) 跨 3 域插值），
    这里保留该函数；仅在 n_src != 3 时回落为等价的 Dirichlet(1,...,1) 泛化。
    """

    def update(self, minibatches):
        x, labels, interval, _, _ = self._split(minibatches)
        n_src = len(minibatches)
        ys = [y.to(self.device).long() for _, y in minibatches]

        if n_src == 3:
            aug_x, lam = mixup_augmentation(x, interval, self.device)
            lam = lam[0]
        else:
            lam = torch.as_tensor(np.random.dirichlet(alpha=(1.0,) * n_src),
                                  device=self.device, dtype=x.dtype)
            aug_x = sum(lam[i] * x.narrow(0, i * interval, interval)
                        for i in range(n_src))

        features = self.Gf(x)
        logits = self.Gc(features)
        ce_loss = F.cross_entropy(logits, labels)

        aug_logits = self.Gc(self.Gf(aug_x.to(x.dtype)))
        aug_loss = sum(lam[i] * F.cross_entropy(aug_logits, ys[i])
                       for i in range(n_src))
        loss = ce_loss + aug_loss

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        return {'loss_total': float(loss.detach()),
                'loss_ce': float(ce_loss.detach()),
                'loss_aux': float(aug_loss.detach())}


# ---------------------------------------------------------------------------
# 6. MLDG —— 元学习（内环 meta-train / 外环 meta-test）
# ---------------------------------------------------------------------------
class MLDG_(_OrigeightBase):
    """复刻 utils/MLDG_train.py：随机留 1 域作 meta-test，其余为 meta-train。

    每步：
      1. deepcopy 出 inner 模型，在 meta-train 上走一步（inner_lr）；
      2. 把 inner 的 meta-train 梯度累加到主模型 `grad += q.grad`；
      3. 用 inner 模型在 meta-test 上算 loss，反传后累加 `grad += β·q.grad`；
      4. 主优化器 step。

    超参：inner_lr 取与主 lr 相同；β 取标准 MLDG 的 1.0（原代码为 args.trade_off，
    但那是元目标权重，与其余方法对齐损失用的 trade_off 语义不同，故不复用 0.01）。
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.inner_lr = float(getattr(configs, 'inner_lr', self.lr))
        self.beta = float(getattr(configs, 'meta_beta', 1.0))

    def update(self, minibatches):
        x, labels, interval, _, _ = self._split(minibatches)
        n_src = len(minibatches)
        xs = [xi.to(self.device) for xi, _ in minibatches]
        ys = [yi.to(self.device).long() for _, yi in minibatches]

        idx = list(range(n_src))
        meta_test = random.choice(idx)
        meta_train = [i for i in idx if i != meta_test]
        tr_x = torch.cat([xs[i] for i in meta_train])
        tr_y = torch.cat([ys[i] for i in meta_train])
        te_x, te_y = xs[meta_test], ys[meta_test]

        inner_Gf = copy.deepcopy(self.Gf)
        inner_Gc = copy.deepcopy(self.Gc)
        inner_opt = torch.optim.Adam(
            list(inner_Gf.parameters()) + list(inner_Gc.parameters()),
            lr=self.inner_lr, weight_decay=self.weight_decay)

        # --- 内环：meta-train 上走一步 ---
        meta_train_loss = F.cross_entropy(inner_Gc(inner_Gf(tr_x)), tr_y)
        inner_opt.zero_grad()
        meta_train_loss.backward()
        inner_opt.step()

        def acc(main_mod, inner_mod, scale):
            """把 inner 模型的梯度按 scale 累加到主模型（None 安全）。"""
            for p, q in zip(main_mod.parameters(), inner_mod.parameters()):
                if q.grad is None:
                    continue
                g = q.grad.detach() * scale
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.add_(g)

        self.optimizer.zero_grad()
        acc(self.Gf, inner_Gf, 1.0)            # 内环 meta-train 梯度
        acc(self.Gc, inner_Gc, 1.0)

        # --- 外环：inner 模型在 meta-test 上评估并反传 ---
        meta_test_loss = F.cross_entropy(inner_Gc(inner_Gf(te_x)), te_y)
        inner_opt.zero_grad()
        meta_test_loss.backward()
        acc(self.Gf, inner_Gf, self.beta)      # 外环 meta-test 梯度 ×β
        acc(self.Gc, inner_Gc, self.beta)

        # BN running stats 刷新（原实现的一步前向）
        self.Gf.train()
        _ = self.Gf(tr_x)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        loss = meta_train_loss.detach() + meta_test_loss.detach()
        return {'loss_total': float(loss),
                'loss_ce': float(meta_train_loss.detach()),
                'loss_aux': float(meta_test_loss.detach())}


# ---------------------------------------------------------------------------
# 7. DAEL —— 每源域一个分类器 + 跨域集成一致性
# ---------------------------------------------------------------------------
class DAEL(_OrigeightBase):
    """复刻 utils/DAEL_train.py：n_src 个分类器共享同一骨干。

    loss = mean_k CE(Gc_k(f_k), y_k)
           + trade_off * mean_k MSE( mean_{j≠k} Gc_j(f_k), Gc_k(f_k) )

    即「每个域的分类器对该域特征的预测」要与「其它域分类器对该域特征的预测均值」
    一致。评估时用 n_src 个分类器的平均 logits（原实现即如此）。
    训练全程只用源域；目标域仅在 test_model 中出现（DG 协议合规）。
    """

    def __init__(self, configs):
        super().__init__(configs)
        n = max(self.n_src, 2)
        self.heads = nn.ModuleList(
            [Classifier(configs.num_classes) for _ in range(n)]).to(self.device)
        self.mse = nn.MSELoss()
        # 重建优化器以纳入全部分类器
        self.optimizer = torch.optim.Adam(
            list(self.Gf.parameters()) + list(self.heads.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)
        self.n_heads = n

    def _fused_logits(self, features):
        return torch.stack([h(features) for h in self.heads], dim=0).mean(dim=0)

    def update(self, minibatches):
        x, labels, interval, _, _ = self._split(minibatches)
        n_src = len(minibatches)

        feats = list(torch.split(self.Gf(x), interval, dim=0))
        ys = list(torch.split(labels, interval, dim=0))

        logits_k, ce = [], 0.0
        for k in range(n_src):
            lg = self.heads[k](feats[k])
            logits_k.append(lg)
            ce = ce + F.cross_entropy(lg, ys[k])
        ce = ce / n_src

        # 跨域集成一致性：mean_{j≠k} Gc_j(f_k)  <->  Gc_k(f_k)
        el = 0.0
        for k in range(n_src):
            others = [self.heads[j](feats[k]).unsqueeze(1)
                      for j in range(n_src) if j != k]
            if not others:
                continue
            el = el + self.mse(torch.cat(others, dim=1).mean(dim=1), logits_k[k])
        el = el / n_src

        loss = ce + self.trade_off * el
        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        return {'loss_total': float(loss.detach()),
                'loss_ce': float(ce.detach()),
                'loss_aux': float(el.detach())}

    def predict(self, x):
        with torch.no_grad():
            return torch.max(self._fused_logits(self.Gf(x)), dim=1)[1]


# 方法名 -> 类名（供 run_cell.METHOD_MODELS 使用）
CLASS_NAMES = {
    'AGG': 'AGG',
    'MMD': 'MMD_',
    'CORAL': 'CORAL_',
    'Triplet_loss': 'Triplet_',
    'Mixup': 'Mixup_',
    'MLDG': 'MLDG_',
    'DAEL': 'DAEL',
}


def main(idx, configs):
    raise NotImplementedError('origeight 家族由 run_dgfdb.py 驱动，不单独 main 运行')
