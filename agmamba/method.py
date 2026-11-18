#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
AG-Mamba 方法模块（对齐 run_cell.py 的方法接口）。

交付（干净版）训练目标：L = CE(分类) + w_delta * ||Delta||_2^2
  · 无域对齐头：w_mmd = w_adv = 0 -> 域判别器不构建、MMD/JMMD/CDAN 不计算
    （消融证据见 docs/manuscript/_s64_align_data.md：last10 口径下
     去掉对齐项 +8.63 点，p = 0.0023，13/15 胜）；
  · 无液体项：tau_mode = 'zero' -> τ₀ 估计器与 γ 调制整条不构建。
前身（带对齐）目标：CE + w_mmd*MMD + w_adv*L_adv + w_delta*||Delta||^2
  可通过配置 w_mmd = 0.5 / w_adv = 0.1 复现（臂 xg2_std / xg2_6x）。

用法（由 run_dgfdb.py 驱动）:
  python run_dgfdb.py --method AGMamba --dataset PHM2009 --task T0 --seed 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import numpy as np
import yaml
import os
import sys

from utils.DictObj import DictObj
from utils.CalIndex import cal_index
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset

from models.agmamba_backbone import (
    LiquidMambaDGNet, DomainDiscriminator,
)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'configs.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = DictObj(configs)
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'


# ---------------------------------------------------------------------------
# MMD（多核 RBF，源域间两两对齐）
# ---------------------------------------------------------------------------
def rbf_kernel(x, y, sigma):
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True)
    d = xx + yy.T - 2.0 * x @ y.T
    return (-d / (2.0 * sigma * sigma)).exp()


def mmd_rbf(a, b, sigmas=(2.0,)):
    total = 0.0
    for sigma in sigmas:
        k_aa = rbf_kernel(a, a, sigma).mean()
        k_bb = rbf_kernel(b, b, sigma).mean()
        k_ab = rbf_kernel(a, b, sigma).mean()
        total = total + k_aa + k_bb - 2.0 * k_ab
    return total / len(sigmas)


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5,
                    fix_sigma=None):
    """多核高斯核（JMMD 用，来自 JAN/WIDAN）。"""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


def jmmd_loss(source_list, target_list, kernel_muls=(2.0, 2.0),
              kernel_nums=(5, 1), fix_sigma_list=(None, 1.68)):
    """JMMD 联合最大均值差异（特征层 x softmax 层联合核）。

    逐层核相乘得到联合核，等价于在联合分布上做对齐 —— 比只对齐边缘
    分布更强，隐式实现类别条件对齐（JAN/WIDAN 的做法）。
    source_list/target_list: [特征, softmax] 两层的张量。
    """
    batch_size = int(source_list[0].size()[0])
    joint_kernels = None
    for i in range(len(source_list)):
        kernels = guassian_kernel(source_list[i], target_list[i],
                                  kernel_mul=kernel_muls[i],
                                  kernel_num=kernel_nums[i],
                                  fix_sigma=fix_sigma_list[i])
        joint_kernels = kernels if joint_kernels is None else (
            joint_kernels * kernels)
    XX = joint_kernels[:batch_size, :batch_size]
    YY = joint_kernels[batch_size:, batch_size:]
    XY = joint_kernels[:batch_size, batch_size:]
    YX = joint_kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)


def cond_mmd_rbf(zs, ss, zt, st, sigma=2.0):
    """类别条件 MMD（CDAN 的软分配版本）。

    用 softmax 概率作为软类别权重，在每个类别的**子分布**上分别做 MMD，
    对齐的是类条件分布 P(z|y) 而非边缘分布 P(z)——比边缘对齐更强，
    可缓解"不同类别被错误对齐"的问题。

    zs/zt: 源/目标域特征 (B, D)；ss/st: 对应 softmax 概率 (B, C)。
    """
    k_ss = rbf_kernel(zs, zs, sigma)
    k_tt = rbf_kernel(zt, zt, sigma)
    k_st = rbf_kernel(zs, zt, sigma)
    n_cls = ss.shape[1]
    total = 0.0
    for c in range(n_cls):
        ws = ss[:, c]
        wt = st[:, c]
        ws = ws / (ws.sum() + 1e-8)
        wt = wt / (wt.sum() + 1e-8)
        total = total + (ws @ k_ss @ ws - 2.0 * ws @ k_st @ wt
                         + wt @ k_tt @ wt)
    return total / n_cls


def _time_scale(x, alpha):
    """按给定尺度因子做时间尺度缩放（可导）。

    alpha: 标量或 (B,) 张量。重采样源索引 src = center + (i-center)/alpha，
    alpha > 1 -> 拉伸（转速降低），alpha < 1 -> 压缩（转速升高）。
    """
    B, C, L = x.shape
    if not torch.is_tensor(alpha):
        alpha = torch.full((B,), float(alpha), device=x.device, dtype=x.dtype)
    elif alpha.dim() == 0:
        alpha = alpha.expand(B)
    idx = torch.arange(L, device=x.device, dtype=x.dtype)
    center = (L - 1) / 2.0
    src = center + (idx - center) / alpha.view(B, 1)
    src_norm = src / (L - 1) * 2.0 - 1.0
    gy = torch.zeros_like(src_norm)
    grid = torch.stack([src_norm, gy], dim=-1).view(B, 1, L, 2)
    out = F.grid_sample(x.view(B, C, 1, L), grid, mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out.view(B, C, L)


def time_scale_jitter(x, delta=0.1):
    """per-sample 随机时间尺度抖动增强（可导）。

    x: (B, 1, L) -> (B, 1, L)
    """
    if delta <= 0:
        return x
    B = x.shape[0]
    alpha = 1.0 + delta * (2.0 * torch.rand(B, device=x.device) - 1.0)
    return _time_scale(x, alpha)


def domain_time_scale_jitter(x, batch_size=64, num_domains=3, delta=0.1):
    """域级时间尺度增强（推荐）。

    DG 动机：跨转速的尺度偏移在测试时是**整个目标域一致**的，而非类内
    随机噪声。因此对每个源域整体施加一个随机但固定的尺度因子
    alpha_d ~ U[1-delta, 1+delta]，模拟"该域转速偏高/偏低"。

    x: (num_domains * batch_size, 1, L)，按域连续排列。
    """
    if delta <= 0:
        return x
    alphas = 1.0 + delta * (2.0 * torch.rand(num_domains,
                                             device=x.device) - 1.0)
    out = torch.empty_like(x)
    n = x.shape[0]
    for d in range(num_domains):
        s = d * batch_size
        if s >= n:
            break
        e = min(s + batch_size, n)
        out[s:e] = _time_scale(x[s:e], alphas[d])
    return out


class AGMamba(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        # 当前步序号（train_model 每步更新；此处兜底以便单独调用 update 时不报错）
        self.current_step = 0

        self.w_mmd = float(getattr(configs, 'w_mmd', 0.5))
        self.w_adv = float(getattr(configs, 'w_adv', 0.1))
        self.w_delta = float(getattr(configs, 'w_delta', 0.01))
        # 时间尺度增强强度（0 = 关闭）；物理上模拟跨转速域偏移
        # ts_jitter_mode: 'sample' = 逐样本抖动；'domain' = 逐源域一致偏移
        self.ts_jitter = float(getattr(configs, 'ts_jitter', 0.0))
        self.ts_jitter_mode = str(getattr(configs, 'ts_jitter_mode', 'sample'))
        # JMMD 联合核对齐权重（特征层 x softmax 层；0 = 关闭）
        self.w_jmmd = float(getattr(configs, 'w_jmmd', 0.0))
        # RSC（Representation Self-Challenging）：丢弃贡献最大的特征比例
        self.rsc_p = float(getattr(configs, 'rsc_p', 0.0))
        # Mixup：域内样本插值增强（0 = 关闭）
        self.mixup_alpha = float(getattr(configs, 'mixup_alpha', 0.0))
        # 类别条件 MMD（CDAN）权重；0 = 关闭
        self.w_cdan = float(getattr(configs, 'w_cdan', 0.0))
        # IMPACT 影响加权（域级简化）：损失小的源域权重高，抑制负迁移域
        self.use_impact = bool(getattr(configs, 'use_impact', False))
        self.impact_temp = float(getattr(configs, 'impact_temp', 1.0))

        self.model = LiquidMambaDGNet(
            num_classes=configs.num_classes,
            d_model=int(getattr(configs, 'd_model', 128)),
            d_inner=int(getattr(configs, 'd_inner', 128)),
            n_blocks=int(getattr(configs, 'n_blocks', 3)),
            n_state=int(getattr(configs, 'n_state', 8)),
            stride=int(getattr(configs, 'front_stride', 4)),
            front_chan=int(getattr(configs, 'front_chan', 128)),
            kernel_size=int(getattr(configs, 'front_ksize', 64)),
            wave_init=bool(getattr(configs, 'wave_init', False)),
            detach_tau=bool(getattr(configs, 'detach_tau', False)),
            gamma_init=float(getattr(configs, 'gamma_init', 0.1)),
            a_lo=float(getattr(configs, 'a_lo', 0.5)),
            a_hi=float(getattr(configs, 'a_hi', 3.0)),
            tau_min_lag=int(getattr(configs, 'tau_min_lag', 8)),
            tau_max_lag=int(getattr(configs, 'tau_max_lag', 300)),
            tau_temp=float(getattr(configs, 'tau_temp', 0.05)),
            tau_f_lo=int(getattr(configs, 'tau_f_lo', 3)),
            tau_f_hi=int(getattr(configs, 'tau_f_hi', 120)),
            use_pos_embed=bool(getattr(configs, 'use_pos_embed', True)),
            head_dropout=float(getattr(configs, 'head_dropout', 0.3)),
            tau_estimator=str(getattr(configs, 'tau_estimator', 'autocorr')),
            tau_mode=str(getattr(configs, 'tau_mode', 'normal')),
            tau_proj=bool(getattr(configs, 'tau_proj', False)),
            ssm_mode=str(getattr(configs, 'ssm_mode', 'single')),
            ms_d_inner=int(getattr(configs, 'ms_d_inner', 64)),
            front_mode=str(getattr(configs, 'front_mode', 'cnn')),
            front_norm=str(getattr(configs, 'front_norm', 'bn')),
            front_init=str(getattr(configs, 'front_init', 'none')),
            gabor_sigma=float(getattr(configs, 'gabor_sigma', 0.5)),
            gabor_f_lo=float(getattr(configs, 'gabor_f_lo', 0.005)),
            gabor_f_hi=float(getattr(configs, 'gabor_f_hi', 0.45)),
            gabor_dc_frac=float(getattr(configs, 'gabor_dc_frac', 0.5)),
            gabor_alias=float(getattr(configs, 'gabor_alias', 0.0)),
            freeze_init=bool(getattr(configs, 'freeze_init', False)),
            film=bool(getattr(configs, 'film', False)),
            use_ssm=bool(getattr(configs, 'use_ssm', True)),
        ).to(self.device)
        # 直接采用模型实际输出的特征维度（避免与 use_ssm / front_mode 等
        # 开关不一致导致判别器维度错配）
        feat_dim = int(self.model.feat_dim)
        # 干净版：w_adv = 0 时完全不构建域判别器（不占参数、不进优化器、
        # 前向不做 GRL）；只有明确要求对抗对齐时才创建。
        self.disc = None
        if self.w_adv > 0:
            self.disc = DomainDiscriminator(
                d_feat=feat_dim,
                n_domains=int(getattr(configs, 'n_domains', 3)),
                grl_lambda=1.0).to(self.device)

        params = list(self.model.parameters())
        if self.disc is not None:
            params += list(self.disc.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=1e-4)

        # LR 调度：'cos' | 'step' | 'none'。留空时按旧开关 use_scheduler 推断
        # （True -> 'cos'，保持交付配置行为不变）。
        #   'cos'  余弦退火至零：lr * 0.5*(1+cos(pi*step/steps))
        #   'step' 基准框架（DGFDBenchmark）的阶梯衰减语义：
        #          lr = base_lr * prod(0.5 if step >= m else 1.0 for m in schedule)
        #          —— 每到一个 milestone 乘 0.5，与 baselines/*.py 的 update_lr 一致
        #   'none' 恒定 lr
        self.schedule = [int(x) for x in (getattr(configs, 'schedule', None) or [])]
        self.sched_mul = float(getattr(configs, 'sched_mul', 0.5))
        self.sched_mode = str(getattr(configs, 'sched_mode', '') or '') or (
            'cos' if bool(getattr(configs, 'use_scheduler', False)) else 'none')
        self.scheduler = None
        if self.sched_mode == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.steps)
        elif self.sched_mode == 'step' and not self.schedule:
            # 未给 milestones 时按 1/3、2/3 预算切分（与基准框架一致）
            self.schedule = [self.steps // 3, 2 * self.steps // 3]

        # EMA 权重平均（近似零计算量开销，平滑 SGD 噪声、提升泛化）
        self.ema_decay = float(getattr(configs, 'ema_decay', 0.0))
        self.ema_start = int(getattr(configs, 'ema_start', 100))
        self.ema_model = None
        self._ema_initialized = False
        if self.ema_decay > 0:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_model.to(self.device)
            self.ema_model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_ema(self):
        """更新 EMA 影子权重（仅前向参数，不含 disc）。

        前 ema_start 步跳过（避免把随机初始化权重混入），第 ema_start
        步用当前权重初始化影子模型，之后按 decay 做指数滑动平均。
        """
        if self.current_step < self.ema_start:
            return
        if not self._ema_initialized:
            self.ema_model.load_state_dict(self.model.state_dict())
            self._ema_initialized = True
            return
        d = self.ema_decay
        for pe, pm in zip(self.ema_model.parameters(),
                          self.model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.ema_model.buffers(), self.model.buffers()):
            be.copy_(bm)

    def _eval_net(self):
        """推理所用网络：启用 EMA 时用影子权重，否则用当前权重。"""
        if self.ema_model is not None:
            return self.ema_model
        return self.model

    # ------------------------------------------------------------------
    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device).long()

        # 时间尺度增强（仅训练时；τ0 从增强后信号估计，自动保持一致）
        if self.ts_jitter > 0:
            if self.ts_jitter_mode == 'domain':
                x = domain_time_scale_jitter(x, self.batch_size, 3,
                                             self.ts_jitter)
            else:
                x = time_scale_jitter(x, self.ts_jitter)

        # Mixup：域内样本插值（避免跨域混合造成语义混乱）
        labels_perm = None
        mixup_lam = 1.0
        if self.mixup_alpha > 0:
            n_dom = len(minibatches)
            B_ = self.batch_size
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            idx = torch.cat([torch.randperm(B_, device=self.device) + d * B_
                             for d in range(n_dom)])
            x = lam * x + (1.0 - lam) * x[idx]
            labels_perm = labels[idx]
            mixup_lam = lam

        # 源域索引（仅在启用对抗对齐时需要）
        domains = None
        if self.disc is not None:
            doms = []
            for i, (_, y) in enumerate(minibatches):
                doms.append(torch.full((len(y),), i, dtype=torch.long,
                                       device=self.device))
            domains = torch.cat(doms)

        logits, z, delta_norm, tau0 = self.model(x)

        # RSC（Representation Self-Challenging）：丢弃对损失贡献最大的特征维度，
        # 迫使模型依赖多样化特征而非少数主导通道
        if self.rsc_p > 0:
            z_r = z.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(
                F.cross_entropy(self.model.head(z_r), labels), z_r)[0]
            imp = g.abs().mean(dim=0)
            k = max(1, int(imp.numel() * self.rsc_p))
            drop = torch.topk(imp, k).indices
            mask = torch.ones_like(z)
            mask[:, drop] = 0.0
            z = z * mask
            logits = self.model.head(z)

        # 源域间 pairwise MMD（DG 设置下目标域不可见，只在源域间对齐）
        B = self.batch_size
        logits_s = torch.split(logits, B, dim=0)
        labels_s = torch.split(labels, B, dim=0)

        # IMPACT 影响加权（域级）：softmax(-CE/T) 对"迁移损失小"的源域加权
        if self.use_impact:
            ce_d = torch.stack(
                [F.cross_entropy(lg, lb) for lg, lb in zip(logits_s, labels_s)])
            w = torch.softmax(-ce_d.detach() / self.impact_temp, dim=0)
            w = w * w.numel()  # 归一化到均值为 1，保持 loss 量级
            ce_loss = (w * ce_d).mean()
        elif labels_perm is not None:
            ce_loss = (mixup_lam * F.cross_entropy(logits, labels)
                       + (1.0 - mixup_lam) * F.cross_entropy(logits,
                                                             labels_perm))
        else:
            ce_loss = F.cross_entropy(logits, labels)

        # ---- 对齐项（干净版：三个权重全为 0 时整段不做任何计算）----
        # 交付配置 w_mmd = w_jmmd = w_cdan = 0 -> 连 z 的域切分都不发生，
        # 不存在"算了再乘 0"的无用前向/显存开销。
        n_dom = len(minibatches)
        pairs = [(i, j) for i in range(n_dom) for j in range(i + 1, n_dom)]
        need_align = (self.w_mmd > 0 or self.w_jmmd > 0 or self.w_cdan > 0)
        zs = torch.split(z, B, dim=0) if need_align else None
        mmd_loss = torch.zeros((), device=x.device, dtype=z.dtype)
        jmmd = torch.zeros((), device=x.device, dtype=logits.dtype)
        cdan = torch.zeros((), device=x.device, dtype=logits.dtype)
        if need_align and pairs:
            if self.w_mmd > 0:
                mmd_loss = sum(mmd_rbf(zs[i], zs[j])
                               for i, j in pairs) / len(pairs)
            # JMMD（联合核）+ CDAN（类别条件 MMD）：共用 pairwise 域对
            sm_s = (torch.split(F.softmax(logits, dim=1), B, dim=0)
                    if (self.w_jmmd > 0 or self.w_cdan > 0) else None)
            if self.w_jmmd > 0:
                for i, j in pairs:
                    jmmd = jmmd + jmmd_loss([zs[i], sm_s[i]],
                                            [zs[j], sm_s[j]])
                jmmd = jmmd / len(pairs)
            if self.w_cdan > 0:
                for i, j in pairs:
                    cdan = cdan + cond_mmd_rbf(zs[i], sm_s[i],
                                               zs[j], sm_s[j])
                cdan = cdan / len(pairs)

        adv_loss = torch.zeros((), device=x.device, dtype=logits.dtype)
        if self.disc is not None:
            adv_loss = F.cross_entropy(self.disc(z), domains)

        loss = (ce_loss + self.w_mmd * mmd_loss + self.w_jmmd * jmmd
                + self.w_cdan * cdan
                + self.w_adv * adv_loss + self.w_delta * delta_norm)

        assert not torch.any(torch.isnan(loss)), 'NaN loss detected'

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止 SSM 反传尖峰
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        elif self.sched_mode == 'step':
            # 基准框架语义：每到一个 milestone 把 lr 乘 sched_mul（默认 0.5）
            lr = self.lr
            for m in self.schedule:
                if self.current_step >= m:
                    lr *= self.sched_mul
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        if self.ema_model is not None:
            self._update_ema()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_mmd'] = mmd_loss.detach().cpu().data.numpy()
        losses['loss_jmmd'] = jmmd.detach().cpu().data.numpy()
        losses['loss_cdan'] = cdan.detach().cpu().data.numpy()
        losses['loss_adv'] = adv_loss.detach().cpu().data.numpy()
        losses['loss_delta'] = delta_norm.detach().cpu().data.numpy()
        losses['tau0'] = tau0.detach().cpu().mean().data.numpy()
        # E3: 记录液体时间常数灵敏度 gamma（干净版下 gamma 不存在 -> 0）
        gs = [float(b.gamma.detach().cpu()) for b in self.model.blocks
              if getattr(b, 'gamma', None) is not None]
        losses['gamma'] = float(np.mean(gs)) if gs else 0.0
        return losses

    # ------------------------------------------------------------------
    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        self.train()

        loss_acc_result = {'loss_total': [], 'loss_ce': [], 'loss_mmd': [],
                           'loss_jmmd': [], 'loss_adv': [], 'loss_delta': [],
                           'acces': []}

        for step in range(1, self.steps + 1):
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            for k in ['loss_total', 'loss_ce', 'loss_mmd', 'loss_jmmd',
                      'loss_adv', 'loss_delta']:
                loss_acc_result[k].append(losses[k])

            self.logger.info(
                'loss_total: {:.4f} | ce: {:.4f} | mmd: {:.4f} | jmmd: {:.4f} | '
                'adv: {:.4f} | delta: {:.4f} | tau0: {:.4f} | gamma: {:.5f} | '
                'lr: {:.6f}'.format(
                    float(losses['loss_total']), float(losses['loss_ce']),
                    float(losses['loss_mmd']), float(losses['loss_jmmd']),
                    float(losses['loss_adv']), float(losses['loss_delta']),
                    float(losses['tau0']), float(losses['gamma']),
                    float(self.optimizer.param_groups[0]['lr'])))

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
            logits, _, _, _ = self._eval_net()(x)
            return torch.max(logits, dim=1)[1]


def main(idx, configs):
    raise NotImplementedError('AGMamba 由 run_dgfdb.py 驱动，不单独 main 运行')
