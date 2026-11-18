#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
AG-Mamba: Aliased-Gabor Mamba Network.

物理小波前端 (Laplace 初始化) + 液体时间常数选择性 SSM (Mamba-style) +
因果卷积 + 源域间域对齐特征。

本文件完全独立，不依赖、也不修改项目内任何现有模型文件。
全部计算在 fp32 下进行，SSM 扫描优先使用官方 mamba_ssm CUDA 内核
（channel-first 语义），CPU/导入失败时回落纯 PyTorch 分块扫描。

设计要点（见 docs/legacy/design_proposal_v0.md）：
- 跨转速域偏移 = 时间尺度变化 x(t) -> x(a*t)；
- 时间缩放等价于状态方程时间常数缩放 a 倍；
- Mamba 的输入依赖离散化步长 Delta 本质上是一种"液体时间常数"，
  本实现显式加入输入瞬时周期估计 tau0(x) 的调制项 gamma * tau0。
"""
import math

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

# 官方 mamba_ssm CUDA 选择性扫描内核（环境自带 mamba_ssm 2.2.4）；
# CPU 或导入失败时自动回落纯 Python 分块扫描。
#
# ⚠ 确定性：官方内核的**反传含原子累加**，同一输入同一权重两次前反向的梯度
#   并不逐位相同（实测 max|Δgrad| ≈ 4e-8），经上千步训练会放大成 ~1 点精度漂移。
#   设环境变量 AGM_FORCE_PY_SCAN=1 可强制使用纯 PyTorch 分块扫描（完全确定，较慢）。
#   见 docs/COMPARE_PROTOCOL.md §2 的"可复现性"一节。
try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _OFFICIAL_SCAN)
except Exception:  # noqa: BLE001
    _OFFICIAL_SCAN = None

if os.environ.get('AGM_FORCE_PY_SCAN', '') == '1':
    _OFFICIAL_SCAN = None

__all__ = ['LiquidMambaDGNet', 'DomainDiscriminator', 'GradientReversal',
           'envelope_autocorr_period', 'envelope_spectrum_period',
           'hilbert_envelope', 'gabor_bank', 'GaborBankFrontend']


# ---------------------------------------------------------------------------
# 梯度反转层（域判别器用）
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return -ctx.lambda_ * grads, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# 信号处理（可导）
# ---------------------------------------------------------------------------
def hilbert_envelope(x):
    """解析信号包络，x: (B, L)"""
    B, L = x.shape
    X = torch.fft.rfft(x, dim=-1)
    N = X.shape[-1]
    Z = torch.zeros_like(X)
    Z[:, 0] = X[:, 0]
    if N > 2:
        Z[:, 1:N - 1] = 2.0 * X[:, 1:N - 1]
    Z[:, N - 1] = X[:, N - 1]
    analytic = torch.fft.irfft(Z, n=L, dim=-1)
    return torch.abs(analytic)


def envelope_autocorr_period(x, min_lag=8, max_lag=300, temp=0.05):
    """包络自相关 + soft-argmax 估计瞬时冲击周期。

    x: (B, 1, L) -> tau0: (B, 1, 1)，归一化周期 ∈ (min_lag/L, max_lag/L)。
    转速降低 -> 周期变长 -> tau0 变大，液体时间常数相应增大。
    """
    B, _, L = x.shape
    x = x[:, 0]  # (B, L)
    env = hilbert_envelope(x)
    env = env - env.mean(dim=-1, keepdim=True)
    ac = torch.fft.irfft(torch.abs(torch.fft.rfft(env, dim=-1)) ** 2,
                         n=L, dim=-1)  # (B, L) 自相关
    seg = ac[:, min_lag:max_lag]
    seg = seg - seg.mean(dim=-1, keepdim=True)
    std = seg.std(dim=-1, keepdim=True).clamp_min(1e-3)
    w = torch.softmax(seg / (temp * std), dim=-1)
    lags = torch.arange(min_lag, max_lag, device=x.device, dtype=x.dtype)
    tau = (w * lags).sum(dim=-1, keepdim=True)  # (B, 1)
    return (tau / float(L)).unsqueeze(-1)  # (B, 1, 1)


def envelope_spectrum_period(x, f_lo=3, f_hi=120, temp=0.05):
    """包络谱主峰频率倒数估计冲击周期（改进版 τ0 估计器）。

    包络谱在故障特征频率处有清晰谱峰，比自相关更抗谐波混淆。
    x: (B,1,L) -> tau0: (B,1,1)。f_lo..f_hi 为 FFT bin 索引
    （对应周期 L/f ∈ [L/f_hi, L/f_lo] 样本）。
    """
    B, _, L = x.shape
    x = x[:, 0]  # (B, L)
    env = hilbert_envelope(x)
    env = env - env.mean(dim=-1, keepdim=True)
    spec = torch.abs(torch.fft.rfft(env, dim=-1))  # (B, N)
    f_hi = min(f_hi, spec.shape[-1] - 1)
    seg = spec[:, f_lo:f_hi]
    seg = seg - seg.mean(dim=-1, keepdim=True)
    std = seg.std(dim=-1, keepdim=True).clamp_min(1e-3)
    w = torch.softmax(seg / (temp * std), dim=-1)
    freqs = torch.arange(f_lo, f_hi, device=x.device, dtype=x.dtype)
    f_peak = (w * freqs).sum(dim=-1, keepdim=True).clamp_min(1e-3)  # (B,1)
    tau = float(L) / f_peak  # 样本周期 (B,1)
    return (tau / float(L)).unsqueeze(-1)  # (B,1,1) 归一化


# ---------------------------------------------------------------------------
# 物理小波前端（WIDAN 思想：Laplace 小波核初始化 + 可学习微调）
# ---------------------------------------------------------------------------
def laplace_wavelet_kernel(out_channels, kernel_size, f_lo=0.01, f_hi=0.08, zeta=0.08):
    """生成 Laplace 小波卷积核 (out_channels, 1, kernel_size)。

    psi(t) = exp(-2*pi*f*zeta*|t|) * sin(2*pi*f*t)，f 在 [f_lo, f_hi] 按通道线性分布。
    """
    t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size / 2.0
    freqs = torch.linspace(f_lo, f_hi, out_channels).view(-1, 1)
    omega = 2.0 * math.pi * freqs
    damp = torch.exp(-omega * zeta * t.abs().unsqueeze(0))
    psi = damp * torch.sin(omega * t.unsqueeze(0))
    psi = psi / (psi.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6))
    return psi.unsqueeze(1)  # (C, 1, K)


def gabor_kernel(out_channels, kernel_size, f_lo=0.01, f_hi=0.08,
                 sigma_ratio=0.5):
    """生成 Gabor 小波卷积核 (out_channels, 1, kernel_size)。

    psi(t) = exp(-t^2/(2*sigma^2)) * cos(2*pi*f*t)，f 在 [f_lo, f_hi] 按通道线性分布。
    Gabor 在 Heisenberg 不确定性意义下达到最优时频局部化，相比 Laplace 小波
    衰减为高斯型，对轴承冲击的瞬态特征响应更集中。

    sigma = sigma_ratio / f：时宽与中心频率成反比（恒定相对带宽）。
    核做去直流处理（减均值）避免对直流分量响应。
    """
    t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size / 2.0
    freqs = torch.linspace(f_lo, f_hi, out_channels).view(-1, 1)   # (C,1)
    sigma = sigma_ratio / freqs                                    # (C,1)
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    omega = 2.0 * math.pi * freqs
    psi = env * torch.cos(omega * t.unsqueeze(0))
    psi = psi - psi.mean(dim=-1, keepdim=True)   # 去直流
    psi = psi / (psi.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6))
    return psi.unsqueeze(1)  # (C, 1, K)


def gabor_bank(out_channels, kernel_size, f_lo=0.005, f_hi=0.45,
               sigma_ratio=0.6, dc_frac=0.5, alias=0.0,
               fm_lo=0.20, fm_hi=0.48):
    """宽频段 Gabor 滤波器组 (out_channels, 1, kernel_size)。

    设计要点（针对宽转速跨度域泛化）：
      1. 频率按**对数均匀**覆盖 [f_lo, f_hi]，横跨几乎整个可用频带。
      2. **混合直流通道**：低频 dc_frac 比例通道保留直流（携带单向冲击/包络信息），
         其余通道去直流（避免被接近常数的偏置主导）。
      3. **受控相位混叠**（alias>0）：在载波相位上叠加高频相位调制

             phi(t) = 2*pi*f*t + beta * sin(2*pi*fm*t)

         使瞬时频率在 f +- beta*fm 间摆动，把每通道的频谱"抹开"。
         实测 WIDAN 的 Laplace 核过零率 33.9、峰频 0.268（混叠），但其在
         内插与外推上**同时**最优 -> 它给的是「空间有结构、频谱弱偏好」的先验。
         alias 就是用来复现这一特性的旋钮（0=解析带通，越大越接近混叠）。
      4. 每通道 L2 归一化，保证能量一致。
    """
    t = (torch.arange(kernel_size, dtype=torch.float32)
         - (kernel_size - 1) / 2.0)
    freqs = torch.exp(torch.linspace(math.log(f_lo), math.log(f_hi),
                                     out_channels)).view(-1, 1)   # (C,1)
    sigma = sigma_ratio / freqs
    env = torch.exp(-(t.unsqueeze(0) ** 2) / (2.0 * sigma ** 2))
    phase = 2.0 * math.pi * freqs * t.unsqueeze(0)
    if alias > 0:
        fms = torch.exp(torch.linspace(math.log(fm_lo), math.log(fm_hi),
                                       out_channels)).view(-1, 1)
        phase = phase + float(alias) * torch.sin(
            2.0 * math.pi * fms * t.unsqueeze(0))
    psi = env * torch.cos(phase)

    n_keep_dc = int(round(out_channels * dc_frac))
    # 低频（前 n_keep_dc 个通道）保留直流：它们在一个窗内不满一个周期，
    # 直流分量本身携带"单向冲击/包络"信息（WIDAN 核实测 max|mean|=0.50）
    # 高频（其余通道）去直流，避免被接近常数的偏置主导
    if out_channels - n_keep_dc > 0:
        psi[n_keep_dc:] = (psi[n_keep_dc:]
                           - psi[n_keep_dc:].mean(dim=-1, keepdim=True))
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # L2 归一化
    return psi.unsqueeze(1)  # (C,1,K)


def laplace_bank(out_channels, kernel_size, frequency=64000.0, eps=0.1,
                 mode='sigmoid'):
    """WIDAN 的宽核 Laplace 小波族[10] —— §3.3 / Fig. 2 平面上的参照区域。

    与 `gabor_bank` 的关键区别：这里用的是**参考实现本身**
    （`models.widan_net.Laplace_fast`，`eps=0.1`、`mode='sigmoid'`、
    `frequency=64000`），逐字沿用（与 `WideResNetSeqFrontend` 的初始化一致），
    只把原子数与核长对齐到本文首层（64 原子 × 核长 `kernel_size`）。

    **不做 L2 归一化**，因为参考实现不归一化；ζ（过零率）与 Π（谱峰均比）
    都是尺度不变量，且首层卷积（`bias=False`）之后紧跟 BatchNorm，
    逐通道的正向尺度在 forward 中等价于被 BN 吸收。

    返回 (out_channels, 1, kernel_size)，可直接 `copy_` 进 conv1.weight。
    """
    from models.widan_net import Laplace_fast   # 局部导入：避免模块级循环依赖
    with torch.no_grad():
        bank = Laplace_fast(out_channels=out_channels, kernel_size=kernel_size,
                            eps=eps, frequency=frequency,
                            mode=mode).forward()
    return bank.float()


def _make_norm(num_channels, kind='bn'):
    """归一化层工厂：'bn' -> BatchNorm1d，'gn' -> GroupNorm。

    DG 文献（AdaBN / TransNorm）指出 BN 依赖源域统计量，跨域时
    统计偏移会造成负迁移；GroupNorm 不依赖 batch 统计，通常更鲁棒。
    """
    if str(kind).lower() == 'gn':
        g = 8
        while num_channels % g != 0:
            g //= 2
        return nn.GroupNorm(g, num_channels)
    return nn.BatchNorm1d(num_channels)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（1D），参数增量极小。"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)

    def forward(self, x):
        b, c, _ = x.shape
        s = x.mean(dim=-1)                      # 全局平均池化
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)
        return x * s


class WaveletFrontend(nn.Module):
    """强 CNN 前端（可选小波初始化第一层），保持时序维。

    输入 (B,1,L) -> (B,128,L')。结构贴近项目内 Feature_extractor
    （LReLU+BN+MaxPool），保证局部冲击特征的表示能力。
    """

    def __init__(self, kernel_size=64, stride=4, f_lo=0.01, f_hi=0.08,
                 wave_init=True, out_chan=128, se=False, norm='bn',
                 init_mode='none', gabor_sigma=0.5, freeze_init=False):
        super(WaveletFrontend, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size, stride=stride,
                               padding=0, bias=False)
        # 前端第一层核初始化：'none' | 'laplace' | 'gabor'
        mode = str(init_mode).lower()
        if mode == 'none' and wave_init:
            mode = 'laplace'
        if mode != 'none':
            if mode == 'gabor':
                kernel = gabor_kernel(16, kernel_size, f_lo, f_hi, gabor_sigma)
            else:
                kernel = laplace_wavelet_kernel(16, kernel_size, f_lo, f_hi,
                                                0.08)
            with torch.no_grad():
                self.conv1.weight.copy_(kernel)
            if freeze_init:
                self.conv1.weight.requires_grad_(False)
        self.bn1 = _make_norm(16, norm)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding=7, bias=False)
        self.bn2 = _make_norm(32, norm)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn3 = _make_norm(64, norm)

        self.conv4 = nn.Conv1d(64, out_chan, kernel_size=5, padding=2, bias=False)
        self.bn4 = _make_norm(out_chan, norm)
        self.se = SEBlock(out_chan) if se else None
        self.out_chan = out_chan

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.pool2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        if self.se is not None:
            x = self.se(x)
        return x


class GaborBankFrontend(nn.Module):
    """宽频段 Gabor 滤波器组 + 加宽纯 CNN 前端（约 1M 参数）。

    与 WaveletFrontend 的区别：
      · 第一层：宽频段（对数均匀 f∈[f_lo,f_hi]）+ 混合直流通道的 Gabor 滤波器组，
        而非窄带（[0.01,0.08]）零均值 Laplace 小波
      · 容量：层宽 64/128/256/512，约 0.96M 参数（vs 60K）
      · 骨干：普通 1D CNN（不使用 ResNet）

    输入 (B,1,1024) -> 输出 (B, out_chan, 128)，保留时序交给 SSM。
    """

    def __init__(self, kernel_size=64, f_lo=0.005, f_hi=0.45,
                 sigma_ratio=0.6, dc_frac=0.5, out_chan=512, norm='bn',
                 init='gabor', alias=0.0, fm_lo=0.20, fm_hi=0.48,
                 freeze_init=False):
        super(GaborBankFrontend, self).__init__()
        # 记录**实际生效**的 init 模式：起飞自检可直接从模型自身核验，
        # 而不是去看配置（配置值曾被上游白名单静默改写 —— 见 LiquidMambaDGNet）
        self.init_mode = str(init).lower()
        # 宽度按比例扩容：512 -> 64/128/256/512（与旧版逐位一致）；
        # 768 -> 96/192/384/768；1024 -> 128/256/512/1024
        c1, c2, c3, c4 = (out_chan // 8, out_chan // 4,
                          out_chan // 2, out_chan)

        self.conv1 = nn.Conv1d(1, c1, kernel_size, stride=2,
                               padding=kernel_size // 2, bias=False)
        mode = str(init).lower()
        if mode == 'gabor':
            with torch.no_grad():
                self.conv1.weight.copy_(gabor_bank(
                    c1, kernel_size, f_lo, f_hi, sigma_ratio, dc_frac,
                    alias, fm_lo, fm_hi))
        elif mode == 'laplace':
            # §3.3 / Fig. 2 平面上「宽核 Laplace 族」那一区的对照臂
            # （WIDAN 参考实现，不 L2 归一化；核长与本文首层一致）
            with torch.no_grad():
                self.conv1.weight.copy_(laplace_bank(c1, kernel_size))
        elif mode == 'mix':
            # 软化先验：一半通道 Gabor（物理先验），一半通道随机（灵活性）
            n_g = max(1, c1 // 2)
            with torch.no_grad():
                self.conv1.weight[:n_g].copy_(gabor_bank(
                    n_g, kernel_size, f_lo, f_hi, sigma_ratio, dc_frac,
                    alias, fm_lo, fm_hi))
            nn.init.kaiming_normal_(self.conv1.weight[n_g:], mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        # 冻结首层（§6.5 "冻结分析器"腿）：只关梯度，不改变前向
        if freeze_init:
            self.conv1.weight.requires_grad_(False)
        self.bn1 = _make_norm(c1, norm)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=9, padding=4, bias=False)
        self.bn2 = _make_norm(c2, norm)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=7, padding=3, bias=False)
        self.bn3 = _make_norm(c3, norm)
        self.conv4 = nn.Conv1d(c3, c4, kernel_size=5, padding=2, bias=False)
        self.bn4 = _make_norm(c4, norm)
        self.out_chan = c4

    def forward(self, x):
        # 1024 -> conv1(s2) 512 -> pool 256
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.max_pool1d(x, 2, 2)
        # 256 -> conv2 256 (k9 s1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        # 256 -> conv3 256 (k7 s1) -> pool 128
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.max_pool1d(x, 2, 2)
        # 128 -> conv4 128 (k5 s1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        return x


class DeepCNN6Frontend(nn.Module):
    """6 层加深 CNN 前端，保持时序分辨率 L' >= 128。

    输入 (B,1,1024) -> (B,out_chan,240)。相比 WaveletFrontend 的 60 步，
    时序分辨率提升 4 倍，给下游液体 SSM 更长的可扫描序列。
    conv1 用 stride=2（而非 4），pool 只做 2 倍下采样，后续 4 层全 padding 卷积。
    """

    def __init__(self, kernel_size=64, f_lo=0.01, f_hi=0.08,
                 wave_init=True, out_chan=128):
        super(DeepCNN6Frontend, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size, stride=2,
                               padding=0, bias=False)
        if wave_init:
            with torch.no_grad():
                self.conv1.weight.copy_(laplace_wavelet_kernel(
                    16, kernel_size, f_lo, f_hi, 0.08))
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False)
        self.bn5 = nn.BatchNorm1d(128)

        self.conv6 = nn.Conv1d(128, out_chan, kernel_size=5, padding=2, bias=False)
        self.bn6 = nn.BatchNorm1d(out_chan)
        self.out_chan = out_chan

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.pool2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        return x


class TCNBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积 + 残差连接（Bai et al. 2018）。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super(TCNBlock, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               dilation=dilation, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               dilation=dilation, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                           if in_ch != out_ch else None)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        L = x.shape[-1]
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)[..., :L]))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out)[..., :L])
        out = self.relu(out + res)
        return out


class TCNFrontend(nn.Module):
    """TCN（时序卷积网络）前端：膨胀因果卷积 + 残差，保持长时序分辨率。

    输入 (B,1,1024) -> (B,out_chan,240)。相比逐层小卷积的 CNN，
    膨胀卷积以指数方式扩大感受野，残差连接缓解加深带来的梯度问题。
    conv_in 只做 2 倍下采样（k=64, s=2），4 个 TCN 块（d=1/2/4/8）
    保持长度不变，最后 2 倍池化，总时序分辨率 240（>=128）。
    """

    def __init__(self, kernel_size=64, out_chan=128):
        super(TCNFrontend, self).__init__()
        self.conv_in = nn.Conv1d(1, 64, kernel_size, stride=2,
                                 padding=0, bias=False)
        self.bn_in = nn.BatchNorm1d(64)
        self.block1 = TCNBlock(64, 64, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.block3 = TCNBlock(64, 64, kernel_size=3, dilation=4)
        self.block4 = TCNBlock(64, out_chan, kernel_size=3, dilation=8)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.out_chan = out_chan

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return x


class SepConvBlock(nn.Module):
    """深度可分离卷积块（1D）：depthwise + pointwise，参数/计算量约为标准卷积的 1/k。"""

    def __init__(self, ch_in, ch_out, kernel, stride=1, groups=None):
        super(SepConvBlock, self).__init__()
        g = groups if groups is not None else ch_in
        self.dw = nn.Conv1d(ch_in, ch_in, kernel, stride=stride,
                            padding=kernel // 2, groups=g, bias=False)
        self.bn1 = nn.BatchNorm1d(ch_in)
        self.pw = nn.Conv1d(ch_in, ch_out, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.dw(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.pw(x)), 0.2)
        return x


class SepCNNFrontend(nn.Module):
    """深度可分离 CNN 前端（MobileNet 思想，1D）。

    输入 (B,1,1024) -> (B,out_chan,60)。用深度可分离卷积在"与 4 层 CNN
    相当"的计算量下堆到 8 层，获得更深层次、更大感受野的局部冲击特征。
    """

    def __init__(self, kernel_size=64, out_chan=128):
        super(SepCNNFrontend, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=4,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        self.block1 = SepConvBlock(32, 64, 7)
        self.block2 = SepConvBlock(64, 64, 7)
        self.block3 = SepConvBlock(64, 128, 7)
        self.block4 = SepConvBlock(128, 128, 7)
        self.block5 = SepConvBlock(128, 128, 7)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.out_chan = out_chan

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)   # L=241
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)                                  # L=60
        return x


class InceptionBlock(nn.Module):
    """1D Inception 多尺度块：并行 k=1/3/7/15 分支，1x1 卷积先降维。"""

    def __init__(self, ch_in, ch_out):
        super(InceptionBlock, self).__init__()
        c = ch_out // 4
        self.branch1 = nn.Conv1d(ch_in, c, 1, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv1d(ch_in, c, 1, bias=False),
            nn.Conv1d(c, c, 3, padding=1, bias=False))
        self.branch7 = nn.Sequential(
            nn.Conv1d(ch_in, c, 1, bias=False),
            nn.Conv1d(c, c, 7, padding=3, bias=False))
        self.branch15 = nn.Sequential(
            nn.Conv1d(ch_in, c, 1, bias=False),
            nn.Conv1d(c, c, 15, padding=7, bias=False))
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        b15 = self.branch15(x)
        out = torch.cat([b1, b3, b7, b15], dim=1)
        return F.leaky_relu(self.bn(out), 0.2)


class InceptionFrontend(nn.Module):
    """Inception 多尺度 CNN 前端（IEDGNet 思想）。

    输入 (B,1,1024) -> (B,out_chan,60)。并行多尺度卷积（k=1/3/7/15）
    同时捕获不同尺度的冲击特征，计算量与传统 4 层 CNN 相当。
    """

    def __init__(self, kernel_size=64, out_chan=128):
        super(InceptionFrontend, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=4,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.inc1 = InceptionBlock(32, 64)
        self.inc2 = InceptionBlock(64, 128)
        self.inc3 = InceptionBlock(128, out_chan)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.out_chan = out_chan

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # L=241
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = self.pool(x)                                # L=60
        return x


class MRAEncoderBranch(nn.Module):
    """MRA-SNN 单尺度编码分支（连续化：脉冲神经元 ASN -> LeakyReLU）。

    对应 MRA-SNN 的 coding3/5/7：Conv(k)->BN->激活->AvgPool(2)->Conv(k)->BN。
    输入 (B,1,L) -> (B,64,L/2)。
    """

    def __init__(self, kernel):
        super(MRAEncoderBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel, stride=1, padding=kernel // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel, stride=1, padding=kernel // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return x


class MRAFrontend(nn.Module):
    """MRA-SNN 多尺度注意力编码前端（连续化）。

    三个并行分支（k=3/5/7）提取多尺度冲击特征，concat 后经 ECA 通道
    注意力加权，最后 2 倍池化。输入 (B,1,1024) -> (B,192,256)。
    保留 MRA-SNN 的多尺度编码 + ECA 注意力结构；脉冲神经元(ASN)替换为
    LeakyReLU，以输出连续特征适配下游液体 SSM（SSM 需要连续值输入）。
    """

    def __init__(self):
        super(MRAFrontend, self).__init__()
        self.coding3 = MRAEncoderBranch(3)
        self.coding5 = MRAEncoderBranch(5)
        self.coding7 = MRAEncoderBranch(7)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.pool = nn.AvgPool1d(2)
        self.out_chan = 192

    def forward(self, x):
        o3 = self.coding3(x)
        o5 = self.coding5(x)
        o7 = self.coding7(x)
        x = torch.cat((o3, o5, o7), dim=1)  # (B, 192, L/2)

        # ECA 通道注意力（与 MRA-SNN 一致）
        y = F.adaptive_avg_pool1d(x, 1)
        y = self.eca_conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = torch.sigmoid(y)
        x = x * y.expand_as(x)

        x = self.pool(x)  # (B, 192, L/4)
        return x


# ---------------------------------------------------------------------------
# Liquid-Mamba 块：因果卷积 + 液体时间常数 + 选择性 SSM
# ---------------------------------------------------------------------------
class LiquidMambaBlock(nn.Module):
    def __init__(self, d_model=128, d_inner=256, n_state=16, conv_kernel=4,
                 dt_rank=16, dt_min=1e-3, dt_max=20.0, gamma_init=0.1,
                 a_lo=0.5, a_hi=3.0, tau_proj=False, film=False,
                 use_tau=True):
        super(LiquidMambaBlock, self).__init__()
        self.d_inner = d_inner
        self.n_state = n_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        # 干净版开关：False -> 不构建 γ/τ_proj/FiLM，也不计算液体项
        self.use_tau = bool(use_tau)

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                                padding=conv_kernel - 1, groups=d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * n_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # 状态矩阵 A 固定为负（衰减），保证数值稳定
        A_log = torch.log(torch.linspace(a_lo, a_hi, n_state)).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(A_log)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 转速敏感门控 gamma：tau0 -> 时间常数调制
        # 干净版（use_tau=False）下整条液体 τ₀ 通路不存在的参数都不创建，
        # 状态方程退化为标准 Mamba 选择性 Δ。
        self.gamma = (nn.Parameter(torch.tensor(gamma_init))
                      if self.use_tau else None)
        # 通道级 tau0 投影（把标量周期扩展为每通道灵敏度）
        self.tau_proj = (nn.Linear(1, d_inner, bias=False)
                         if (self.use_tau and tau_proj) else None)

        self.norm = nn.LayerNorm(d_model)

        # τ₀ 条件化归一化（FiLM）：用瞬时周期调制 LayerNorm 的 scale/shift，
        # 让归一化显式感知转速（零初始化 -> 初始等价恒等，不破坏训练行为）
        self.film = (nn.Linear(1, 2 * d_model, bias=True)
                     if (self.use_tau and film) else None)
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, tau0):
        """x: (B, L, d_model); tau0: (B,1,1) -> out (B, L, d_model), delta_norm"""
        res = x
        x_and_z = self.in_proj(x)  # (B, L, 2*d_inner)
        x_b, z_b = x_and_z.split(self.d_inner, dim=-1)

        # 因果卷积（只看向过去，CCM-TAD 因果思想）
        x_b = x_b.transpose(1, 2)  # (B, D, L)
        x_b = self.conv1d(x_b)[..., :x_b.shape[-1]]
        x_b = F.silu(x_b).transpose(1, 2)  # (B, L, D)

        # 液体时间常数：Delta = softplus(W_dt x + b + gamma * tau0)
        # 干净版（use_tau=False 或未传 tau0）跳过该调制项
        s = self.x_proj(x_b)  # (B, L, dt_rank + 2N)
        dt, Bm, Cm = s.split([self.x_proj.out_features - 2 * self.n_state,
                              self.n_state, self.n_state], dim=-1)
        delta = self.dt_proj(dt)  # (B, L, D)
        if self.use_tau and tau0 is not None:
            tau_c = (self.tau_proj(tau0) if self.tau_proj is not None
                     else tau0)  # (B, 1, D) 通道级灵敏度
            delta = delta + self.gamma * tau_c  # 液体项：瞬时周期调制
        delta = torch.clamp(F.softplus(delta + self.dt_proj.bias),
                            min=self.dt_min, max=self.dt_max)

        # 选择性 SSM 扫描（全 fp32）
        u = x_b.float().transpose(1, 2)  # (B, D, L)
        A = -torch.exp(self.A_log)  # (D, N)，严格负
        delta_f = delta.float()  # (B, L, D)
        B_f = Bm.float()  # (B, L, N)
        C_f = Cm.float()  # (B, L, N)
        y = self._selective_scan(u, delta_f, A, B_f, C_f)  # (B, D, L)

        y = y.transpose(1, 2).to(x.dtype)  # (B, L, D)
        out = y * F.silu(z_b)
        out = self.out_proj(out)
        x = res + self.norm(out)

        # FiLM：τ₀ 条件化 scale/shift（tau0: (B,1,1) -> (B,1,2D)）
        if self.film is not None and tau0 is not None:
            gamma, beta = self.film(tau0).chunk(2, dim=-1)
            x = x * (1.0 + gamma) + beta

        delta_norm = delta.square().mean()
        return x, delta_norm

    @staticmethod
    def _selective_scan(u, delta, A, B, C, chunk=8):
        """选择性 SSM 扫描：(B,D,L) -> (B,D,L)。

        优先官方 mamba_ssm CUDA 内核（语义与 _python_scan 完全一致：
        delta_softplus=False、delta_bias=None、z=None、D=None；
        内核需要 channel-first：u/delta (B,D,L)、B/C (B,N,L)）。
        CPU 或导入失败时回落纯 Python 分块扫描。
        """
        if _OFFICIAL_SCAN is not None and u.is_cuda:
            return _OFFICIAL_SCAN(
                u.contiguous(), delta.transpose(1, 2).contiguous(),
                A.contiguous(), B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                None, None, None, False)  # D, z, delta_bias, delta_softplus
        return LiquidMambaBlock._python_scan(u, delta, A, B, C, chunk)

    @staticmethod
    def _python_scan(u, delta, A, B, C, chunk=8):
        """分块并行选择性扫描（纯 Python 后备实现，Mamba1 ZOH 语义）。

        u: (B,D,L); delta: (B,L,D); A: (D,N); B,C: (B,L,N) -> (B,D,L)
        递推: z_t = Abar_t * z_{t-1} + delta_t * B_t * u_t,  y_t = C_t * z_t
        其中 Abar_t = exp(delta_t * A)。
        块内用 log-space cumprod/cumsum 全并行，块间顺序递推；
        中间量用 float64 防止下溢（chunk 内 logP 最负约 -480）。
        """
        B_, D, L = u.shape
        N = A.shape[-1]
        logA = (delta.unsqueeze(-1) * A.view(1, 1, D, N)).double()  # (B,L,D,N) 严格负
        K = (L + chunk - 1) // chunk
        pad = K * chunk - L
        if pad:
            logA = F.pad(logA, (0, 0, 0, 0, 0, pad))  # (B,L,D,N) pad L 维
            delta = F.pad(delta, (0, 0, 0, pad))      # (B,L,D) pad L 维
            u = F.pad(u, (0, pad, 0, 0))              # (B,D,L) pad L 维
            B = F.pad(B, (0, 0, 0, pad))              # (B,L,N) pad L 维
            C = F.pad(C, (0, 0, 0, pad))
        logA = logA.view(B_, K, chunk, D, N)
        delta = delta.view(B_, K, chunk, D).double()
        Bv = B.view(B_, K, chunk, N).double()
        Cv = C.view(B_, K, chunk, N).double()
        # u 为 (B,D,L)，view 前先转成 (B,L,D)
        uv = u.transpose(1, 2).contiguous().view(B_, K, chunk, D).double()

        z = torch.zeros(B_, D, N, device=u.device, dtype=torch.float64)
        ys = []
        for k in range(K):
            logP = torch.cumsum(logA[:, k], dim=1)  # (B,chunk,D,N)
            P = torch.exp(logP)                     # float64, chunk 内不溢出
            # 标准 ZOH 离散化：B_bar = delta * B
            dB = delta[:, k].unsqueeze(-1) * Bv[:, k].unsqueeze(2)  # (B,chunk,D,N)
            Bu = dB * uv[:, k].unsqueeze(-1)  # (B,chunk,D,N)
            S = torch.cumsum(Bu / P, dim=1)
            zc = P * (z.unsqueeze(1) + S)  # (B,chunk,D,N)
            yk = (Cv[:, k].unsqueeze(2) * zc).sum(-1)  # (B,chunk,D)
            ys.append(yk.float().transpose(1, 2))  # (B,D,chunk)
            z = zc[:, -1]
        return torch.cat(ys, dim=2)[:, :, :L]  # (B, D, L)


# ---------------------------------------------------------------------------
# ResNet18-1D 前端（返回 layer3 时序特征 + 最终全局特征）
# ---------------------------------------------------------------------------
def _conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.conv1 = _conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetSeqFrontend(nn.Module):
    """ResNet18-1D 前端。

    forward(x) -> (seq, final)：
      seq:   layer3 输出 (B, 256, L3)，保留时序维，供 SSM 处理
      final: layer4 + 全局平均池化 (B, 512)
    """

    def __init__(self):
        super(ResNetSeqFrontend, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * _BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _BasicBlock.expansion, stride),
                nn.BatchNorm1d(planes * _BasicBlock.expansion),
            )
        layers = []
        layers.append(_BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * _BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        seq = x  # (B, 256, L3)
        x = self.layer4(x)
        x = self.avgpool(x)
        final = x.view(x.size(0), -1)  # (B, 512)
        return seq, final


class WideResNetSeqFrontend(nn.Module):
    """宽核 Laplace 小波 ResNet18-1D 前端（复刻 WIDAN backbone）。

    与 ResNetSeqFrontend 的区别：conv1 用 **kernel_size=64 宽核**（WIDAN 的关键
    设计，匹配轴承冲击的物理尺度）+ Laplace 小波核初始化（可学习微调）。

    forward(x) -> (seq, final)：
      seq:   layer3 输出 (B, 256, L3)，保留时序维供液体 SSM 处理
      final: layer4 + 全局平均池化 (B, 512)
    """

    def __init__(self, kernel_size=64, wave_init=True, wavelet_frequency=64000.0):
        super(WideResNetSeqFrontend, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size, stride=2,
                               padding=kernel_size // 2, bias=False)
        if wave_init:
            with torch.no_grad():
                self.conv1.weight.copy_(
                    laplace_wavelet_kernel(64, kernel_size, 0.01, 0.08, 0.08))
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) and m is not self.conv1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * _BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _BasicBlock.expansion, stride),
                nn.BatchNorm1d(planes * _BasicBlock.expansion),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        seq = x                              # (B, 256, L3)
        x = self.layer4(x)
        x = self.avgpool(x)
        final = x.view(x.size(0), -1)        # (B, 512)
        return seq, final


# ---------------------------------------------------------------------------
# 多尺度 Liquid-Mamba 层：不同膨胀因果卷积分支 + SSM，融合多时间尺度上下文
# ---------------------------------------------------------------------------
class MultiScaleMambaLayer(nn.Module):
    def __init__(self, d_model=128, d_inner=64, n_state=8, dilations=(1, 2, 4),
                 conv_kernel=4, gamma_init=0.1, a_lo=0.5, a_hi=3.0,
                 tau_proj=False, use_tau=True):
        super(MultiScaleMambaLayer, self).__init__()
        self.dilations = dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                      dilation=d, padding=(conv_kernel - 1) * d,
                      groups=d_model, bias=True)
            for d in dilations
        ])
        self.blocks = nn.ModuleList([
            LiquidMambaBlock(d_model=d_model, d_inner=d_inner, n_state=n_state,
                             gamma_init=gamma_init, a_lo=a_lo, a_hi=a_hi,
                             tau_proj=tau_proj, use_tau=use_tau)
            for _ in dilations
        ])
        self.fuse = nn.Linear(len(dilations) * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, tau0):
        res = x
        outs = []
        delta_norm = torch.zeros((), device=x.device, dtype=x.dtype)
        for conv, blk in zip(self.convs, self.blocks):
            hb = conv(x.transpose(1, 2))[..., :x.shape[1]]  # 因果（裁剪右侧）
            hb = F.silu(hb).transpose(1, 2)  # (B,L,D)
            hb, dn = blk(hb, tau0)
            outs.append(hb)
            delta_norm = delta_norm + dn
        h = torch.cat(outs, dim=-1)  # (B, L, n*D)
        h = self.fuse(h)
        return res + self.norm(h), delta_norm / len(self.dilations)


# ---------------------------------------------------------------------------
# 域判别器（轻量 MLP，输入经 GRL）
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, d_feat=128, n_domains=3, grl_lambda=1.0):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReversal(grl_lambda)
        self.net = nn.Sequential(
            nn.Linear(d_feat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_domains),
        )

    def forward(self, z):
        return self.net(self.grl(z))


# ---------------------------------------------------------------------------
# 总体模型
# ---------------------------------------------------------------------------
class LiquidMambaDGNet(nn.Module):
    def __init__(self, num_classes=14, d_model=128, d_inner=128, n_blocks=3,
                 n_state=8, num_patches=256, front_chan=128,
                 kernel_size=64, stride=4, detach_tau=False, wave_init=False,
                 gamma_init=0.1, a_lo=0.5, a_hi=3.0,
                 tau_min_lag=8, tau_max_lag=300, tau_temp=0.05,
                 tau_f_lo=3, tau_f_hi=120,
                 use_pos_embed=True, head_dropout=0.3,
                 tau_estimator='autocorr', tau_proj=False, tau_mode='normal',
                 ssm_mode='single', ms_dilations=(1, 2, 4), ms_d_inner=64,
                 front_mode='cnn', front_norm='bn', front_init='none',
                 gabor_sigma=0.5, gabor_f_lo=0.005, gabor_f_hi=0.45,
                 gabor_dc_frac=0.5, gabor_alias=0.0,
                 freeze_init=False, film=False, use_ssm=True):
        super(LiquidMambaDGNet, self).__init__()
        self.detach_tau = detach_tau
        self.tau_min_lag = tau_min_lag
        self.tau_max_lag = tau_max_lag
        self.tau_temp = tau_temp
        self.tau_f_lo = tau_f_lo
        self.tau_f_hi = tau_f_hi
        self.use_pos_embed = use_pos_embed
        self.tau_estimator = tau_estimator
        # 是否启用 SSM 时序分支（False -> 仅用前端 + 池化，消融用）
        self.use_ssm = bool(use_ssm)
        # E3 因果消融：'normal' | 'zero'(切断 gamma*tau0 通路) |
        #              'shuffle'(打乱样本对应) | 'const'(去样本信息)
        self.tau_mode = tau_mode
        # 干净版：'zero' 时整条 τ₀ 通路（估计器 + γ 调制 + FiLM）都不构建
        self.use_tau = str(tau_mode).lower() != 'zero'
        self.ssm_mode = ssm_mode
        self.front_mode = front_mode

        if front_mode == 'resnet18':
            self.front = ResNetSeqFrontend()
            self.patch_proj = nn.Linear(256, d_model)
            self.cnn_proj = nn.Linear(512, d_model)
        elif front_mode == 'wide_resnet18':
            self.front = WideResNetSeqFrontend(
                kernel_size=kernel_size, wave_init=wave_init)
            self.patch_proj = nn.Linear(256, d_model)
            self.cnn_proj = nn.Linear(512, d_model)
        elif front_mode == 'deep6':
            self.front = DeepCNN6Frontend(kernel_size=kernel_size,
                                          wave_init=wave_init,
                                          out_chan=front_chan)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'tcn':
            self.front = TCNFrontend(kernel_size=kernel_size,
                                     out_chan=front_chan)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'sep':
            self.front = SepCNNFrontend(kernel_size=kernel_size,
                                        out_chan=front_chan)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'se':
            self.front = WaveletFrontend(out_chan=front_chan,
                                         kernel_size=kernel_size,
                                         stride=stride, wave_init=wave_init,
                                         se=True, norm=front_norm,
                                         init_mode=front_init,
                                         gabor_sigma=gabor_sigma,
                                         freeze_init=freeze_init)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'incept':
            self.front = InceptionFrontend(kernel_size=kernel_size,
                                           out_chan=front_chan)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'mra':
            self.front = MRAFrontend()
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        elif front_mode == 'gaborwide':
            self.front = GaborBankFrontend(
                kernel_size=kernel_size, out_chan=front_chan,
                f_lo=gabor_f_lo, f_hi=gabor_f_hi,
                sigma_ratio=gabor_sigma, dc_frac=gabor_dc_frac,
                alias=gabor_alias,
                # ⚠ 白名单必须包含 'laplace'：否则 §6.3 的 Laplace 臂会被
                # 静默回退成 'gabor'（配置写着 laplace、模型跑的是 Gabor）
                init=front_init if front_init in ('gabor', 'random', 'mix',
                                                  'laplace') else 'gabor',
                freeze_init=freeze_init)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        else:
            self.front = WaveletFrontend(out_chan=front_chan,
                                         kernel_size=kernel_size,
                                         stride=stride, wave_init=wave_init,
                                         norm=front_norm,
                                         init_mode=front_init,
                                         gabor_sigma=gabor_sigma,
                                         freeze_init=freeze_init)
            self.patch_proj = nn.Linear(self.front.out_chan, d_model)
            self.cnn_proj = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter('pos_embed', None)

        # 通道级 tau0 投影在块内部实现（tau_proj 参数传给每个块）

        if ssm_mode == 'multiscale':
            self.ms_layers = nn.ModuleList([
                MultiScaleMambaLayer(d_model=d_model, d_inner=ms_d_inner,
                                     n_state=n_state,
                                     dilations=tuple(ms_dilations),
                                     gamma_init=gamma_init, a_lo=a_lo, a_hi=a_hi,
                                     tau_proj=tau_proj,
                                     use_tau=self.use_tau)
                for _ in range(max(n_blocks, 1))
            ])
            self.blocks = nn.ModuleList()
        else:
            self.ms_layers = nn.ModuleList()
            self.blocks = nn.ModuleList([
                LiquidMambaBlock(d_model=d_model, d_inner=d_inner, n_state=n_state,
                                 gamma_init=gamma_init, a_lo=a_lo, a_hi=a_hi,
                                 tau_proj=tau_proj, film=film,
                                 use_tau=self.use_tau)
                for _ in range(n_blocks)
            ])
        self.norm_f = nn.LayerNorm(d_model)
        # 双分支：CNN 直连特征 + SSM 时序特征，mean+max 各自拼接
        if front_mode in ('resnet18', 'wide_resnet18'):
            feat_dim = d_model + (2 * d_model if use_ssm else 0)
        elif n_blocks > 0 and use_ssm:
            feat_dim = 4 * d_model
        else:
            feat_dim = 2 * d_model
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )
        self.feat_dim = feat_dim
        self.n_blocks = n_blocks

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B,1,L) -> logits (B,C), z (B,feat), delta_norm, tau0"""
        # ---- 干净版：τ₀ 通路整体关闭 ----
        # use_tau=False 时完全不调用包络/FFT 估计器（省掉 2 次 FFT + 自相关），
        # 也不进入 ρ 调制与 FiLM，状态方程退化为标准 Mamba 选择性 Δ。
        if not self.use_tau:
            tau0 = None
        else:
            if self.tau_estimator == 'envspec':
                tau0 = envelope_spectrum_period(x, f_lo=self.tau_f_lo,
                                                f_hi=self.tau_f_hi,
                                                temp=self.tau_temp)
            else:
                tau0 = envelope_autocorr_period(x, min_lag=self.tau_min_lag,
                                                max_lag=self.tau_max_lag,
                                                temp=self.tau_temp)  # (B,1,1)
            # ---- E3 因果消融：干预 tau0 通路 ----
            if self.tau_mode == 'shuffle':
                if tau0.shape[0] > 1:
                    tau0 = tau0[torch.randperm(tau0.shape[0],
                                               device=tau0.device)]
            elif self.tau_mode == 'const':
                tau0 = tau0.mean().expand_as(tau0).contiguous()
            if self.detach_tau:
                tau0 = tau0.detach()
        if self.front_mode in ('resnet18', 'wide_resnet18'):
            h_seq, h_final = self.front(x)  # (B,256,L3), (B,512)
            h = h_seq.transpose(1, 2)       # (B, L3, 256)
            h = self.patch_proj(h)          # (B, L3, D)
        else:
            h = self.front(x)               # (B, C, L')
            h = h.transpose(1, 2)           # (B, L', C)
            h = self.patch_proj(h)          # (B, L', D)
        if self.use_pos_embed:
            L = h.shape[1]
            h = h + self.pos_embed[:, :L]

        # 通道级 tau0 投影在块内处理，这里直接传原始 tau0
        delta_norm = torch.zeros((), device=x.device, dtype=x.dtype)
        h_ssm = h
        if self.ssm_mode == 'multiscale':
            for layer in self.ms_layers:
                h_ssm, dn = layer(h_ssm, tau0)
                delta_norm = delta_norm + dn
            n_units = max(len(self.ms_layers), 1)
        else:
            for blk in self.blocks:
                h_ssm, dn = blk(h_ssm, tau0)
                delta_norm = delta_norm + dn
            n_units = max(len(self.blocks), 1)

        h_ssm = self.norm_f(h_ssm)

        if self.front_mode in ('resnet18', 'wide_resnet18'):
            z_cnn = self.cnn_proj(h_final)      # (B, D)
        else:
            z_cnn = self._pool_cat(h)           # (B, 2D)
        if self.use_ssm:
            z_ssm = self._pool_cat(h_ssm)       # (B, 2D)
            z = torch.cat([z_cnn, z_ssm], dim=1)
        else:
            # 消融：完全不用 SSM 分支（h_ssm 未参与，视为恒等）
            z = z_cnn
        logits = self.head(z)
        # 干净版下 tau0 恒为 0（仅为日志/接口兼容保留张量形状）
        tau0_out = (tau0 if tau0 is not None
                    else torch.zeros(x.shape[0], 1, 1,
                                     device=x.device, dtype=x.dtype))
        return logits, z, delta_norm / n_units, tau0_out

    @staticmethod
    def _pool_cat(h):
        return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
