#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
P2C-DGNet: Physics-guided, Prototype-aware, Causal Domain Generalization Network.

多视图输入 + 物理语义锚点 + 因果解耦 + 原型图 + 域加权 + 物理一致外推增强。
本文件完全独立，不依赖、也不修改项目内任何现有模型文件。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['P2CDGNet']


# ---------------------------------------------------------------------------
# 梯度反转层（用于域判别器）
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
# 信号视图构造（全部可导，torch 实现，不依赖 scipy）
# ---------------------------------------------------------------------------
def hilbert_envelope(x):
    """解析信号包络，x: (B, L)"""
    B, L = x.shape
    X = torch.fft.rfft(x, dim=-1)  # (B, N), N = L//2 + 1
    N = X.shape[-1]
    Z = torch.zeros_like(X)
    Z[:, 0] = X[:, 0]             # DC 保持不变
    if N > 2:
        Z[:, 1:N - 1] = 2.0 * X[:, 1:N - 1]  # 正频率翻倍
    Z[:, N - 1] = X[:, N - 1]     # Nyquist 保持不变
    analytic = torch.fft.irfft(Z, n=L, dim=-1)
    return torch.abs(analytic)


def envelope_spectrum(x, out_len=512):
    """包络谱视图，x: (B, L) -> (B, out_len)"""
    env = hilbert_envelope(x)
    spec = torch.abs(torch.fft.rfft(env, dim=-1))  # (B, L//2+1)
    if spec.shape[-1] >= out_len:
        spec = spec[:, :out_len]
    else:
        spec = F.pad(spec, (0, out_len - spec.shape[-1]))
    return spec


def stft_magnitude(x, n_fft=64, hop_length=16, window=None):
    """时频图视图，x: (B, L) -> (B, 1, F, T)"""
    if window is None:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                      window=window, center=True, pad_mode='reflect',
                      normalized=False, onesided=True, return_complex=True)
    mag = spec.abs().unsqueeze(1)  # (B, 1, F, T)
    return mag


def speed_migrate(x, ratio):
    """频域频率迁移：特征频率按 ratio 缩放（模拟转速变化），时域冲击结构保持。
    x: (B, L) -> (B, L)。ratio>1 表示转速升高（特征频率向高频迁移）。"""
    B, L = x.shape
    X = torch.fft.rfft(x, dim=-1)  # (B, N)
    N = X.shape[-1]
    Lf = float(N - 1)
    out_coords = torch.arange(N, device=x.device).float().expand(B, N)  # (B, N)
    src = out_coords * ratio.unsqueeze(1)  # (B, N) 源频率索引
    src_norm = 2.0 * src / Lf - 1.0  # (B, N) -> [-1, 1]
    # grid_sample 坐标 (x, y)：x 对应 W 维（频率轴），y 对应 H 维（恒为 0）
    grid = torch.stack([src_norm, torch.zeros_like(src_norm)], dim=-1).unsqueeze(1)  # (B, 1, N, 2)
    Xr = X.real.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
    Xi = X.imag.unsqueeze(1).unsqueeze(2)
    Xr_new = F.grid_sample(Xr, grid, mode='bilinear', align_corners=True,
                           padding_mode='zeros').squeeze(2).squeeze(1)  # (B, N)
    Xi_new = F.grid_sample(Xi, grid, mode='bilinear', align_corners=True,
                           padding_mode='zeros').squeeze(2).squeeze(1)
    X_new = torch.complex(Xr_new, Xi_new)
    return torch.fft.irfft(X_new, n=L, dim=-1)


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    """1D 时域编码器，输入 (B, 1, L)，输出 (B, 256)"""

    def __init__(self, feat_dim=256):
        super(TemporalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=25, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Linear(128 * 4, feat_dim)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SpectrumEncoder(nn.Module):
    """1D 包络谱编码器，输入 (B, 1, N)，输出 (B, 128)"""

    def __init__(self, feat_dim=128):
        super(SpectrumEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.AdaptiveMaxPool1d(2)
        self.fc = nn.Linear(128 * 2, feat_dim)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TFSpectrogramEncoder(nn.Module):
    """2D 时频图编码器，输入 (B, 1, F, T)，输出 (B, 128)"""

    def __init__(self, feat_dim=128):
        super(TFSpectrogramEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(64 * 2 * 2, feat_dim)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 物理语义锚点（cross-attention）
# ---------------------------------------------------------------------------
class PhysicsSemanticAnchor(nn.Module):
    """可学习物理语义锚点库 + cross-attention 交互"""

    def __init__(self, num_classes, dim):
        super(PhysicsSemanticAnchor, self).__init__()
        self.anchor = nn.Parameter(torch.randn(num_classes, dim) * 0.1)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, h):
        # h: (B, dim)
        Q = self.wq(h)                      # (B, dim)
        K = self.wk(self.anchor)            # (K, dim)
        V = self.wv(self.anchor)            # (K, dim)
        attn_logits = Q @ K.transpose(0, 1) * self.scale  # (B, K)
        attn = torch.softmax(attn_logits, dim=-1)         # (B, K)
        h_sem = attn @ V                    # (B, dim)
        return h_sem, attn, attn_logits


# ---------------------------------------------------------------------------
# 域判别器
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_domains, grl_lambda=1.0):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReversal(lambda_=grl_lambda)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains),
        )

    def forward(self, z):
        return self.net(self.grl(z))


# ---------------------------------------------------------------------------
# P2C-DGNet 主模型
# ---------------------------------------------------------------------------
class P2CDGNet(nn.Module):
    def __init__(self, num_classes, num_domains=3, feat_dim=256, aug_ratio=0.6):
        super(P2CDGNet, self).__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feat_dim = feat_dim
        # 频域迁移增强的转速比范围 [aug_ratio, 1/aug_ratio]，覆盖跨转速偏移
        self.aug_ratio = aug_ratio

        self.enc_t = TemporalEncoder(feat_dim=feat_dim)
        self.enc_s = SpectrumEncoder(feat_dim=128)
        self.enc_tf = TFSpectrogramEncoder(feat_dim=128)

        # 频域视图 dropout + 降权（频域视图在跨转速下易过拟合，时域主导）
        self.freq_drop = nn.Dropout(0.3)
        self.freq_weight = 0.2

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 128 + 128, feat_dim),
            nn.ReLU(inplace=True),
        )
        # LayerNorm 稳定各表示范数（防止后续归一化损失因范数趋零而爆炸）
        self.ln_h = nn.LayerNorm(feat_dim)
        self.ln_sem = nn.LayerNorm(feat_dim)
        self.ln_f = nn.LayerNorm(feat_dim)
        self.ln_d = nn.LayerNorm(feat_dim)

        self.anchor = PhysicsSemanticAnchor(num_classes, feat_dim)

        self.proj_f = nn.Linear(feat_dim, feat_dim)   # 故障因果投影
        self.proj_d = nn.Linear(feat_dim, feat_dim)   # 工况扰动投影

        self.classifier = nn.Linear(feat_dim, num_classes)
        self.domain_disc = DomainDiscriminator(feat_dim, num_domains, grl_lambda=1.0)

        # 可学习类原型 + 原型图卷积
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.1)
        self.proto_gcn = nn.Linear(feat_dim, feat_dim)

        self.register_buffer('hann64', torch.hann_window(64))

    # -- 前向（默认输入 (B, 1, L) 时域） ------------------------------------
    def forward(self, x, return_dict=True):
        B = x.size(0)
        x1d = x.squeeze(1)                       # (B, L)
        env_spec = envelope_spectrum(x1d, out_len=512)      # (B, 512)
        tf_view = stft_magnitude(x1d, n_fft=64, hop_length=16, window=self.hann64)

        f_t = self.enc_t(x)                                     # (B, 256)
        f_s = self.freq_drop(self.enc_s(env_spec.unsqueeze(1))) * self.freq_weight  # (B, 128)
        f_tf = self.freq_drop(self.enc_tf(tf_view)) * self.freq_weight              # (B, 128)

        h = self.fusion(torch.cat([f_t, f_s, f_tf], dim=1))     # (B, 256)
        h = self.ln_h(h)                                        # 稳定范数
        h_sem, attn, attn_logits = self.anchor(h)               # (B, 256), (B, K), (B, K)
        h_sem = self.ln_sem(h_sem)

        z_f = self.ln_f(self.proj_f(h_sem))                     # 故障因果表示
        z_d = self.ln_d(self.proj_d(h))                         # 工况扰动表示

        logits = self.classifier(z_f)
        domain_logits = self.domain_disc(z_d)

        if not return_dict:
            return logits

        return {
            'logits': logits,          # (B, num_classes)
            'z_f': z_f,                # (B, 256)
            'z_d': z_d,                # (B, 256)
            'h': h,                    # 融合中间表示（自蒸馏 student）
            'h_sem': h_sem,            # 语义增强表示（自蒸馏 teacher）
            'attn': attn,              # 物理注意图 (B, K)
            'attn_logits': attn_logits,  # 注意力 logits (B, K)
            'domain_logits': domain_logits,  # (B, num_domains)
        }

    # -- 物理一致外推增强（频域频率迁移模拟转速变化） --------------------
    def physics_augment(self, x):
        """x: (B, 1, L)。频域频率迁移：特征频率按随机转速比缩放，
        时域冲击结构保持。ratio 范围 [aug_ratio, 1/aug_ratio]，
        让模型在训练时见到不同转速的虚拟样本，学习转速不变性。"""
        if not self.training:
            return x
        B, C, L = x.shape
        lo = self.aug_ratio
        hi = 1.0 / self.aug_ratio
        # 对数均匀采样，使转速比在 log 空间均匀分布
        log_ratio = torch.empty(B, device=x.device).uniform_(math.log(lo), math.log(hi))
        ratio = torch.exp(log_ratio)
        x1d = speed_migrate(x.squeeze(1), ratio)  # (B, L)
        return x1d.unsqueeze(1)

    # -- 原型图前向传播 -----------------------------------------------------
    def graph_propagate(self):
        """对类原型做一层图卷积（高斯邻接）。返回更新后的原型 (K, dim)。"""
        P = self.prototypes  # (K, dim)
        with torch.no_grad():
            dist = torch.cdist(P, P, p=2)  # (K, K)
            sigma = dist.mean().clamp(min=1e-6)
            A = torch.exp(-dist ** 2 / (2 * sigma ** 2))  # (K, K)
            D = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
            A_norm = A / D
        P_refined = self.proto_gcn(P)             # (K, dim)
        P_refined = torch.relu(A_norm @ P_refined)
        return P_refined

    def prototype_loss(self, z_f, labels):
        """prototype compactness + separation。z_f: (B, dim), labels: (B,)"""
        P = self.graph_propagate()  # (K, dim)
        P = F.normalize(P, dim=-1)
        z = F.normalize(z_f, dim=-1)
        # compactness：样本与其类原型的距离
        idx = labels.view(-1, 1).expand(-1, P.size(1))
        proto_per_sample = P.gather(0, idx)       # (B, dim)
        compact = (z - proto_per_sample).pow(2).sum(dim=1).mean()
        # separation：不同原型之间应相互远离
        sim = P @ P.transpose(0, 1)               # (K, K) 余弦相似
        eye = torch.eye(P.size(0), device=P.device)
        separation = sim.masked_select(~eye.bool()).mean()
        return compact - 0.5 * separation
