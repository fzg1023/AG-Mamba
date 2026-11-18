#!/usr/bin/python
# -*- coding:utf-8 -*-
"""MRA-SNN 网络（移植自 other/MRA-SNN-main/networks.py，SIGKDD'25）。

Multi-scale Residual Attention SNN：
  多尺度编码模块 (coding3/5/7) -> ECA 通道注意力 -> 注意力脉冲神经元(ASN)
  -> 脉冲残差注意力块 (AttSpikingBasicBlock) -> 全局池化 -> 分类头

关键机制：
- TimeStep=4 时间步展开：输入需复制 TimeStep 次（见原版 train.py），
  网络内部按 TimeStep 分段做膜电位积分与脉冲发放。
- 替代梯度 (surrogate gradient)：前向阶跃函数，反向三角近似，
  使不可微的脉冲发放可以端到端训练。

本文件仅做移植与接口适配（num_classes 可配置），网络结构保持原版。
"""
from math import log

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['MRASNN', 'TimeStep']

Vth = 1.0
a = 1.0
TimeStep = 4
tau = 0.5


class SpikeFunction(torch.autograd.Function):
    """替代梯度：前向 spike = (u > Vth)，反向三角窗近似。"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.gt(input, Vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth) < (a / 2)) / a
        return grad_input * hu


spikefunc = SpikeFunction.apply


class ASN(nn.Module):
    """注意力脉冲神经元（Attention Spiking Neuron）。

    ECA 式通道注意力 + 空间注意力 -> 加权输入 -> LIF 膜电位积分 -> 脉冲发放。
    输入/输出均为 (TimeStep*B, C, L)（按 TimeStep 分段处理）。
    """

    def __init__(self, channel=64):
        super(ASN, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2),
                                 stride=1, bias=False)
        self.sa_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, stride=1,
                                 bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(TimeStep):
            tmp = self.pool(x[t * bs:(t + 1) * bs, ...]).transpose(-1, -2)
            ca_w = self.ca_conv(tmp).transpose(-1, -2)
            sa_w = self.sa_conv(torch.mean(x[t * bs:(t + 1) * bs, ...],
                                           dim=1, keepdim=True))
            weight = self.sigmoid(ca_w * sa_w)
            info = weight * x[t * bs:(t + 1) * bs, ...]
            u = tau * u + info
            spike = spikefunc(u)
            u = u - spike
            o[t * bs:(t + 1) * bs, ...] = spike
        return o

    def extra_repr(self):
        return super().extra_repr() + ' attention spiking neuron, tau={}, Vth={}'.format(
            tau, Vth)


class LIF(nn.Module):
    """基础 LIF 脉冲神经元（无注意力）。"""

    def __init__(self):
        super(LIF, self).__init__()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
            u = u - spikefunc(u)
        return o

    def extra_repr(self):
        return 'LIF neuron, tau={}, Vth={}'.format(tau, Vth)


class AttSpikingBasicBlock(nn.Module):
    """脉冲残差注意力块：Conv-BN-ASN x2 + ECA(CA+SA) 注意力 + 残差。"""
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(AttSpikingBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.spike_func = ASN(channel=planes)
        self.spike_func2 = ASN(channel=planes)
        self.shortcut = nn.Sequential()

        t = int(abs((log(planes, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2),
                                 stride=1, bias=False)
        self.sa_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, stride=1,
                                 bias=False)
        self.sigmoid = nn.Sigmoid()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        out = self.spike_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        tmp = self.pool(out).transpose(-1, -2)
        ca_w = self.sigmoid(self.ca_conv(tmp).transpose(-1, -2))
        out = out * ca_w
        sa_w = self.sa_conv(torch.mean(out, dim=1, keepdim=True))
        sa_w = self.sigmoid(sa_w)
        out = out * sa_w

        out += self.shortcut(x)
        out = self.spike_func2(out)
        return out


class BLock_Layer(nn.Module):
    """由若干 block 组成的层，首 block 可选下采样。"""

    def __init__(self, block, in_planes, planes, num_block, downsample):
        super(BLock_Layer, self).__init__()
        layers = []
        if downsample:
            layers.append(block(in_planes, planes, 2))
        else:
            layers.append(block(in_planes, planes, 1))
        for _ in range(1, num_block):
            layers.append(block(planes, planes, 1))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class MRASNN(nn.Module):
    """Multi-scale Residual Attention SNN。

    forward(x): x 为 (TimeStep*B, L)（batch 已复制 TimeStep 次）
              -> logits (B, num_classes)
    """

    def __init__(self, num_classes=14):
        super(MRASNN, self).__init__()
        self.coding3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm1d(32),
            ASN(channel=32),
            nn.AvgPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm1d(64),
        )
        self.coding5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1,
                      padding=2),
            nn.BatchNorm1d(32),
            ASN(channel=32),
            nn.AvgPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1,
                      padding=2),
            nn.BatchNorm1d(64),
        )
        self.coding7 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1,
                      padding=3),
            nn.BatchNorm1d(32),
            ASN(channel=32),
            nn.AvgPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1,
                      padding=3),
            nn.BatchNorm1d(64),
        )
        self.sn = ASN(channel=192)
        self.layer2 = BLock_Layer(AttSpikingBasicBlock, 192, 256, 1, True)
        self.layer3 = BLock_Layer(AttSpikingBasicBlock, 256, 512, 1, True)
        self.adpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_classes, bias=False)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        o3 = self.coding3(x.unsqueeze(1))
        o5 = self.coding5(x.unsqueeze(1))
        o7 = self.coding7(x.unsqueeze(1))
        x = torch.cat((o3, o5, o7), dim=1)

        y = self.adpool(x)
        y = self.eca_conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = torch.sigmoid(y)
        x = x * y.expand_as(x)

        out = self.sn(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.adpool(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)

        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o += out[t * bs:(t + 1) * bs, ...]
        o /= TimeStep
        return o
