#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
WIDAN（物理信息小波域适应网络）骨干与 JMMD 损失，
移植自 other/WIDAN-main/models/resnet18_1d.py 与 loss/JAN.py。

- Laplace_fast: Laplace 小波核生成（第一层卷积初始化，物理带通先验）
- ResNet18Features: conv1 宽核 64 小波初始化的 ResNet18-1D（输出 512）
- JMMD: 联合最大均值差异（特征层 5 核 + softmax 层 1 核的联合核）
"""
import math

import torch
from torch import nn

__all__ = ['ResNet18Features', 'jmmd_loss', 'Laplace_fast']


# ---------------------------------------------------------------------------
# Laplace 小波核（来自 WIDAN，frequency 为采样率 Hz）
# ---------------------------------------------------------------------------
class Laplace_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, frequency, eps=0.3,
                 mode='sigmoid'):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency
        self.a_ = torch.linspace(0, self.out_channels,
                                 self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels,
                                 self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1,
                                        steps=int(self.kernel_size))

    def Laplace(self, p):
        w = 2 * torch.pi * self.fre
        q = torch.tensor(1 - pow(0.03, 2))
        if self.mode == 'vanilla':
            return ((1 / math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1))))
        if self.mode == 'maxmin':
            a = (1 / math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))
            return (a - a.min()) / (a.max() - a.min())
        if self.mode == 'sigmoid':
            return (1 / math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))
        if self.mode == 'softmax':
            return (1 / math.e) * torch.exp(torch.nn.functional.softmax(
                (-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)), dim=-1)) * (torch.sin(w * (p - 0.1)))
        if self.mode == 'tanh':
            return (1 / math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).tanh()) * (
                torch.sin(w * (p - 0.1)))
        if self.mode == 'atan':
            return (1 / math.e) * torch.exp((2 / torch.pi) * (((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))).atan()) * (
                torch.sin(w * (p - 0.1)))
        raise ValueError('unknown mode')

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        return self.Laplace(p1).view(self.out_channels, 1, self.kernel_size)


# ---------------------------------------------------------------------------
# ResNet18-1D（WIDAN 版：conv1 kernel=64 小波初始化）
# ---------------------------------------------------------------------------
def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=1, wavelet_frequency=64000,
                 zero_init_residual=False, wavelet_init=True, widen=1.0):
        super(ResNet, self).__init__()

        def w(n):
            return max(8, int(round(n * widen)))

        self.inplanes = w(64)
        self.conv1 = nn.Conv1d(in_channel, w(64), kernel_size=64, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(w(64))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, w(64), layers[0])
        self.layer2 = self._make_layer(block, w(128), layers[1], stride=2)
        self.layer3 = self._make_layer(block, w(256), layers[2], stride=2)
        self.layer4 = self._make_layer(block, w(512), layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = w(512)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # wavelet_init=False 时保留上面的 kaiming_normal_ 随机初始化（消融用）
        if wavelet_init:
            self._initialize_weights(wavelet_frequency)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self, wavelet_frequency):
        # 第一层宽核卷积用 Laplace 小波核初始化（WIDAN 物理信息先验）
        with torch.no_grad():
            self.conv1.weight.data = Laplace_fast(
                out_channels=self.conv1.out_channels, kernel_size=64, eps=0.1,
                frequency=wavelet_frequency, mode='sigmoid').forward()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet18Features(nn.Module):
    def __init__(self, wavelet_frequency=64000, wavelet_init=True, widen=1.0):
        super(ResNet18Features, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channel=1,
                            wavelet_frequency=wavelet_frequency,
                            wavelet_init=wavelet_init, widen=widen)

    def forward(self, x):
        return self.model(x)

    def output_num(self):
        return self.model.out_dim


# ---------------------------------------------------------------------------
# JMMD 联合最大均值差异（来自 WIDAN loss/JAN.py）
# ---------------------------------------------------------------------------
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5,
                    fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def jmmd_loss(source_list, target_list, kernel_muls=(2.0, 2.0),
              kernel_nums=(5, 1), fix_sigma_list=(None, 1.68)):
    """source_list/target_list: 每层一个张量（特征层 + softmax 层）。"""
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        kernels = guassian_kernel(source_list[i], target_list[i],
                                  kernel_mul=kernel_muls[i],
                                  kernel_num=kernel_nums[i],
                                  fix_sigma=fix_sigma_list[i])
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    XX = joint_kernels[:batch_size, :batch_size]
    YY = joint_kernels[batch_size:, batch_size:]
    XY = joint_kernels[:batch_size, batch_size:]
    YX = joint_kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)
