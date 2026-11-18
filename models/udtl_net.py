#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
UDTL (Unsupervised Deep Transfer Learning) 骨干与多域判别器，
移植自 other/UDTL_multi_domain-main/models/。

- resnet18_features: ResNet18-1D 特征提取器（输出 512 维）
- AdversarialNet_multi: 梯度反转多域判别器（Step 退火系数）
- UDTLNet: 特征提取 + bottleneck(256) + 分类器 + 多域判别器 封装
"""
import numpy as np
import torch
from torch import nn

__all__ = ['UDTLNet']


# ---------------------------------------------------------------------------
# ResNet18-1D（来自 UDTL，去除 ImageNet 预训练下载）
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
    def __init__(self, block, layers, in_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

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
    """ResNet18-1D 特征提取器，输出 512 维。"""

    def __init__(self):
        super(ResNet18Features, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channel=1)

    def forward(self, x):
        return self.model(x)

    def output_num(self):
        return 512


# ---------------------------------------------------------------------------
# 梯度反转多域判别器（来自 UDTL models/AdversarialNet.py）
# ---------------------------------------------------------------------------
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
                 - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class AdversarialNetMulti(nn.Module):
    def __init__(self, in_feature, output_size, hidden_size, max_iter=10000.0,
                 trade_off_adversarial='Step', lam_adversarial=1.0):
        super(AdversarialNetMulti, self).__init__()
        self.ad_layer1 = nn.Sequential(nn.Linear(in_feature, hidden_size),
                                       nn.ReLU(inplace=True), nn.Dropout())
        self.ad_layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(inplace=True), nn.Dropout())
        self.ad_layer3 = nn.Linear(hidden_size, output_size)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        else:
            coeff = calc_coeff(self.iter_num, self.high, self.low,
                               self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        return y


# ---------------------------------------------------------------------------
# 封装
# ---------------------------------------------------------------------------
class UDTLNet(nn.Module):
    """特征提取 + bottleneck + 分类器 + 多域判别器（域对抗对齐源域间）。"""

    def __init__(self, num_classes=14, bottleneck_num=256, n_domains=3,
                 hidden_size=1024, max_iter=10000.0):
        super(UDTLNet, self).__init__()
        self.features = ResNet18Features()
        self.bottleneck = nn.Sequential(
            nn.Linear(512, bottleneck_num),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(bottleneck_num, num_classes)
        self.adv = AdversarialNetMulti(bottleneck_num, n_domains,
                                       hidden_size, max_iter=max_iter)

    def forward(self, x):
        feats = self.bottleneck(self.features(x))
        logits = self.classifier(feats)
        return feats, logits

    def adversarial(self, feats):
        return self.adv(feats)
