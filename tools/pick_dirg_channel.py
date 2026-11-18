#!/usr/bin/python
# -*- coding:utf-8 -*-
"""挑 DIRG 的"区分度最大"通道（**用实测，不靠猜**）。

对 6 个通道各算三个量（L1000 载荷档，7 类 × 5 转速）：
  1. 同域 CV 精度   —— 每个转速内 5 折交叉验证的 LDA 精度（"这通道有没有类信息"）
  2. **跨转速迁移** —— 用转速 i 训练、转速 j 测试，对所有有序对取平均（**本试验真正关心的量**）
  3. Fisher 比      —— 类间方差 / 类内方差（与分类器无关的分离度）

特征：逐窗 z-score → 1024 点功率谱 → 10log10 → 归并成 128 个 100 Hz 带（0–12.8 kHz）。
样本：每（类·转速）取记录**前 50 窗**（与训练集同源）。

用法：python tools/pick_dirg_channel.py [--load 1000] [--win 50]
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'DIRG_bearing', 'VariableSpeedAndLoad')
SIG, FS = 1024, 51200.0
NBIN, BINHZ = 128, 100.0          # 128 带 × 100 Hz = 0–12.8 kHz
LOADS = {'L1000': (400, 520), 'L1400': (650, 760), 'L1800': (850, 930)}
CLASSES = ['0A', '1A', '2A', '3A', '4A', '5A', '6A']
SPEEDS = [100, 200, 300, 400, 500]
NAME_RE = re.compile(r'^(C\dA)_(\d{3})_(\d{3})_(\d)\.mat$')


def index(load):
    lo, hi = LOADS['L%s' % load]
    idx = {}
    for p in glob.glob(os.path.join(SRC, '*.mat')):
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        cls, sp, vvv = m.group(1)[1:], int(m.group(2)), int(m.group(3))
        if lo <= vvv <= hi:
            idx[(cls, sp)] = p
    return idx


def feats(x, dc_frac=0.3):
    """x: (n, 1024) 原始窗 -> (n, NBIN) 归并后的 log 功率谱。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    x = x / np.maximum(sd, 1e-12)
    P = np.abs(np.fft.rfft(x, n=SIG, axis=1)) ** 2
    P = P[:, :NBIN * int(BINHZ * SIG / FS)]          # 只取到 12.8 kHz
    P = P.reshape(P.shape[0], NBIN, -1).mean(axis=2)
    return 10.0 * np.log10(np.maximum(P, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--win', type=int, default=50, help='每(类·转速)用的窗数')
    args = ap.parse_args()

    idx = index(args.load)
    speeds = [s for s in SPEEDS if all((c, s) in idx for c in CLASSES)]
    print('载荷 L%d：%d 个转速 %s，%d 类，每(类·转速) %d 窗'
          % (args.load, len(speeds), speeds, len(CLASSES), args.win))

    rows = []
    for ch in range(6):
        X = {s: [] for s in speeds}
        y = {s: [] for s in speeds}
        rms = []
        for s in speeds:
            for lab, c in enumerate(CLASSES):
                M = loadmat(idx[(c, s)])
                k = [z for z in M if not z.startswith('__')][0]
                A = np.asarray(M[k], dtype=np.float64)
                x = A[:args.win * SIG, ch]
                rms.append(float(np.sqrt(np.mean(x ** 2))))
                X[s].append(feats(x.reshape(args.win, SIG)))
                y[s].append(np.full(args.win, lab))
        X = {s: np.concatenate(X[s]) for s in speeds}
        y = {s: np.concatenate(y[s]) for s in speeds}

        # 1) 同域 CV（池化全部转速）
        Xa = np.concatenate([X[s] for s in speeds])
        ya = np.concatenate([y[s] for s in speeds])
        cv = float(np.mean(cross_val_score(LDA(), Xa, ya, cv=5)))

        # 2) 跨转速迁移（训练 i -> 测试 j，所有有序对）
        accs = []
        for i in speeds:
            clf = LDA().fit(X[i], y[i])
            for j in speeds:
                if i == j:
                    continue
                accs.append(float(clf.score(X[j], y[j])))

        # 3) Fisher 比（池化特征上）
        mu, mua = [], Xa.mean(axis=0)
        for c in range(len(CLASSES)):
            mu.append(Xa[ya == c].mean(axis=0))
        mu = np.asarray(mu)
        Sb = np.var(mu, axis=0).mean()
        Sw = np.mean([np.var(Xa[ya == c], axis=0).mean() for c in range(len(CLASSES))])
        rows.append((ch, cv, float(np.mean(accs)), float(np.std(accs)),
                     Sb / max(Sw, 1e-12) / 100.0, float(np.mean(rms))))

    print('\n%-4s %-12s %-14s %-14s %-10s %s'
          % ('ch', '同域CV', '跨转速均值', '跨转速std', 'Fisher/100', 'RMS'))
    for ch, cv, m, sd, f, r in rows:
        star = ''
        if m == max(x[2] for x in rows):
            star = '   <== 跨转速最好'
        print('%-4s %-12.4f %-14.4f %-14.4f %-10.3f %.2f%s'
              % ('Ch%d' % (ch + 1), cv, m, sd, f, r, star))
    best = max(rows, key=lambda t: t[2])
    print('\n结论：跨转速区分度最大的是 **Ch%d**（0-based index = %d）'
          % (best[0] + 1, best[0]))
    print('换通道：python tools/prepare_dirg.py --ch %d' % best[0])


if __name__ == '__main__':
    main()
