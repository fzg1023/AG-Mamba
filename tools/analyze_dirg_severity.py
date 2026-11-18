#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 低分归因：是"同一位置、不同严重程度"造成的吗？

做法（LDA 代理，不训练神经网络；特征与 pick_dirg_channel 同源）：
  逐窗 z-score -> 1024 点功率谱 -> 10log10 -> 128 个等宽归一化频带
对每个**有序转速对**（训练 i -> 测试 j，i!=j）用同一套特征分别做 4 个任务：
  (1) 7 类       0A 1A 2A 3A 4A 5A 6A
  (2) 3 组       健康 / 内圈(1A,2A,3A) / 滚动体(4A,5A,6A)     <- 把"程度"合并
  (3) 内圈程度   只用 1A/2A/3A 三分类
  (4) 滚动体程度 只用 4A/5A/6A 三分类

读法：若 (2) 远高于 (1)，说明错误大量发生在"同位置不同严重程度"之间
      （即"程度"在转速偏移下不可分），这就是对该假说的直接检验。

用法：python tools/analyze_dirg_severity.py --ch 0 --fs 51200
      python tools/analyze_dirg_severity.py --ch 0 --fs 25600
"""
import argparse
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.DIRG_bearing import (CLASSES, SPEEDS, available_domains,  # noqa: E402
                                   data_load)

SIG, NBIN = 1024, 128
GROUP_OF = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}   # 0 健康 / 1 内圈 / 2 滚动体
GNAME = {0: 'healthy', 1: 'inner', 2: 'roller'}


def feat(win):
    x = np.asarray(win, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    P = np.abs(np.fft.rfft(x, n=SIG, axis=1)) ** 2
    nb = P.shape[1] // NBIN
    P = P[:, :NBIN * nb].reshape(P.shape[0], NBIN, nb).mean(axis=2)
    return 10.0 * np.log10(np.maximum(P, 1e-12))


def load_domains(load, ch, fs, nwin):
    ids, sps = available_domains(load, ch=ch, fs=fs)
    X, y = {}, {}
    for d in ids:
        ld = data_load(d, load, nwin, mode='head', ch=ch, fs=fs)
        X[d] = feat(np.asarray(ld[0])[:, :, 0])
        y[d] = np.asarray(ld[1])
    return ids, sps, X, y


def run(load, ch, fs, nwin):
    ids, sps, X, y = load_domains(load, ch, fs, nwin)
    a7, a3g, ain, aro = [], [], [], []
    CM = np.zeros((7, 7), dtype=float)
    per_tgt = {}
    for i in ids:
        m7 = LDA().fit(X[i], y[i])
        gi = np.array([GROUP_OF[c] for c in y[i]])
        mg = LDA().fit(X[i], gi)
        sel_i = np.isin(y[i], [1, 2, 3])
        sel_i2 = np.isin(y[i], [4, 5, 6])
        mi = LDA().fit(X[i][sel_i], y[i][sel_i] - 1)
        mr = LDA().fit(X[i][sel_i2], y[i][sel_i2] - 4)
        for j in ids:
            if i == j:
                continue
            gj = np.array([GROUP_OF[c] for c in y[j]])
            a7.append(m7.score(X[j], y[j]))
            a3g.append(mg.score(X[j], gj))
            sj = np.isin(y[j], [1, 2, 3])
            sj2 = np.isin(y[j], [4, 5, 6])
            ain.append(mi.score(X[j][sj], y[j][sj] - 1))
            aro.append(mr.score(X[j][sj2], y[j][sj2] - 4))
            pred = m7.predict(X[j])
            for t, p in zip(y[j], pred):
                CM[t, p] += 1
            per_tgt.setdefault(j, []).append(m7.score(X[j], y[j]))
    return dict(ids=ids, sps=sps, a7=np.array(a7), a3g=np.array(a3g),
                ain=np.array(ain), aro=np.array(aro), CM=CM, per_tgt=per_tgt)


def show(tag, r, load, ch):
    a7, a3g, ain, aro = r['a7'], r['a3g'], r['ain'], r['aro']
    print('=' * 74)
    print('%s  (load=L%d, ch=%d, 每(类,转速) 50 窗；有序转速对 n=%d)'
          % (tag, load, ch + 1, len(a7)))
    print('=' * 74)
    print('%-14s %8s %8s %8s %8s %7s' % ('task', 'mean', 'std', 'n_class', 'chance', 'chance%'))
    for name, arr, ncls in [('7 classes', a7, 7), ('3 groups', a3g, 3),
                            ('inner severity', ain, 3), ('roller severity', aro, 3)]:
        print('%-14s %8.3f %8.3f %8d %8.3f %8.1f%%'
              % (name, arr.mean(), arr.std(), ncls, 1.0 / ncls, 100.0 / ncls))
    print()
    print('gap  (3 groups - 7 classes)      = %+.3f  <- 合并"程度"能挽回多少' % (a3g.mean() - a7.mean()))
    print('gap  (7 classes - inner sev)     = %+.3f' % (a7.mean() - ain.mean()))
    print('gap  (7 classes - roller sev)    = %+.3f' % (a7.mean() - aro.mean()))
    print()
    CM = r['CM']
    tot = CM.sum()
    err = tot - np.trace(CM)
    same = 0.0
    for t in range(7):
        for p in range(7):
            if t != p and GROUP_OF[t] == GROUP_OF[p]:
                same += CM[t, p]
    print('confusion over all ordered pairs: total=%d  errors=%d  same-group errors=%.0f (%.1f%% of errors)'
          % (tot, err, same, 100.0 * same / max(err, 1)))
    print()
    print('row-normalised confusion (%):  rows = true class')
    print('        ' + ''.join('%8s' % c for c in CLASSES) + '     acc')
    for t in range(7):
        row = CM[t] / max(CM[t].sum(), 1) * 100.0
        print('%-8s' % CLASSES[t] + ''.join('%8.1f' % v for v in row) + '%8.1f' % row[t])
    print()
    print('per-target (7 classes):')
    for j in sorted(r['per_tgt']):
        print('   -> %-6s  acc=%.3f' % (SPEEDS[j], np.mean(r['per_tgt'][j])))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--fs', type=int, default=51200)
    ap.add_argument('--win', type=int, default=50)
    args = ap.parse_args()
    r = run(args.load, args.ch, args.fs, args.win)
    show('fs=%d Hz (window %.1f ms)' % (args.fs, 1024.0 / args.fs * 1000.0), r,
         args.load, args.ch)


if __name__ == '__main__':
    main()
