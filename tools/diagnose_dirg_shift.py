#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 跨转速失效的机制定位（LDA 代理，CPU，几分钟）。

把"低分"拆成几个可互相甄别的对照：
  (0) 同域同段   ：head 训练 -> 同域 head 测试        （上界，应 ~1.0）
  (A) 同域异段   ：head 训练 -> 同域 **tail** 测试    （控制变量：train/test 分段本身的影响）
  (B) 异域同段   ：head[i] 训练 -> head[j] 测试       （纯转速偏移，位置与 A 相同）
  (C) 异域 + oracle 域均值对齐（减各自均值）
  (C2)异域 + oracle CORAL（均值 + 协方差重着色）
  (D) 特征方差分解：类间 / 转速间 / 残差 各占多少
  (E) 分频带：哪个频带还带得动跨转速的类信息

读法：
  · 若 A << 0，说明"低分"有一部分是**分段伪影**（记录内非平稳），不是转速；
  · 若 C/C2 明显抬高 B，说明偏移主要是**特征空间的平移/二阶偏移**，本方法没做对齐就吃亏；
  · 若 C/C2 也救不动，说明类结构本身被重新组织（不是简单对齐能解决）；
  · D 告诉你特征空间到底按"类"还是按"转速"组织。

用法：python tools/diagnose_dirg_shift.py --ch 0 --fs 51200
"""
import argparse
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from datasets.DIRG_bearing import SPEEDS, available_domains, data_load  # noqa: E402

SIG, NBIN = 1024, 128


def feat(win):
    x = np.asarray(win, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    P = np.abs(np.fft.rfft(x, n=SIG, axis=1)) ** 2
    nb = P.shape[1] // NBIN
    P = P[:, :NBIN * nb].reshape(P.shape[0], NBIN, nb).mean(axis=2)
    return 10.0 * np.log10(np.maximum(P, 1e-12))


def grab(d, load, ch, fs, nwin, mode):
    ld = data_load(d, load, nwin, mode=mode, ch=ch, fs=fs)
    return feat(np.asarray(ld[0])[:, :, 0]), np.asarray(ld[1])


def coral(Xs, Xt):
    """把 Xt 对齐到 Xs 的均值/协方差（oracle，仅作诊断上界）。"""
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs.T) + np.eye(Xs.shape[1]) * 1e-6
    Ct = np.cov(Xt.T) + np.eye(Xt.shape[1]) * 1e-6
    def whiten(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    A = whiten(Cs) @ whiten(Ct)
    return (Xt - mt) @ A.T + ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--fs', type=int, default=51200)
    ap.add_argument('--win', type=int, default=50)
    args = ap.parse_args()

    ids, sps = available_domains(args.load, ch=args.ch, fs=args.fs)
    H, T, Y = {}, {}, {}
    for d in ids:
        H[d], Y[d] = grab(d, args.load, args.ch, args.fs, args.win, 'head')
        T[d], _ = grab(d, args.load, args.ch, args.fs, args.win, 'tail')
    print('fs=%d  domains=%s  每(类,域) %d 窗  特征 %d 维'
          % (args.fs, [SPEEDS[d] for d in ids], args.win, H[ids[0]].shape[1]))
    print('=' * 76)

    # (0) 同域同段
    a0 = [LDA().fit(H[d], Y[d]).score(H[d], Y[d]) for d in ids]
    # (A) 同域异段
    aA = [LDA().fit(H[d], Y[d]).score(T[d], Y[d]) for d in ids]
    # (B) 异域同段
    aB, aC, aC2 = [], [], []
    for i in ids:
        m = LDA().fit(H[i], Y[i])
        for j in ids:
            if i == j:
                continue
            aB.append(m.score(H[j], Y[j]))
            # (C) oracle 均值对齐
            aC.append(m.score(H[j] - H[j].mean(0) + H[i].mean(0), Y[j]))
            # (C2) oracle CORAL
            aC2.append(m.score(coral(H[i], H[j]), Y[j]))
    print('(0) same domain, same segment  (head->head)  : %.3f' % np.mean(a0))
    print('(A) same domain, other segment (head->tail)  : %.3f   <- 分段伪影上界'
          % np.mean(aA))
    print('(B) cross speed, same segment  (head->head)  : %.3f' % np.mean(aB))
    print('(C) cross speed + oracle mean align          : %.3f' % np.mean(aC))
    print('(C2)cross speed + oracle CORAL               : %.3f' % np.mean(aC2))
    print('    chance = %.3f' % (1.0 / 7))
    print()

    # (D) 方差分解（用 head 样本）
    X = np.concatenate([H[d] for d in ids])
    y = np.concatenate([Y[d] for d in ids])
    dom = np.concatenate([np.full(len(H[d]), d) for d in ids])
    mu = X.mean(0)
    mu_cd = {}
    for c in range(7):
        for d in ids:
            mu_cd[(c, d)] = X[(y == c) & (dom == d)].mean(0)
    mu_d = {d: X[dom == d].mean(0) for d in ids}
    mu_c = {c: X[y == c].mean(0) for c in range(7)}
    S_class = np.mean([((mu_cd[(c, d)] - mu_d[d]) ** 2).sum()
                       for c in range(7) for d in ids])
    S_speed = np.mean([((mu_cd[(c, d)] - mu_c[c]) ** 2).sum()
                       for c in range(7) for d in ids])
    S_res = np.mean([((X[(y == c) & (dom == d)] - mu_cd[(c, d)]) ** 2).sum()
                     for c in range(7) for d in ids])
    tot = S_class + S_speed + S_res
    print('(D) feature-variance decomposition:')
    print('    between-class (within speed)  %8.1f  %5.1f%%'
          % (S_class, 100 * S_class / tot))
    print('    between-speed (within class)  %8.1f  %5.1f%%   <- 越大越"按转速组织"'
          % (S_speed, 100 * S_speed / tot))
    print('    residual (within class+speed) %8.1f  %5.1f%%'
          % (S_res, 100 * S_res / tot))
    print()

    # (E) 分频带（band -> 跨域 LDA）
    band_hz = [(0, 500), (500, 1000), (1000, 2000), (2000, 6400)]
    bw = (args.fs / 2.0) / NBIN
    print('(E) cross-speed LDA per band (bin width %.0f Hz):' % bw)
    for lo, hi in band_hz:
        b0 = max(0, int(lo / bw))
        b1 = min(NBIN, int(np.ceil(hi / bw)))
        if b1 - b0 < 4:
            print('    %5d-%-5d Hz: 带太窄(%d bin)，跳过' % (lo, hi, b1 - b0))
            continue
        accs = []
        for i in ids:
            m = LDA().fit(H[i][:, b0:b1], Y[i])
            for j in ids:
                if i != j:
                    accs.append(m.score(H[j][:, b0:b1], Y[j]))
        ina = [LDA().fit(H[d][:, b0:b1], Y[d]).score(H[d][:, b0:b1], Y[d]) for d in ids]
        print('    %5d-%-5d Hz: cross=%.3f   same-domain=%.3f'
              % (lo, hi, np.mean(accs), np.mean(ina)))


if __name__ == '__main__':
    main()
