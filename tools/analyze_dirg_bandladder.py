#!/usr/bin/python
# -*- coding:utf-8 -*-
# DIRG band-limited LDA ladder: where does the "defect location" information live,
# and how much of it survives a speed shift?
#
# Feature: per-window z-score -> 1024-pt power spectrum -> averaged into bands of
# `edge_hz` -> 10log10.  Then an LDA is trained on a *low-pass restricted* set of
# bands (centres <= cutoff) and tested on another speed.
#
# For every ordered speed pair we report SAME-speed (i->i) and CROSS-speed (i!=j)
# accuracy at each cutoff, so the gap isolates transferable vs domain-specific content.
#
# Usage: python tools/analyze_dirg_bandladder.py --ch 0 --fs 51200
import argparse
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.DIRG_bearing import CLASSES, SPEEDS, available_domains, data_load  # noqa: E402

SIG = 1024
GROUP_OF = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
GNAMES = ['healthy', 'inner', 'roller']


def feat_db(win, fs, edge_hz):
    x = np.asarray(win, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    P = np.abs(np.fft.rfft(x, n=SIG, axis=1)) ** 2
    fr = np.fft.rfftfreq(SIG, 1.0 / fs)
    edges = np.arange(0.0, fs / 2.0 + edge_hz, edge_hz)
    idx = np.clip(np.digitize(fr, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    Pb = np.zeros((P.shape[0], nb))
    for b in range(nb):
        m = idx == b
        if m.any():
            Pb[:, b] = P[:, m].mean(axis=1)
    cent = np.array([fr[idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)])
    return 10.0 * np.log10(np.maximum(Pb, 1e-12)), cent


def load(load, ch, fs, nwin, edge_hz):
    ids, sps = available_domains(load, ch=ch, fs=fs)
    X, y = {}, {}
    cent = None
    for d in ids:
        ld = data_load(d, load, nwin, mode='head', ch=ch, fs=fs)
        X[d], cent = feat_db(np.asarray(ld[0])[:, :, 0], fs, edge_hz)
        y[d] = np.asarray(ld[1])
    return ids, sps, X, y, cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--fs', type=int, default=51200)
    ap.add_argument('--win', type=int, default=50)
    ap.add_argument('--edge', type=float, default=200.0)
    args = ap.parse_args()

    ids, sps, X, y, cent = load(args.load, args.ch, args.fs, args.win, args.edge)
    nyq = args.fs / 2.0
    cut = [c for c in (1600, 3200, 6400, 12800, 25600) if c <= nyq]
    if nyq not in cut:
        cut.append(nyq)
    print('=' * 100)
    print('DIRG band-limited LDA ladder   fs=%d Hz (window %.1f ms)  load=L%d ch=%d'
          % (args.fs, 1024.0 / args.fs * 1000.0, args.load, args.ch + 1))
    print('domains=%s Hz   bands=%d x %.0f Hz   cutoffs=%s'
          % ([SPEEDS[i] for i in ids], len(cent), args.edge, cut))
    print('=' * 100)

    y3 = {d: np.array([GROUP_OF[c] for c in y[d]]) for d in ids}
    y2 = {d: (y[d] > 0).astype(int) for d in ids}

    for tname, yy, ncls in (('2cls faulty/healthy', y2, 2),
                            ('3group loc', y3, 3),
                            ('7class', y, 7)):
        print()
        print('--- %s  (chance=%.1f%%) ---' % (tname, 100.0 / ncls))
        print('%-10s %9s %9s %9s   %s' % ('cutoff', 'same', 'cross', 'gap', 'increment (cross)'))
        prev = None
        for k, c in enumerate(cut):
            m = cent <= c
            same, cross = [], []
            for i in ids:
                Xs, ys = X[i][:, m], yy[i]
                if len(np.unique(ys)) < 2:
                    continue
                mo = LDA().fit(Xs, ys)
                same.append(mo.score(Xs, ys))
                for j in ids:
                    if i != j:
                        cross.append(mo.score(X[j][:, m], yy[j]))
            s, cr = float(np.mean(same)), float(np.mean(cross))
            inc = '-' if prev is None else '%+.3f over [%d,%d] Hz' % (cr - prev, cut[k - 1], c)
            print('%-10s %8.3f%% %8.3f%% %8.3f%%   %s' % ('<=%d' % c, 100 * s, 100 * cr, 100 * (s - cr), inc))
            prev = cr

    print()
    print('--- 3-group row-normalised confusion (cross-speed, pooled) ---')
    for c in cut:
        m = cent <= c
        CM = np.zeros((3, 3))
        for i in ids:
            if len(np.unique(y3[i])) < 2:
                continue
            mo = LDA().fit(X[i][:, m], y3[i])
            for j in ids:
                if i != j:
                    for t, p in zip(y3[j], mo.predict(X[j][:, m])):
                        CM[t, p] += 1
        row = CM / np.maximum(CM.sum(axis=1, keepdims=True), 1) * 100.0
        print('  <=%-7d %s' % (c, '  '.join(
            '%s->%s %5.1f%%' % (GNAMES[a][:3], GNAMES[b][:3], row[a, b])
            for a in range(3) for b in range(3) if a != b)))


if __name__ == '__main__':
    main()
