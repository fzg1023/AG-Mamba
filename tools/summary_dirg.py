#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 附加试验汇总（**不进论文**，只读 results_dirg/）。

输出：
  · stdout 表格
  · `logs/dirg_L<load>_ch<ch>_report.md`（同内容留档）

口径：主读数 last5；只统计完整协议单元（n_eval >= 60）；
      `T*` = 跨转速 LODO，`D*` = 同域参考（源域=目标域，单源域）。
用法：python tools/summary_dirg.py [--load 1000] [--ch 0]
"""
import argparse
import csv
import io
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEEDS = {0: 100, 1: 200, 2: 300, 3: 400, 4: 500}


def load_rows(load, ch):
    p = os.path.join(ROOT, 'results_dirg', 'L%d_ch%d' % (load, ch), '_summary.csv')
    if not os.path.exists(p):
        raise SystemExit('缺 %s' % p)
    rows = [r for r in csv.DictReader(io.open(p, encoding='utf-8'))
            if int(r['n_eval'] or 0) >= 60]
    for r in rows:
        r['last5'] = float(r['last5'])
        r['seed'] = int(r['seed'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = load_rows(args.load, args.ch)
    methods = sorted({r['method'] for r in rows})
    methods = ['AGMamba'] + [m for m in methods if m != 'AGMamba']
    lodo = sorted({r['task'] for r in rows if r['kind'] == 'lodo'})
    indom = sorted({r['task'] for r in rows if r['kind'] == 'indomain'})
    seeds = sorted({r['seed'] for r in rows})

    L = []
    L.append('# DIRG 附加试验汇总（L%d N / Ch%d）' % (args.load, args.ch + 1))
    L.append('')
    L.append('- 域 = 名义转速 %s，**载荷固定 ~1000 N**；`T*` 跨转速 LODO，`D*` 同域参考'
             % ', '.join('%d Hz' % SPEEDS[i] for i in range(5)))
    L.append('- 主读数 `last5`；完整协议单元 n_eval>=60；种子 %s（共 %d 个）'
             % (seeds, len(seeds)))
    L.append('- 单元数：%d（跨转速 %d + 同域 %d）'
             % (len(rows), len([r for r in rows if r['kind'] == 'lodo']),
                len([r for r in rows if r['kind'] == 'indomain'])))
    L.append('')

    a = [r for r in rows if r['method'] == 'AGMamba']
    a_lodo = [r['last5'] for r in a if r['kind'] == 'lodo']
    a_ind = [r['last5'] for r in a if r['kind'] == 'indomain']
    if a_ind and a_lodo:
        L.append('## 1. 本方法：同域 vs 跨转速')
        L.append('')
        L.append('| | last5（均值 ± std） | n |')
        L.append('|---|---|---|')
        L.append('| 同域参考（源=目标） | **%.2f ± %.2f** | %d |'
                 % (np.mean(a_ind), np.std(a_ind), len(a_ind)))
        L.append('| 跨转速 LODO | **%.2f ± %.2f** | %d |'
                 % (np.mean(a_lodo), np.std(a_lodo), len(a_lodo)))
        L.append('| **落差（同域 − 跨转速）** | **%.2f** | |'
                 % (np.mean(a_ind) - np.mean(a_lodo)))
        L.append('')
        L.append('逐域（同域参考）：' + '，'.join(
            '%d Hz %.2f' % (SPEEDS[int(t[1:])], np.mean(
                [r['last5'] for r in a if r['task'] == t]))
            for t in indom))
        L.append('')

    L.append('## 2. 跨转速 LODO：逐方法 / 逐任务（last5，跨种子均值）')
    L.append('')
    L.append('| 方法 | 池化(25) | ' + ' | '.join(
        'T%d→%dHz' % (int(t[1:]), SPEEDS[int(t[1:])]) for t in lodo) + ' |')
    L.append('|---' * (len(lodo) + 2) + '|')
    for m in methods:
        sel = [r for r in rows if r['method'] == m and r['kind'] == 'lodo']
        if not sel:
            continue
        pooled = np.mean([r['last5'] for r in sel])
        cells = []
        for t in lodo:
            v = [r['last5'] for r in sel if r['task'] == t]
            cells.append('%.2f' % np.mean(v) if v else '-')
        star = ' **←**' if m == 'AGMamba' else ''
        L.append('| %s | **%.2f** | %s |' % (m, pooled, ' | '.join(cells)))
        if star:
            L[-1] = L[-1][:-2] + star
    L.append('')

    L.append('## 3. 本方法 vs 各基线（按 (任务,种子) 严格配对，n=25）')
    L.append('')
    L.append('| 基线 | 基线池化 | Δ（本方法−基线） | wins | Wilcoxon p |')
    L.append('|---|---|---|---|---|')
    ref = {(r['task'], r['seed']): r['last5'] for r in a if r['kind'] == 'lodo'}
    for m in methods:
        if m == 'AGMamba':
            continue
        sel = {(r['task'], r['seed']): r['last5']
               for r in rows if r['method'] == m and r['kind'] == 'lodo'}
        keys = sorted(set(ref) & set(sel))
        if not keys:
            continue
        d = np.array([ref[k] - sel[k] for k in keys])
        p = wilcoxon(d).pvalue if np.any(d != 0) else float('nan')
        L.append('| %s | %.2f | **%+.2f** | %d-%d | %.4f |'
                 % (m, np.mean([sel[k] for k in keys]), d.mean(),
                    int((d > 0).sum()), len(d), p))
    L.append('')

    txt = '\n'.join(L) + '\n'
    out = args.out or os.path.join(ROOT, 'logs', 'dirg_L%d_ch%d_report.md'
                                   % (args.load, args.ch))
    io.open(out, 'w', encoding='utf-8', newline='\n').write(txt)
    print(txt)
    print('wrote ' + out)


if __name__ == '__main__':
    main()
