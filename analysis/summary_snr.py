#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.6 SNR 噪声网格汇总 → `docs/manuscript/_s66_snr.md`。

数据源：`logs/_s66_snr.json`（由 `experiments/run_snr_grid.py` 逐单元写入，
每个单元 = 5 列 `last5`：clean / SNR 10 / 5 / 0 / −5 dB）。

口径
----
- **训练干净、测试加噪**（逐窗 AWGN，噪声标准差 = 该窗 RMS / 10^(SNR/20)）。
- 每单元 5 个 test loader 一次性交给 `train_model`，`result['acces']` 的第 0 列
  即**干净**目标（与主口径 `last5` 同定义）、后 4 列为各 SNR。
- 噪声实现只依赖 (dataset|task|seed, 窗序号, SNR)：三种方法看的是**同一组**含噪窗。
- 未完成的单元一律不进均值、并在表尾列出（**不用部分单元充数**）。

输出：docs/manuscript/_s66_snr.md（表 + 判定段）
用法：python analysis/summary_snr.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
SRC = os.path.join(ROOT, 'logs', '_s66_snr.json')
OUT = os.path.join(ROOT, 'docs', 'manuscript', '_s66_snr.md')

SNRS = ['clean', '10 dB', '5 dB', '0 dB', '-5 dB']
METHODS = ['AGMamba', 'WIDAN', 'MRA_SNN']
TASKS = ['T0', 'T4', 'T9']
TASK_HZ = {'T0': 20, 'T4': 40, 'T9': 80}
SEEDS = [0, 1, 2]
FLOOR = 2.55


def main():
    if not os.path.exists(SRC):
        raise SystemExit('missing %s (先跑 experiments/run_snr_grid.py)' % SRC)
    with open(SRC, encoding='utf-8') as f:
        raw = json.load(f)
    cell = {}
    for k, v in raw.items():
        ds, method, task, seed = k.split('|')
        cell[(method, task, int(seed))] = np.asarray(v, dtype=float)

    L = ['# §6.6 SNR 噪声网格（训练干净 / 测试加噪）—— 数值底稿', '',
         '由 `analysis/summary_snr.py` 汇总；数据 `logs/_s66_snr.json` ← '
         '`experiments/run_snr_grid.py`（启动器 `logs/run_snr_grid.sh`）。', '',
         '- 方法：`AGMamba`（本方法，交付配置）+ 本基准最强两个基线 '
         '`WIDAN` / `MRA_SNN`；目标 HUST `T0/T4/T9`；种子 0–2（每格 3 个）',
         '- 噪声：逐窗加性高斯白噪声，σ = 窗 RMS / 10^(SNR/20)（即按窗能量定义 SNR）；'
         '**训练数据始终干净**，只对目标测试集加噪',
         '- 噪声实现只依赖 (dataset|task|seed, 窗序号, SNR) ⇒ 三种方法评估'
         '**同一组含噪测试窗**，方法间差异不混入噪声抽样差异',
         '- 主口径 `last5`；第 0 列 clean 与主对比同定义，可直接与 §6.1/§6.7 对照',
         '- 未完成单元**不进均值**，只在表尾列出', '']

    # ---- 完成度 ----
    L += ['## (a) 完成度', '', '| 方法 | 目标 | 已完成种子 | 缺 |', '|---|---|---|---|']
    missing = []
    for m in METHODS:
        for t in TASKS:
            have = [s for s in SEEDS if (m, t, s) in cell]
            miss = [s for s in SEEDS if s not in have]
            if miss:
                missing.append('%s/%s/seed%s' % (m, t, ','.join(map(str, miss))))
            L.append('| `%s` | %s | %s | %s |'
                     % (m, t, ','.join(map(str, have)) if have else '—',
                        ','.join(map(str, miss)) if miss else '—'))
    L.append('')

    # ---- 主表 ----
    L += ['## (b) 主表（`last5`，每格 = 已完成种子的均值 ± std）', '',
          '| 方法 | 目标 (Hz) | clean | SNR 10 dB | SNR 5 dB | SNR 0 dB | SNR −5 dB |'
          ' Δ(0 dB − clean) |', '|---|---|---|---|---|---|---|---|']
    summary = defaultdict(dict)
    for m in METHODS:
        for t in TASKS:
            arr = [cell[(m, t, s)] for s in SEEDS if (m, t, s) in cell]
            if not arr:
                L.append('| `%s` | %s | – | – | – | – | – | – |'
                         % (m, TASK_HZ[t]))
                continue
            A = np.stack(arr)
            mu, sd = A.mean(0), (A.std(0, ddof=1) if len(A) > 1
                                 else np.zeros(A.shape[1]))
            summary[m][t] = mu
            # ⚠ 单元格内**不得**出现裸 `|`（会冲开 markdown 表列）→ 用 ' / ' 分隔
            # ⚠ 表头声明 **8 列**（方法 | 目标 | clean | 10 | 5 | 0 | −5 | Δ），
            # 故数据行也必须写 **8 个单元格**。早前版本把 5 个 SNR 值用 ' / '
            # 拼成一格、只写 4 列 → markdown 表列数与表头不匹配，下游
            # `make_fig9_efficiency.parse_snr`（要求 `len(c) == 8`）会把**所有**
            # 数据行丢弃，Fig. 8(d) 因此退回“not yet run”分支而静默画错。
            # 教训：**表头列数与数据行列数必须同源**，不要靠肉眼对齐。
            cells = ['%.2f ± %.2f' % (a, b) if len(A) > 1 else '%.2f' % a
                     for a, b in zip(mu, sd)]
            L.append('| ' + ' | '.join(['`%s`' % m, '%d' % TASK_HZ[t]]
                                       + cells
                                       + ['%+.2f' % (mu[3] - mu[0])])
                     + ' |')
    L.append('')

    # ---- 判定 ----
    L += ['## (c) 判定（按可复现下限 2.55 点）', '']
    for m in METHODS:
        if not summary[m]:
            continue
        clean = np.mean([summary[m][t][0] for t in TASKS if t in summary[m]])
        s0 = np.mean([summary[m][t][3] for t in TASKS if t in summary[m]])
        L.append('- `%s`：三目标池化 clean **%.2f** → SNR 0 dB **%.2f**'
                 '（Δ **%+.2f**）' % (m, clean, s0, s0 - clean))
    for t in TASKS:
        if all(t in summary[m] for m in METHODS):
            a = summary['AGMamba'][t][3]
            L.append('- %s（%d Hz）在 SNR 0 dB 下：本方法 **%.2f** vs WIDAN %.2f / '
                     'MRA_SNN %.2f（领先 **%+.2f** / **%+.2f**）'
                     % (t, TASK_HZ[t], a, summary['WIDAN'][t][3],
                        summary['MRA_SNN'][t][3], a - summary['WIDAN'][t][3],
                        a - summary['MRA_SNN'][t][3]))
    L.append('')
    if missing:
        L.append('> 未完成单元（不进均值）：%s' % '；'.join(missing))
        L.append('')
    L.append('> 判读模板：若 SNR 0 dB 下本方法仍≥两个基线，则写“在测试加噪下仍领先”；'
             '若差距缩到下限内，则写“**不主张**鲁棒性优势”，并说明本网格只覆盖测试加噪'
             '（训练仍干净）、只覆盖 HUST 三个目标、每格仅 3 个种子。')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote %s  (%d cells)' % (OUT, len(cell)))
    for k in sorted(cell):
        print('  %-24s %s' % (k, np.round(cell[k], 2)))


if __name__ == '__main__':
    main()
