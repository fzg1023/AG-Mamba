#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读数窗口决策分析：last-k 的 k 该取多少（last5 还是 last10）。

背景
----
协议固定 60 个评估点（每 epoch 一个）。主口径在"末 k 个评估点均值"里选。
本脚本用已完成的 PHM2009 / PU（21 方法 × 4 任务 × 5 种子）回答三个问题：

  Q1 降噪：k 增大能压掉多少种子间噪声？（统计功效）
  Q2 结论稳定性：核心结论（本方法 vs 20 个基线的显著性与名次）在哪个 k 之后不再变？
  Q3 偏差：末 k 均值偏离"真正要交付的模型"（= 末个评估点）多远？这个偏差是否随方法不同？

判据（本脚本采用的决策规则）
--------------------------
  末个评估点是**协议真正的产出**（部署的就是这个模型），故 last1 是估计目标；
  末 k 均值是对"随机停在末段某一轮"的期望性能，作用是**不让结论悬于恰好第 60 轮**。
  → 因此应取"能让结论稳定下来的**最小** k"：更小的 k 更贴近可交付模型，更大的 k 才有降噪。
  若 k=5 与 k=10 给出相同的显著性与名次，则主口径取 last5，last10 作为稳健性列。

输出
----
    docs/manuscript/_readout_window_study.md

用法
----
    python analysis/readout_window_study.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, ROOT)

from summary_compare import (discover, read_series,  # noqa: E402
                             SEEDS, OURS, paired)
from agmamba import paths as _P                     # noqa: E402

DATASETS = ['PHM2009', 'PU']
KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]


# ---------------------------------------------------------------- 读取
def load(dataset):
    """{(method, arm, task, seed): 完整评估点序列}"""
    ser = {}
    for m, arm, t in discover(dataset):
        for sd in SEEDS:
            v = read_series(_P.log_path(dataset, t, m, sd, arm=arm))
            if v:
                ser[(m, arm, t, sd)] = v
    return ser


def stat(vals, k):
    """末 k 个评估点均值；'peak' 则取最大值。"""
    if not vals:
        return None
    if k == 'peak':
        return float(max(vals))
    return float(np.mean(vals[-k:])) if len(vals) >= k else None


def matrix(ser, k):
    """(method, arm) -> {(task, seed): 统计量}"""
    mat = {}
    for (m, arm, t, sd), vals in ser.items():
        v = stat(vals, k)
        if v is not None:
            mat.setdefault((m, arm), {})[(t, sd)] = v
    return mat


def rank_key(mat):
    """按跨任务均值排序，返回 [(name, mean), ...] 降序。"""
    board = []
    for key, td in mat.items():
        if td:
            board.append((key, float(np.mean(list(td.values())))))
    board.sort(key=lambda x: -x[1])
    return board


def spearman(a, b):
    """两个 dict{key: 值} 的 Spearman 相关（只取交集）。"""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float('nan')
    ra = _rank([a[k] for k in keys])
    rb = _rank([b[k] for k in keys])
    ra = np.array(ra, dtype=float)
    rb = np.array(rb, dtype=float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float('nan')


def _rank(x):
    """平均秩（处理并列）。"""
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


# ---------------------------------------------------------------- 主流程
def main():
    L = ['# 读数窗口决策分析：last-k 的 k 该取多少', '',
         '由 `analysis/readout_window_study.py` 基于已完成的 PHM2009 / PU 生成'
         '（21 方法 × 4 任务 × 5 种子）。', '',
         '**判据**：末个评估点是协议真正的产出（要部署的就是这个模型），故 `last1` 是估计目标；'
         '末 k 均值的作用是"不让结论悬于恰好第 60 轮"。因此应取'
         '**能让结论稳定下来的最小 k** —— 更小的 k 更贴近可交付模型，更大的 k 只是多降噪。', '',
         '## 结论（已采纳，2026-09-30）', '',
         '**主口径取 `k = 5`**：它是"结论已稳定"的最小窗口（k ≤ 3 在 PU 上显著数 17→19→18→19 **非单调**，'
         'CCN 在 k=1/3 不显著而 k=2/4 显著），且与可交付模型的偏离小于 k=10。'
         '论文正文**并列报三个口径**：`last5`（主）+ `last1`（部署口径）+ `last10`（稳健性）；'
         '`peak` **不进论文**（偏置随方法变化 0.85–6.89 分，跨方法不可比）。'
         '完整依据见下方 Q1/Q2/Q3 与 Q2b 四张表。', '',
         '**诚实披露**：更大的窗口对本文方法**更有利** —— PU 上 k ≥ 12 可得 20/20 显著、Δ+3.39，'
         '而 k=5 只有 19/20、Δ+2.80；最严格的部署口径 `last1` 下 PU 仅 **17/20**'
         '（CCN、CDDG、Mixup 未达显著）。选 k=5 是主动放弃更有利的读数。', '']

    allrows = {}
    for ds in DATASETS:
        ser = load(ds)
        if not ser:
            continue
        tasks = sorted({t for _, _, t, _ in ser})
        boards = {k: matrix(ser, k) for k in KS}

        # ---------- Q1 降噪：种子间标准差 vs k ----------
        L += ['---', '', '## %s — Q1 降噪：种子间标准差 vs k' % ds, '',
              '对每个 (方法, 臂, 任务) 求 5 个种子的样本标准差，再对所有格子取中位数与均值。', '',
              '| k | 种子间 std 中位数 | 种子间 std 均值 | 相对 k=1 的降幅 |',
              '|---|---|---|---|']
        base_med = None
        for k in KS:
            sds = []
            for (m, arm), td in boards[k].items():
                for t in tasks:
                    vals = [td[(t, sd)] for sd in SEEDS if (t, sd) in td]
                    if len(vals) > 1:
                        sds.append(float(np.std(vals, ddof=1)))
            if not sds:
                continue
            med = float(np.median(sds))
            if base_med is None:
                base_med = med
            L.append('| %d | %.2f | %.2f | %s |'
                     % (k, med, float(np.mean(sds)),
                        '—' if k == 1 else '%.0f%%' % (100 * (1 - med / base_med))))
        L.append('')

        # ---------- Q2 结论稳定性 ----------
        ours = {k: boards[k].get((OURS, 'unif')) for k in KS}
        methods = sorted({m for m, a in boards[KS[0]] if m != OURS})
        L += ['## %s — Q2 结论稳定性：显著性计数与 Δ vs k' % ds, '',
              '「显著数」= 本方法 unif 臂相对 20 个基线中双侧 Wilcoxon p<0.05 的个数'
              '（同 (task, seed) 配对，n=%d）。' % (len(tasks) * len(SEEDS)), '',
              '| k | 显著数 /20 | 本方法 cross | 最强基线 | Δ vs 最强基线 | 排序 Spearman(vs k=10) | 未显著的基线 |',
              '|---|---|---|---|---|---|---|']
        # Spearman 参照取 k=10 的跨任务均值（在循环外算，才能覆盖小 k）
        ref = {key: float(np.mean(list(td.values())))
               for key, td in boards[10].items() if td}
        for k in KS:
            o = ours[k]
            if not o:
                continue
            sig = 0
            fails = []
            best_base, best_val = None, -1e9
            for m in methods:
                b = boards[k].get((m, 'unif')) or boards[k].get((m, 'pub'))
                if not b:
                    continue
                n, md, p, _, _ = paired(o, b)
                if n and p == p and p < 0.05 and md > 0:
                    sig += 1
                else:
                    fails.append(m)
                bv = float(np.mean(list(b.values())))
                if bv > best_val:
                    best_base, best_val = m, bv
            bd = {key: float(np.mean(list(td.values())))
                  for key, td in boards[k].items() if td}
            L.append('| %d | **%d** | %.2f | %s %.2f | %+.2f | %.4f | %s |'
                     % (k, sig, float(np.mean(list(o.values()))), best_base,
                        best_val, float(np.mean(list(o.values()))) - best_val,
                        spearman(ref, bd),
                        ', '.join(fails) if fails else '—'))
        L.append('')

        # ---------- Q3 偏差：末 k 均值 − 末个评估点 ----------
        L += ['## %s — Q3 偏差：末 k 均值偏离可交付模型（末个评估点）的量' % ds, '',
              '对每个单元算 `末 k 均值 − 末个评估点`，再按方法分组求均值。'
              '若该偏差**随方法变化**，则长窗口不只是加噪声，而会改变方法间比较。', '',
              '| k | 全体平均 (偏差) | 本方法 (偏差) | 基线平均 (偏差) | 全体 平均\\|偏差\\| |',
              '|---|---|---|---|---|']
        for k in KS:
            if k == 1:
                L.append('| 1 | 0.00 | 0.00 | 0.00 | 0.00 |')
                continue
            ours_d, base_d, all_d = [], [], []
            for (m, arm, t, sd), vals in ser.items():
                if len(vals) < k:
                    continue
                d = float(np.mean(vals[-k:])) - float(vals[-1])
                all_d.append(d)
                (ours_d if m == OURS else base_d).append(d)
            L.append('| %d | %+.2f | %+.2f | %+.2f | %.2f |'
                     % (k, float(np.mean(all_d)), float(np.mean(ours_d)),
                        float(np.mean(base_d)), float(np.mean(np.abs(all_d)))))
        L.append('')

        # ---------- Q2b 逐基线 Δ vs k（看结论是否随 k 漂移） ----------
        watch = ['Mixup', 'WIDAN', 'CDDG', 'DANN', 'MRA_SNN', 'UDTL']
        L += ['## %s — Q2b 关键基线的 Δ 随 k 的漂移' % ds, '',
              '若 Δ 随 k 单调漂移，说明窗口选择会实质改变结论（不只是加噪声）。', '',
              '| 基线 | ' + ' | '.join('k=%d' % k for k in KS) + ' | 漂移(最大−最小) |',
              '|---|' + '---|' * (len(KS) + 1)]
        for m in watch:
            cells, ds_ = [], []
            for k in KS:
                o, b = ours[k], (boards[k].get((m, 'unif'))
                                 or boards[k].get((m, 'pub')))
                if o and b:
                    _, md, _, _, _ = paired(o, b)
                    cells.append('%+.2f' % md)
                    ds_.append(md)
                else:
                    cells.append('—')
            L.append('| %s | %s | %.2f |'
                     % (m, ' | '.join(cells),
                        (max(ds_) - min(ds_)) if ds_ else float('nan')))
        L.append('')
        allrows[ds] = boards

    out = os.path.join(ROOT, 'docs', 'manuscript', '_readout_window_study.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\n[written] %s' % out)


if __name__ == '__main__':
    main()
