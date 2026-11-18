#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四口径完整排序表（各域 + cross，均值 ± 标准差）。

口径（与 `analysis/summary_compare.py` 完全一致，函数直接复用）
--------------------------------------------------------------
· last1    = 最后一个评估点（第 60 轮，末轮终值）
· last5    = 末 5 个评估点均值
· last10   = 末 10 个评估点均值  ← 协议主口径
· peak     = 60 个评估点的最大值（**注意**：逐单元取 max 会受单轮噪声上偏，
             偏置量级与方法相关 0.85–6.89 分，跨方法比较不公平；
             见 `_cmp_<DS>_stats.md` 的「peak 的口径偏置实测」节）

统计口径
--------
· 每个单元格 = 该 (方法, 臂, 任务) 下 **5 个种子**的 均值 ± 标准差（n=5）
· `cross`   = 4 个任务均值的**平均**；`±` 为**任务间标准差**（n=4），
              反映工况之间的离散度，不是种子噪声
· 排序      按 cross 降序；同 cross 时按方法与臂名

输出
----
    docs/manuscript/_cmp_<DS>_rankings.md

用法
----
    python analysis/summary_rankings.py PHM2009
    python analysis/summary_rankings.py PU
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)          # 让 import summary_compare 生效
sys.path.insert(0, ROOT)

# 复用主分析脚本的解析与归约，杜绝两处口径漂移
from summary_compare import (discover, read_series, reduce_series,  # noqa: E402
                             SEEDS, OURS)
from agmamba import paths as _P                                    # noqa: E402

READOUTS = [(5, 'last5', '末 5 个评估点均值 —— **协议主口径（进论文）**'),
            (1, 'last1', '最后一个评估点（第 60 轮）—— **部署口径（进论文）**'),
            (10, 'last10', '末 10 个评估点均值 —— **稳健性口径（进论文）**'),
            ('peak', 'peak', '60 个评估点的最大值 —— **不进论文，仅在本文件留档**')]


def ms(vals):
    """均值 ± 标准差（样本标准差 ddf=1）。"""
    if not vals:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def build(dataset, k):
    """{(method, arm): {'tasks': {t: (m, s)}, 'cross': (m, s), 'n': int}}"""
    per = {}
    for m, arm, t in discover(dataset):
        vals = []
        for sd in SEEDS:
            v = reduce_series(
                read_series(_P.log_path(dataset, t, m, sd, arm=arm)), k)
            if v is not None:
                vals.append(v)
        st = ms(vals)
        if st:
            per.setdefault((m, arm), {})[t] = st
    out = {}
    for key, td in per.items():
        if not td:
            continue
        task_means = [v[0] for _, v in sorted(td.items())]
        out[key] = {'tasks': td, 'cross': ms(task_means), 'n': len(td)}
    return out


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'PHM2009'
    tasks = sorted({t for _, _, t in discover(dataset)})

    L = ['# %s 完整排序（各域 + cross，均值 ± 标准差）—— 主口径 `last5`'
         % dataset, '',
         '由 `analysis/summary_rankings.py` 生成；解析与归约函数直接复用 '
         '`analysis/summary_compare.py`，保证与配对检验同一口径。', '',
         '- 单元格 = 该 (方法, 臂, 任务) 下 **5 个种子的均值 ± 种子间标准差**（n=5）',
         '- `cross` = 各任务均值的**平均**；其 `±` 为**任务间标准差**（n=%d），'
         '反映工况差异而非种子噪声' % len(tasks),
         '- 口径定义：`last5` = 末 5 个评估点均值（**主口径**）、`last1` = 末轮终值'
         '（**部署口径**）、`last10` = 末 10 个评估点均值（**稳健性口径**）'
         '—— 这三个进论文；`peak` = 60 个评估点最大值（**不进论文**：偏置随方法'
         '变化 0.85–6.89 分，跨方法不可比；偏置实测见 `_cmp_%s_stats.md`）' % dataset,
         '']

    for k, name, desc in READOUTS:
        per = build(dataset, k)
        if not per:
            continue
        board = sorted(per.items(),
                       key=lambda kv: (-kv[1]['cross'][0], kv[0][0], kv[0][1]))
        L += ['## 口径 `%s` —— %s' % (name, desc), '',
              '| 排名 | 方法 | 臂 | cross | '
              + ' | '.join(tasks) + ' |',
              '|---|---|---|---|' + '---|' * len(tasks)]
        for i, ((m, arm), d) in enumerate(board, 1):
            cells = []
            for t in tasks:
                v = d['tasks'].get(t)
                cells.append('%.2f ± %.2f' % v if v else '—')
            L.append('| %d | %s | %s | **%.2f ± %.2f** | %s |'
                     % (i, m, arm, d['cross'][0], d['cross'][1],
                        ' | '.join(cells)))
        L.append('')

    # 排名漂移总览：各口径下每个 (方法, 臂) 的名次
    ranks = {}
    for k, name, _ in READOUTS:
        per = build(dataset, k)
        if not per:
            continue
        board = sorted(per.items(),
                       key=lambda kv: (-kv[1]['cross'][0], kv[0][0], kv[0][1]))
        for i, ((m, arm), d) in enumerate(board, 1):
            ranks.setdefault((m, arm), {})[name] = (
                i, d['cross'][0], d['cross'][1])
    L += ['## 排名漂移总览（cross 按各口径）', '',
          '| 方法 | 臂 | ' + ' | '.join('`%s`' % r[1] for r in READOUTS)
          + ' | 最好名次 | 最差名次 |',
          '|---|---|' + '---|' * (len(READOUTS) + 2)]
    rows = []
    for (m, arm), d in ranks.items():
        rk = [d.get(r[1]) for r in READOUTS]
        best = min([x[0] for x in rk if x]) if any(rk) else None
        worst = max([x[0] for x in rk if x]) if any(rk) else None
        rows.append((best or 99, m, arm, rk, best, worst))
    rows.sort()
    for _, m, arm, rk, best, worst in rows:
        cells = []
        for x in rk:
            cells.append('#%d (%.2f ± %.2f)' % x if x
                         else '—')
        L.append('| %s | %s | %s | %d | %d |'
                 % (m, arm, ' | '.join(cells), best, worst))
    L.append('')

    out = os.environ.get(
        'AGM_RANKINGS_OUT',
        os.path.join(ROOT, 'docs', 'manuscript',
                     '_cmp_%s_rankings.md' % dataset))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # 显式 utf-8：内容含 U+2212（−）/U+00B1（±），Windows 默认编码会报错
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\n[written] %s' % out)


if __name__ == '__main__':
    main()
