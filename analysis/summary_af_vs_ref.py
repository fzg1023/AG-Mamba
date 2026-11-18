# -*- coding: utf-8 -*-
"""PU / PHM2009："有对齐头 vs 无对齐头" 四臂对照底稿。

问题（用户 2026-09-28）：跳数据集上删掉对齐头是否安全？需要峰值与末十两
口径，且必须同批次配对（§5.4：同配置跨批次平均绝对差 1.07、最大 2.89 点，
与待测效应量同量级）。

四臂（前端逐位相同，仅损失与预算不同）：
  xg2_std / xg2_6x : 对齐开（w_mmd 0.5 / w_adv 0.1）
  af_std  / af_6x  : 对齐关（w_mmd = w_adv = 0）
早期跨批次日志 xg_std / xg_6x 也一并列出，用于量化"批次效应"有多大。

输出：docs/manuscript/_s62_af_vs_ref.md
"""
import os
import re

import numpy as np
from scipy import stats

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
OUT = os.path.join(ROOT, 'results')
PAT_STEP = re.compile(r'=+\s*Step\s+(\d+)\s*=+')
PAT_ACC = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
TASKS = ['T0', 'T1', 'T2', 'T3']
SEEDS = list(range(5))
DATA = ['PU', 'PHM2009']
PAIRS = [('xg2_std', 'af_std', 'std budget (60 epochs)'),
         ('xg2_6x', 'af_6x', '2,880 steps + cosine')]
CROSSB = [('xg_std', 'af_std'), ('xg_6x', 'af_6x')]
L = []
P = L.append


def traj(data, task, arm, seed):
    p = _findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(
        data, task, arm, seed))
    if not os.path.exists(p) or os.path.getsize(p) < 500:
        return None
    cur, out = 0, []
    with open(p, errors='ignore') as f:
        for line in f:
            m = PAT_STEP.search(line)
            if m:
                cur = int(m.group(1))
                continue
            m = PAT_ACC.search(line)
            if m:
                out.append((cur, float(m.group(1)) * 100.0))
    return out or None


def best(tr):
    return None if tr is None else max(a for _, a in tr)


def lastk(tr, k=10):
    if tr is None or len(tr) < k:
        return None
    return float(np.mean([a for _, a in tr][-k:]))


def series(data, arm, key, tasks=TASKS, seeds=SEEDS):
    vals = []
    for t in tasks:
        for s in seeds:
            tr = traj(data, t, arm, s)
            v = best(tr) if key == 'best' else lastk(tr)
            vals.append(np.nan if v is None else v)
    return np.array(vals, dtype=float)


def ms(x):
    return np.nanmean(x), np.nanstd(x)


def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan, np.nan, 0, int(m.sum())
    _, p = stats.ttest_rel(a[m], b[m])
    d = a[m] - b[m]
    return float(d.mean()), float(p), int((d > 0).sum()), int(m.sum())


P('# PU / PHM2009：有对齐头 vs 无对齐头（四臂同批次对照）')
P('')
P('口径：`peak` = 逐评估最大值；`last-ten` = 末十个评估均值。单位 %。')
P('Δ = 无对齐 − 有对齐（正 = 删掉更好），p 为双侧配对 t 检验。')
P('')

for data in DATA:
    P('## {}'.format(data))
    P('')
    have = any(traj(data, t, a, s) is not None
               for t in TASKS for a in ('xg2_std', 'af_std') for s in SEEDS)
    P('本批次日志是否存在：{}'.format('是' if have else '**否（尚未跑）**'))
    P('')
    arms = ['xg_std', 'xg2_std', 'af_std', 'xg_6x', 'xg2_6x', 'af_6x']
    P('### A. 各臂绝对值（逐任务 peak / last-ten，池化 n = 20）')
    P('')
    hdr = '| 臂 | 批次 | 对齐 | 预算 | ' + ' | '.join(
        '{} peak / l10'.format(t) for t in TASKS) + ' | pooled peak | pooled l10 |'
    P(hdr)
    P('|---' * (4 + len(TASKS) + 2) + '|')
    for a in arms:
        row = []
        for t in TASKS:
            b = series(data, a, 'best', [t])
            l = series(data, a, 'last10', [t])
            if np.all(np.isnan(b)):
                row.append('– / –')
            else:
                row.append('{:.2f} / {:.2f}'.format(np.nanmean(b),
                                                   np.nanmean(l)))
        bp, lp = series(data, a, 'best'), series(data, a, 'last10')
        if np.all(np.isnan(bp)):
            P('| {} | {} | {} | {} | {} | – | – |'.format(
                a, '早前' if a in ('xg_std', 'xg_6x') else '本批',
                '开' if a.startswith('xg') else '关',
                '60 ep' if a.endswith('std') else '2,880',
                ' | '.join(row)))
            continue
        P('| {} | {} | {} | {} | {} | {:.2f} ± {:.2f} | {:.2f} |'.format(
            a, '早前' if a in ('xg_std', 'xg_6x') else '本批',
            '开' if a.startswith('xg') else '关',
            '60 ep' if a.endswith('std') else '2,880',
            ' | '.join(row), *ms(bp), np.nanmean(lp)))
    P('')
    P('### B. 同批次配对：无对齐 − 有对齐')
    P('')
    P('| 预算 | 统计量 | Δ | p | wins | n | 逐任务 Δ |')
    P('|---|---|---|---|---|---|---|')
    for ref, af, lab in PAIRS:
        for key in ('best', 'last10'):
            a, b = series(data, af, key), series(data, ref, key)
            d, p, w, n = paired(a, b)
            per = []
            for t in TASKS:
                d2, _, _, _ = paired(series(data, af, key, [t]),
                                     series(data, ref, key, [t]))
                per.append('{}: {:+.2f}'.format(t, d2))
            P('| {} | {} | {:+.2f} | {:.3g} | {}/{} | {} | {} |'.format(
                lab, key, d, p, w, n, n, ', '.join(per)))
    P('')
    P('### C. 批次效应量（早期跨批次日志 vs 本次，同配置同种子）')
    P('')
    P('| 对比 | 统计量 | Δ | 平均绝对 Δ | 最大绝对 Δ | n |')
    P('|---|---|---|---|---|---|')
    for old, new in CROSSB:
        for key in ('best', 'last10'):
            a, b = series(data, new, key), series(data, old, key)
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 3:
                continue
            d = a[m] - b[m]
            P('| {} − {} | {} | {:+.2f} | {:.2f} | {:.2f} | {} |'.format(
                new, old, key, d.mean(), np.abs(d).mean(), np.abs(d).max(),
                int(m.sum())))
    P('')

out = os.path.join(ROOT, 'docs/manuscript', '_s62_af_vs_ref.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
print('written -> {}'.format(out))
