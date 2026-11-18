#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对齐头消融的数值底稿（补 §6.4 / 局限 L5 的空档）。

问题：交付配置用 w_mmd = 0.5、w_adv = 0.1，但这两个权重此前从未在**交付
前端**上被消融过（表 14 现有的对齐头证据来自早期窄带前端）。

数据来源（stage `align_ablate_hust`，480 步探针协议，HUST T0/T4/T9 × 5 种子）：
  · aln_full  : w_mmd 0.5, w_adv 0.1  （同批次对照）
  · aln_nommd : w_mmd 0.0, w_adv 0.1
  · aln_noadv : w_mmd 0.5, w_adv 0.0
  · aln_noall : w_mmd 0.0, w_adv 0.0  （纯分类）
四臂除这两个权重外逐位相同（前端 = 交付前端：gaborwide + chan512 +
gabor-init + f_lo 0.0025 + ksize128 + dc0.3 + alias2.0 + use_ssm + τ₀≡0）。

口径与表 14 一致：`best` = 逐评估最大值（探针协议 480 步无调度器）；
同时给出 `last10`（末十个评估均值）以便与 §5.2 的单一口径互校。
输出：docs/manuscript/_s64_align_data.md
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
PAT_ACC = re.compile(r'tgt_acc:?\s*([0-9.]+)')
ARMS = ['aln_full', 'aln_nommd', 'aln_noadv', 'aln_noall']
ARM_LABEL = {
    'aln_full': 'full (MMD + adv)',
    'aln_nommd': 'no MMD (adv only)',
    'aln_noadv': 'no adv (MMD only)',
    'aln_noall': 'no both (plain CE)',
}
TASKS = ['T0', 'T4', 'T9']
SEEDS = list(range(5))
L = []
P = L.append


def traj(task, name, seed, prefix='HUST'):
    p = _findlog('{}_{}_{}_seed{}.log'.format(prefix, task, name, seed))
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


def series(name, key, tasks=TASKS, seeds=SEEDS):
    vals = []
    for t in tasks:
        for s in seeds:
            tr = traj(t, 'AGMamba_fc_' + name, s)
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


P('# 对齐头消融数值底稿（由 `summary_align.py` 从 results_dgfdb 重算）')
P('')
P('口径：`best` = 逐评估最大值（与表 14 现有行一致）；`last10` = 末十个评估均值。'
  '单位 %。四臂 = 交付前端固定，仅 w_mmd / w_adv 不同；480 步探针协议。')
P('')
P('## A. 四臂逐任务（best / last10）')
P('')
hdr = '| 臂 | ' + ' | '.join('{} best'.format(t) for t in TASKS) + ' | pooled best |'
hdr += ' | ' + ' | '.join('{} last10'.format(t) for t in TASKS) + ' | pooled last10 |'
P(hdr)
P('|---' * (2 + 2 * len(TASKS)) + '|')
vals = {}
for a in ARMS:
    vals[a] = {k: series(a, k) for k in ('best', 'last10')}
    row = ['**{}**'.format(ARM_LABEL[a])]
    for t in TASKS:
        x = series(a, 'best', [t])
        row.append('{:.2f} ± {:.2f}'.format(*ms(x)))
    row.append('{:.2f} ± {:.2f}'.format(*ms(vals[a]['best'])))
    for t in TASKS:
        x = series(a, 'last10', [t])
        row.append('{:.2f}'.format(np.nanmean(x)))
    row.append('{:.2f}'.format(np.nanmean(vals[a]['last10'])))
    P('| ' + ' | '.join(row) + ' |')
P('')

P('## B. 配对差（变体 − 对照，同种子严格配对；与表 14 的 Δ 约定一致）')
P('')
P('| 对比 | 统计量 | Δ | p | wins | n | 逐任务 Δ |')
P('|---|---|---|---|---|---|---|')
for var, ref, lab in [('aln_nommd', 'aln_full', 'no MMD − full'),
                      ('aln_noadv', 'aln_full', 'no adv − full'),
                      ('aln_noall', 'aln_full', 'no both − full'),
                      ('aln_noall', 'aln_nommd', 'no both − no MMD'),
                      ('aln_nommd', 'aln_noadv', 'no MMD − no adv')]:
    for key in ('best', 'last10'):
        d, p, w, n = paired(vals[var][key], vals[ref][key])
        per = []
        for t in TASKS:
            d2, p2, w2, n2 = paired(series(var, key, [t]), series(ref, key, [t]))
            per.append('{}: {:+.2f}'.format(t, d2))
        P('| {} | {} | {:+.2f} | {:.3g} | {}/{} | {} | {} |'.format(
            lab, key, d, p, w, n, n, ', '.join(per)))
P('')

P('## C. 关闭对齐项后的代价（对照 = full，正 = 关掉更好）')
P('')
d_all, p_all, w_all, n_all = paired(vals['aln_noall']['best'],
                                    vals['aln_full']['best'])
d_mmd, p_mmd, _, _ = paired(vals['aln_nommd']['best'], vals['aln_full']['best'])
d_adv, p_adv, _, _ = paired(vals['aln_noadv']['best'], vals['aln_full']['best'])
P('  · 两臂都关（纯分类）: {:+.2f} (p = {:.3g}, {}/{})'.format(
    d_all, p_all, w_all, n_all))
P('  · 只关 MMD          : {:+.2f} (p = {:.3g})'.format(d_mmd, p_mmd))
P('  · 只关对抗          : {:+.2f} (p = {:.3g})'.format(d_adv, p_adv))
P('')

P('## D. 跨批次 sanity：与批次 A 的同一交付前端臂对照')
P('')
P('sa_lowf512（批次 A，配置与 aln_full 相同，不同批次）与 aln_full 的逐种子差')
P('给出跨批次漂移量级，用于判断同批次配对的必要性（§5.4 底噪 2.55 点）。')
P('')
d_cb = []
for t in TASKS:
    for s in SEEDS:
        a = traj(t, 'AGMamba_fc_sa_lowf512', s)
        b = traj(t, 'AGMamba_fc_aln_full', s)
        if a is not None and b is not None:
            d_cb.append(best(a) - best(b))
if d_cb:
    d_cb = np.array(d_cb)
    P('| 跨批次 | n | 均值 | 平均绝对 | 最大绝对 |')
    P('|---|---|---|---|---|')
    P('| sa_lowf512 − aln_full (best) | {} | {:+.2f} | {:.2f} | {:.2f} |'.format(
        len(d_cb), d_cb.mean(), np.abs(d_cb).mean(), np.abs(d_cb).max()))
else:
    P('（sa_lowf512 日志缺失，跳过）')
P('')

out = os.path.join(ROOT, 'docs/manuscript', '_s64_align_data.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
print('written -> {}'.format(out))
