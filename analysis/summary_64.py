#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.4 机制审计数值底稿。

A. 机制消融（前端固定为 α=2 核组）
   · SSM 有/无（gw_a2_ssm / gw_a2_nossm，T0/T9 × 5）
   · v3 三联（v3_full / v3_noliquid / v3_nossm，T0/T9 × 10）
   · τ₀ 处理（abl_base / abl_zero / abl_shuffle / abl_const，T0 × 5）
   · τ 估计器（tau_auto / tau_spec，T0 × 5）
   · PHM 上的 τ₀ 复核（abl10_base / abl10_const，PHM T0 × 10）
B. 预算–精度曲线（匹配步数读取，无检查点选择）
   · v4_noL(480) / sb_long3x(1440) / sc_steps6x(2880) / sd_12x(5760) / WIDAN(480) / WIDAN_long(2880)
   · 在 T0/T4/T9 × 5 上，读取“步数 ≤ B 的最后一个评估点”
C. 被拒的改动（负结果）：EMA、SSM 扩容、前端 256/1024 通道、α=2.5/3、12× 预算
输出：docs/manuscript/_s64_data.md
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
L = []


def traj(task, name, seed, prefix='HUST'):
    """返回 [(step, acc%)...]（含最后一步）。"""
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


def at_step(tr, budget):
    """步数 ≤ budget 的最后一个评估值；若日志总步数 < budget 返回 None。"""
    if tr is None:
        return None
    cap = max(s for s, _ in tr)
    if cap < budget:
        return None
    cand = [a for s, a in tr if s <= budget]
    return cand[-1] if cand else None


def tail_up_to(tr, budget, k=10):
    if tr is None:
        return None
    cand = [a for s, a in tr if s <= budget]
    if len(cand) < k:
        return None
    return float(np.mean(cand[-k:]))


def best(tr):
    return None if tr is None else max(a for _, a in tr)


def lastk(tr, k=10):
    return None if tr is None else float(np.mean([a for _, a in tr][-k:]))


def cap(tr):
    return None if tr is None else max(s for s, _ in tr)


def gather(tasks, name, seeds=range(5), prefix='HUST'):
    return {(t, s): traj(t, name, s, prefix) for t in tasks for s in seeds}


def series(d, key, tasks, seeds=range(5), **kw):
    vals = []
    for t in tasks:
        for s in seeds:
            tr = d[(t, s)]
            if key == 'step':
                v = at_step(tr, kw['budget'])
            elif key == 'best':
                v = best(tr)
            elif key == 'last10':
                v = lastk(tr)
            elif key == 'tail':
                v = tail_up_to(tr, kw['budget'])
            elif key == 'cap':
                v = cap(tr)
            else:
                raise KeyError(key)
            vals.append(np.nan if v is None else v)
    return np.array(vals, dtype=float)


def ms(x):
    return np.nanmean(x), np.nanstd(x)


def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan, np.nan, 0, int(m.sum())
    t, p = stats.ttest_rel(a[m], b[m])
    d = a[m] - b[m]
    return float(d.mean()), float(p), int((d > 0).sum()), int(m.sum())


P = L.append
P('# §6.4 机制审计数值底稿（由 `summary_64.py` 从 results_dgfdb 重算）')
P('')
P('口径：`best` = 逐评估最大值；`last10` = 末十个评估均值；`@B` = 步数 ≤ B 的最后一个评估值'
  '（匹配步数、无检查点选择）。单位 %。')
P('')

# ------------------------------------------------------------ A. 机制消融
P('## A1. SSM 分支（前端 = α=2 核组，T0/T9 × 5 种子，480 步）')
P('')
P('| 变体 | T0 best | T9 best | 池化(10) best | T0 last10 | T9 last10 |')
P('|---|---|---|---|---|---|')
d_ssm, d_nossm = gather(['T0', 'T9'], 'AGMamba_fc_gw_a2_ssm'), \
    gather(['T0', 'T9'], 'AGMamba_fc_gw_a2_nossm')
for nm, d in [('SSM (gw_a2_ssm)', d_ssm), ('no SSM (gw_a2_nossm)', d_nossm)]:
    P('| {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} |'
      .format(nm, *ms(series(d, 'best', ['T0'])), *ms(series(d, 'best', ['T9'])),
              *ms(series(d, 'best', ['T0', 'T9'])),
              np.nanmean(series(d, 'last10', ['T0'])),
              np.nanmean(series(d, 'last10', ['T9']))))
for key in ('best', 'last10'):
    dd, p, w, n = paired(series(d_ssm, key, ['T0', 'T9']),
                         series(d_nossm, key, ['T0', 'T9']))
    P('  {}: SSM − noSSM = {:+.2f} (p = {:.3g}, {}/{})'.format(key, dd, p, w, n))
P('')

P('## A2. v3 三联（T0/T9 × 10 种子；同一前端）')
P('')
P('| 变体 | T0 best | T9 best | 池化(20) |')
P('|---|---|---|---|')
d3 = {k: gather(['T0', 'T9'], 'AGMamba_fc_v3_' + k, seeds=range(10))
      for k in ('full', 'noliquid', 'nossm')}
for k, nm in [('full', 'v3_full (SSM + 液体)'), ('noliquid', 'v3_noliquid (SSM)'),
              ('nossm', 'v3_nossm (无 SSM)')]:
    P('| {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(
        nm, *ms(series(d3[k], 'best', ['T0'], seeds=range(10))),
        *ms(series(d3[k], 'best', ['T9'], seeds=range(10))),
        *ms(series(d3[k], 'best', ['T0', 'T9'], seeds=range(10)))))
P('')
for a, b, lab in [('full', 'noliquid', '液体 ON − OFF'),
                  ('full', 'nossm', 'SSM ON − 无 SSM'),
                  ('noliquid', 'nossm', 'SSM(无液体) − 无 SSM')]:
    dd, p, w, n = paired(series(d3[a], 'best', ['T0', 'T9'], seeds=range(10)),
                         series(d3[b], 'best', ['T0', 'T9'], seeds=range(10)))
    P('  {}: {:+.2f} (p = {:.3g}, {}/{})'.format(lab, dd, p, w, n))
P('')

P('## A3. τ₀ 处理与估计器（T0 × 5 种子，交付前端）')
P('')
P('| 变体 | best | last10 |')
P('|---|---|---|')
for nm, tag in [('abl_base (τ₀ 正常)', 'AGMamba_fc_abl_base'),
                ('abl_zero (τ₀ ≡ 0)', 'AGMamba_fc_abl_zero'),
                ('abl_shuffle (τ₀ 打乱)', 'AGMamba_fc_abl_shuffle'),
                ('abl_const (τ₀ 常数)', 'AGMamba_fc_abl_const'),
                ('τ 估计器 autocorr', 'AGMamba_fc_tau_auto'),
                ('τ 估计器 envspec', 'AGMamba_fc_tau_spec')]:
    d = gather(['T0'], tag)
    P('| {} | {:.2f} ± {:.2f} | {:.2f} |'.format(
        nm, *ms(series(d, 'best', ['T0'])),
        np.nanmean(series(d, 'last10', ['T0']))))
P('')
base_t = gather(['T0'], 'AGMamba_fc_abl_base')
for tag, lab in [('AGMamba_fc_abl_zero', 'τ₀≡0'), ('AGMamba_fc_abl_shuffle', 'τ₀ shuffled'),
                 ('AGMamba_fc_abl_const', 'τ₀ const')]:
    dd, p, w, n = paired(series(base_t, 'best', ['T0']),
                         series(gather(['T0'], tag), 'best', ['T0']))
    P('  abl_base − {}: {:+.2f} (p = {:.3g}, {}/5)'.format(lab, dd, p, w))
P('')
d_pb = gather(['T0'], 'AGMamba_fc_abl10_base', seeds=range(10), prefix='PHM2009')
d_pc = gather(['T0'], 'AGMamba_fc_abl10_const', seeds=range(10), prefix='PHM2009')
dd, p, w, n = paired(series(d_pb, 'best', ['T0'], seeds=range(10)),
                     series(d_pc, 'best', ['T0'], seeds=range(10)))
P('  PHM2009 T0（10 种子）: abl10_base − abl10_const = {:+.2f} (p = {:.3g}, {}/{})；'
  '均值 {:.2f} vs {:.2f}'.format(dd, p, w, n,
                                 np.nanmean(series(d_pb, 'best', ['T0'], seeds=range(10))),
                                 np.nanmean(series(d_pc, 'best', ['T0'], seeds=range(10)))))
P('')

# ------------------------------------------------------------ B. 预算曲线
BUDGETS = [480, 1440, 2880, 5760]
CURVE = [('AGMamba_fc_v4_noL', 'AG-Mamba (480, 交付架构探针)'),
         ('AGMamba_fc_sb_long3x', 'AG-Mamba (1440)'),
         ('AGMamba_fc_sc_steps6x', 'AG-Mamba (2880, 交付)'),
         ('AGMamba_fc_sd_12x', 'AG-Mamba (5760)'),
         ('WIDAN', 'WIDAN (480, 发布)'),
         ('WIDAN_long', 'WIDAN (2880, 匹配)')]
T3 = ['T0', 'T4', 'T9']
P('## B. 预算–精度曲线（T0/T4/T9 × 5 种子；读取步数 ≤ B 的最后一个评估点）')
P('')
P('| 变体 | 总步数 | @480 | @1440 | @2880 | @5760 |')
P('|---|---|---|---|---|---|')
cur = {}
for tag, nm in CURVE:
    d = gather(T3, tag)
    cur[tag] = d
    cells = []
    for b in BUDGETS:
        x = series(d, 'step', T3, budget=b)
        cells.append('–' if np.isnan(x).all() else '{:.2f}'.format(np.nanmean(x)))
    caps = series(d, 'cap', T3)
    P('| {} | {:.0f} | {} |'.format(nm, np.nanmax(caps), ' | '.join(cells)))
P('')
P('  同样用“末十评估均值（截至 B）”复核（非预言，但与 @B 读数高度一致时说明排序稳健）：')
P('')
P('| 变体 | tail10@480 | tail10@1440 | tail10@2880 |')
P('|---|---|---|---|')
for tag, nm in CURVE:
    d = cur[tag]
    cells = []
    for b in [480, 1440, 2880]:
        x = series(d, 'tail', T3, budget=b)
        cells.append('–' if np.isnan(x).all() else '{:.2f}'.format(np.nanmean(x)))
    P('| {} | {} |'.format(nm, ' | '.join(cells)))
P('')

P('## B2. 全 10 任务口径：两条 2880 步轨迹在匹配步数上的读数（无检查点选择）')
P('')
ALL = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
d_ours = gather(ALL, 'AGMamba_fc_sc_steps6x')
d_wl = gather(ALL, 'WIDAN_long')
d_w480 = gather(ALL, 'WIDAN')
d_v4 = gather(ALL, 'AGMamba_fc_v4_noL')
P('| 步数 | 本方法 (2880, 余弦) | WIDAN (2880, 匹配) | Δ |')
P('|---|---|---|---|')
for b in [480, 960, 1440, 2160, 2880]:
    a = series(d_ours, 'step', ALL, budget=b)
    w = series(d_wl, 'step', ALL, budget=b)
    P('| @{} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:+.2f} |'.format(
        b, *ms(a), *ms(w), np.nanmean(a) - np.nanmean(w)))
P('')
P('  配对检验（同任务同种子，n = 50）：')
for b in [480, 1440, 2880]:
    dd, p, w_, n = paired(series(d_ours, 'step', ALL, budget=b),
                          series(d_wl, 'step', ALL, budget=b))
    P('    @{}: Δ = {:+.2f} (p = {:.3g}, {}/{})'.format(b, dd, p, w_, n))
P('')
P('  480 步预算的两个“发布协议”运行（同一协议、同一预算，n = 50）：')
dd, p, w_, n = paired(series(d_v4, 'best', ALL), series(d_w480, 'best', ALL))
P('    best-epoch: 本方法(480, 无调度) − WIDAN(480) = {:+.2f} (p = {:.3g}, {}/{})'.format(
    dd, p, w_, n))
dd, p, w_, n = paired(series(d_v4, 'last10', ALL), series(d_w480, 'last10', ALL))
P('    last-ten:   {:+.2f} (p = {:.3g}, {}/{})'.format(dd, p, w_, n))
dd, p, w_, n = paired(series(d_v4, 'step', ALL, budget=480),
                      series(d_w480, 'step', ALL, budget=480))
P('    @480 末值:  {:+.2f} (p = {:.3g}, {}/{})'.format(dd, p, w_, n))
P('')
P('  峰–末端落差（best − final，全 10 任务池化，单位点）：')
for nm, d in [('本方法 2880', d_ours), ('WIDAN 2880', d_wl),
              ('本方法 480（探针，无调度）', d_v4), ('WIDAN 480', d_w480)]:
    b = np.array([np.nan if d[k] is None else max(a for _, a in d[k])
                  for k in d], dtype=float)
    f = np.array([np.nan if d[k] is None else [a for _, a in d[k]][-1]
                  for k in d], dtype=float)
    P('    {}: best {:.2f} − final {:.2f} = {:+.2f}'.format(
        nm, np.nanmean(b), np.nanmean(f), np.nanmean(b) - np.nanmean(f)))
P('')

# --------------------------------------------------------- C. 被拒的改动
P('## C. 被拒/无效的改动')
P('')
P('| 改动 | 对比对象 | Δ (best, p) | 结论 |')
P('|---|---|---|---|')
def cmp_variants(tag_a, tag_b, tasks, nm, seeds=range(5)):
    da, db = gather(tasks, tag_a, seeds=seeds), gather(tasks, tag_b, seeds=seeds)
    dd, p, w, n = paired(series(da, 'best', tasks, seeds=seeds),
                         series(db, 'best', tasks, seeds=seeds))
    P('| {} | {} | {:+.2f} ({:.3g}, {}/{}) | |'.format(nm, tag_b, dd, p, w, n))

cmp_variants('AGMamba_fc_sd_6x_ssmbig', 'AGMamba_fc_sc_steps6x', T3,
             'SSM 扩容（4 块 / d=192，2880 步）')
cmp_variants('AGMamba_fc_sd_6x_k256', 'AGMamba_fc_sc_steps6x', T3,
             '前端 256 通道（2880 步）')
cmp_variants('AGMamba_fc_sc_alias3', 'AGMamba_fc_sc_steps6x', T3,
             'α = 3（2880 步）')
cmp_variants('AGMamba_fc_sb_a25', 'AGMamba_fc_v4_noL', T3, 'α = 2.5（480 步）')
cmp_variants('AGMamba_fc_sb_ssmbig', 'AGMamba_fc_v4_noL', T3, 'SSM 扩容（480 步）')
cmp_variants('AGMamba_fc_sd_12x', 'AGMamba_fc_sc_steps6x', T3, '预算 12×（5760 步）')
cmp_variants('AGMamba_fc_sb_long3x', 'AGMamba_fc_v4_noL', T3, '预算 3×（1440 步）')
P('')

# EMA
EMA_T = ['T0', 'T5', 'T8', 'T9']
d_em = gather(EMA_T, 'AGMamba_fc_em999')
d_sc = gather(EMA_T, 'AGMamba_fc_sc_steps6x')
P('## C2. EMA 影子权重（负结果，T0/T5/T8/T9 × 5 种子）')
P('')
P('| 任务 | em999 best | sc_steps6x best | em999 last10 | sc_steps6x last10 |')
P('|---|---|---|---|---|')
for t in EMA_T:
    P('| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
        t, np.nanmean(series(d_em, 'best', [t])),
        np.nanmean(series(d_sc, 'best', [t])),
        np.nanmean(series(d_em, 'last10', [t])),
        np.nanmean(series(d_sc, 'last10', [t]))))
for key in ('best', 'last10'):
    dd, p, w, n = paired(series(d_em, key, EMA_T), series(d_sc, key, EMA_T))
    P('  {}: em999 − sc_steps6x = {:+.2f} (p = {:.3g}, {}/{})'.format(
        key, dd, p, w, n))
P('  池化：em999 best {:.2f} / last10 {:.2f}；sc_steps6x best {:.2f} / last10 {:.2f}'
  .format(np.nanmean(series(d_em, 'best', EMA_T)),
          np.nanmean(series(d_em, 'last10', EMA_T)),
          np.nanmean(series(d_sc, 'best', EMA_T)),
          np.nanmean(series(d_sc, 'last10', EMA_T))))
P('')

path = os.path.join(ROOT, 'docs/manuscript', '_s64_data.md')
with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
print('[OK] -> {}'.format(path))
