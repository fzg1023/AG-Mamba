#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因实验汇总（§6.2b）：

  (a) PU 2x2 —— 弥散 α ∈ {0,2} × 前端宽度 ∈ {128,512}，判定 T2/T3 的
      +8.32/+4.14 有多少来自弥散、多少来自容量/带宽（含交互项）；
  (b) PHM2009 转速x负载协议 —— 协议 B（低负载阶梯 L0–L3）与协议 C
      （双因素外推 M0–M3），本方法 xg_std / 窄带对照 base / 三个基线；
  (c) 三协议对比 —— High 阶梯（T0–T3，已有）、Low 阶梯、混合轴三种偏移下
      "ours − base" 与 "ours − WIDAN" 的差距。

输出：docs/manuscript/_s62b_attribution.md
用法：python -u summary_attribution.py
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
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEEDS = list(range(5))

PU_TASKS = ['T0', 'T1', 'T2', 'T3']
PU_AC = [('puac_a0_c128', 0.0, 128), ('puac_a2_c128', 2.0, 128),
         ('puac_a0_c512', 0.0, 512), ('puac_a2_c512', 2.0, 512)]
PHM_EXT = ['L0', 'L1', 'L2', 'L3', 'M0', 'M1', 'M2', 'M3']
PHM_BASE_METHODS = ['WIDAN', 'UDTL', 'DANN']


def series(path):
    if not os.path.exists(path) or os.path.getsize(path) < 500:
        return None
    with open(path, errors='ignore') as f:
        vals = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return vals or None


def best(path):
    s = series(path)
    return max(s) if s else None


def lastten(path):
    s = series(path)
    return float(np.mean(s[-10:])) if s else None


def lm_path(data, task, name, seed):
    return _findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(
        data, task, name, seed))


def bl_path(data, task, method, seed):
    """基线日志：新协议（L*/M*）写在 results_dgfdb；原八基线在
    results_phm2009 / results_pu 下，命名不带数据集前缀；
    WIDAN / UDTL 在 results_widan / results_udtl 下，命名带数据集前缀。"""
    cands = []
    if task[:1] in ('L', 'M'):
        cands.append(_findlog('{}_{}_{}_seed{}.log'.format(
            data, task, method, seed)))
    cands.append(_findlog('{}_{}_{}_seed{}.log'.format(data, task, method,
                                                       seed)))
    cands.append(_findlog('{}_{}_{}_seed{}.log'.format(
        data, task, method, seed)))
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[-1]


def cell(fn, task, seed):
    p = fn(task, seed)
    return best(p), lastten(p)


def paired(a_vals, b_vals):
    """配对双侧 t 检验（逐对剔除缺失值）；返回 (mean, p, wins, n)。"""
    d = np.array([x - y for x, y in zip(a_vals, b_vals)
                  if x is not None and y is not None], dtype=float)
    if len(d) < 2:
        return float('nan'), float('nan'), 0, len(d)
    t, p = stats.ttest_rel(d, np.zeros_like(d))
    return float(d.mean()), float(p), int((d > 0).sum()), len(d)


def safe_mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float('nan')


L = []
P = L.append
P('# §6.2b 归因实验汇总（PU 弥散×容量 2x2；PHM2009 转速x负载协议）')
P('')

# --------------------------------------------------------------------- (a) PU
P('## (a) PU：弥散 α x 前端宽度 2x2（4 任务 x 5 种子；预算 = 60 epoch / 2,640 步）')
P('')
P('| 臂 | T0 | T1 | T2 | T3 | 均值 |')
P('|---|---|---|---|---|---|')
arm_series = {}
for nm, alias, ch in PU_AC:
    fn = lambda t, s, nm=nm: lm_path('PU', t, nm, s)
    per_task, allv = [], []
    for t in PU_TASKS:
        v = [best(fn(t, s)) for s in SEEDS]
        per_task.append(safe_mean(v))
        allv += [x for x in v if x is not None]
    for t in PU_TASKS:
        for s in SEEDS:
            arm_series[(nm, t, s)] = best(fn(t, s))
    P('| {} (α={}, {}ch) | {} | **{:.2f}** |'.format(
        nm, alias, ch, ' | '.join('{:.2f}'.format(x) for x in per_task),
        safe_mean(allv)))
# 参照行：base / xg_std
for label, nm in [('窄带对照 base', 'base'), ('交付配置 xg_std', 'xg_std')]:
    fn = lambda t, s, nm=nm: lm_path('PU', t, nm, s)
    per_task, allv = [], []
    for t in PU_TASKS:
        v = [best(fn(t, s)) for s in SEEDS]
        per_task.append(safe_mean(v))
        allv += [x for x in v if x is not None]
    P('| {}（参照） | {} | **{:.2f}** |'.format(
        label, ' | '.join('{:.2f}'.format(x) for x in per_task),
        safe_mean(allv)))
P('')

P('### 主效应与交互（配对 (任务,种子) 对，n = 20）')
P('')
P('| 对比 | Δ (best) | p | wins | Δ (last-ten) | p | wins |')
P('|---|---|---|---|---|---|---|')


def _pairs(nm_a, nm_b, rule='best'):
    f = best if rule == 'best' else lastten
    a, b = [], []
    for t in PU_TASKS:
        for s in SEEDS:
            a.append(f(lm_path('PU', t, nm_a, s)))
            b.append(f(lm_path('PU', t, nm_b, s)))
    return a, b


CONTRASTS = [
    ('弥散效应 @512ch：a2_c512 − a0_c512', 'puac_a2_c512', 'puac_a0_c512'),
    ('弥散效应 @128ch：a2_c128 − a0_c128', 'puac_a2_c128', 'puac_a0_c128'),
    ('容量效应 @α=2：a2_c512 − a2_c128', 'puac_a2_c512', 'puac_a2_c128'),
    ('容量效应 @α=0：a0_c512 − a0_c128', 'puac_a0_c512', 'puac_a0_c128'),
    ('交付配置 vs 窄带对照：xg_std − base', 'xg_std', 'base'),
]
for lab, a_nm, b_nm in CONTRASTS:
    row = [lab]
    for rule in ('best', 'lastten'):
        a, b = _pairs(a_nm, b_nm, rule)
        m, p, w, n = paired(a, b)
        row.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
    P('| {} | {} |'.format(row[0], ' | '.join(row[1:])))
# 交互项：Δ(α@512) − Δ(α@128)，逐对计算并剔除缺失
for rule in ('best', 'lastten'):
    f = best if rule == 'best' else lastten
    d1, d2 = [], []
    for t in PU_TASKS:
        for s in SEEDS:
            a2, a0 = f(lm_path('PU', t, 'puac_a2_c512', s)), f(lm_path('PU', t, 'puac_a0_c512', s))
            b2, b0 = f(lm_path('PU', t, 'puac_a2_c128', s)), f(lm_path('PU', t, 'puac_a0_c128', s))
            if None in (a2, a0, b2, b0):
                continue
            d1.append(a2 - a0)
            d2.append(b2 - b0)
    d1 = np.array(d1, dtype=float)
    d2 = np.array(d2, dtype=float)
    if len(d1) > 1:
        t_, p_ = stats.ttest_rel(d1, d2)
        P('| 交互项 ({})：Δα@512 − Δα@128 | {:+.2f} | {:.4f} | | | |'.format(
            rule, float(d1.mean() - d2.mean()), float(p_)))
    else:
        P('| 交互项 ({})：Δα@512 − Δα@128 | – | – | | | |'.format(rule))
P('')
P('  读法：若"弥散效应"在两档宽度下都显著且量级相近、而"容量效应"不显著，')
P('  则 §6.2 中 T2/T3 的收益来自弥散；若相反，则应归因于容量/带宽。')
P('')

# -------------------------------------------------------------- (b) PHM2009
P('## (b) PHM2009 转速x负载协议（4 任务 x 5 种子/协议）')
P('')
P('协议 B（L*）：30/35/40/45 Hz 全部 **Low 负载** —— 纯转速阶梯、不同负载水平；')
P('协议 C（M*）：目标为 Low 负载且其转速在源域（全 High 负载）中不出现 —— 两轴同时外推。')
P('')
for proto, tasks, desc in [('B 低负载阶梯', PHM_EXT[:4], 'Low ladder'),
                           ('C 双因素外推', PHM_EXT[4:], 'mixed axes')]:
    P('### 协议 {}（{}）'.format(proto, desc))
    P('')
    P('| 方法 | {} | 均值 |'.format(' | '.join(tasks)))
    P('|---|---{}|---|'.format('|---' * len(tasks)))
    rows = [('AG-Mamba (xg_std)', lambda t, s: lm_path('PHM2009', t, 'xg_std', s)),
            ('窄带对照 base', lambda t, s: lm_path('PHM2009', t, 'base', s))]
    rows += [(m, lambda t, s, m=m: bl_path('PHM2009', t, m, s))
             for m in PHM_BASE_METHODS]
    for label, fn in rows:
        per_task, allv = [], []
        for t in tasks:
            v = [best(fn(t, s)) for s in SEEDS]
            per_task.append(safe_mean(v))
            allv += [x for x in v if x is not None]
        P('| {} | {} | **{:.2f}** |'.format(
            label, ' | '.join('{:.2f}'.format(x) for x in per_task),
            safe_mean(allv)))
    P('')
    P('| 配对对比（n = {}） | Δ best | p | wins | Δ last-ten | p |'.format(
        len(tasks) * len(SEEDS)))
    P('|---|---|---|---|---|---|')
    for other_lab, other_fn in [('base（窄带对照）',
                                 lambda t, s: lm_path('PHM2009', t, 'base', s)),
                                ('WIDAN', lambda t, s: bl_path('PHM2009', t, 'WIDAN', s)),
                                ('UDTL', lambda t, s: bl_path('PHM2009', t, 'UDTL', s))]:
        cells = []
        for rule in ('best', 'lastten'):
            f = best if rule == 'best' else lastten
            a = [f(lm_path('PHM2009', t, 'xg_std', s)) for t in tasks for s in SEEDS]
            b = [f(other_fn(t, s)) for t in tasks for s in SEEDS]
            m, p, w, n = paired(a, b)
            cells.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
        P('| xg_std − {} | {} |'.format(other_lab, ' | '.join(cells)))
    P('')

# ---------------------------------------------------- (c) 三协议对比（ours−base）
P('## (c) 三协议对比：偏移越"混合"，弥散前端越值钱？')
P('')
P('| 协议（偏移类型） | ours − base (best) | p | ours − base (last-ten) | ours − WIDAN (best) |')
P('|---|---|---|---|---|')
proto_specs = [('High 阶梯 T0–T3（纯转速 @High，原协议）', ['T0', 'T1', 'T2', 'T3']),
               ('Low 阶梯 L0–L3（纯转速 @Low，协议 B）', PHM_EXT[:4]),
               ('双因素外推 M0–M3（转速+负载，协议 C）', PHM_EXT[4:])]
for lab, tasks in proto_specs:
    row = [lab]
    for rule in ('best', 'lastten'):
        f = best if rule == 'best' else lastten
        a = [f(lm_path('PHM2009', t, 'xg_std', s)) for t in tasks for s in SEEDS]
        b = [f(lm_path('PHM2009', t, 'base', s)) for t in tasks for s in SEEDS]
        m, p, w, n = paired(a, b)
        row += ['{:+.2f}'.format(m), '{:.4f}'.format(p)]
    f = best
    a = [f(lm_path('PHM2009', t, 'xg_std', s)) for t in tasks for s in SEEDS]
    b = [f(bl_path('PHM2009', t, 'WIDAN', s)) for t in tasks for s in SEEDS]
    m, p, w, n = paired(a, b)
    row.append('{:+.2f}'.format(m))
    P('| {} | {} |'.format(row[0], ' | '.join(row[1:])))
P('')

# ------------------------------------------------- (d) 容量对齐（cap960 系列）
P('## (d) 容量对齐实验：cap960（≈3.85M，对 WIDAN 3.98M 为 0.97×）')
P('')
P('① PHM2009 三协议（默认预算）—— 与 xg_std(512ch) / base / WIDAN 对比：')
P('')
P('| 协议 | 任务 | xg_std(512) | **cap960** | base | WIDAN | cap960−xg_std | cap960−WIDAN |')
P('|---|---|---|---|---|---|---|---|')
for lab, tasks in [('High 阶梯', ['T0', 'T1', 'T2', 'T3']),
                   ('Low 阶梯', PHM_EXT[:4]),
                   ('双因素', PHM_EXT[4:])]:
    cells = []
    for nm, fn in [('xg_std', lambda t, s: lm_path('PHM2009', t, 'xg_std', s)),
                   ('cap960', lambda t, s: lm_path('PHM2009', t, 'cap960', s)),
                   ('base', lambda t, s: lm_path('PHM2009', t, 'base', s)),
                   ('WIDAN', lambda t, s: bl_path('PHM2009', t, 'WIDAN', s))]:
        v = [best(fn(t, s)) for t in tasks for s in SEEDS]
        cells.append(safe_mean(v))
    _d1 = [(best(lm_path('PHM2009', t, 'cap960', s)),
            best(lm_path('PHM2009', t, 'xg_std', s)))
           for t in tasks for s in SEEDS]
    _d2 = [(best(lm_path('PHM2009', t, 'cap960', s)),
            best(bl_path('PHM2009', t, 'WIDAN', s)))
           for t in tasks for s in SEEDS]
    d1 = [a - b for a, b in _d1 if a is not None and b is not None]
    d2 = [a - b for a, b in _d2 if a is not None and b is not None]
    d1 = [x for x in d1 if x is not None]
    d2 = [x for x in d2 if x is not None]
    P('| {} | {} | {:.2f} | **{:.2f}** | {:.2f} | {:.2f} | {} | {} |'.format(
        lab, '{}'.format(len(tasks)),
        cells[0], cells[1], cells[2], cells[3],
        ('{:+.2f}'.format(np.mean(d1)) if d1 else '–'),
        ('{:+.2f}'.format(np.mean(d2)) if d2 else '–')))
P('')
P('| 配对对比（n = 20） | Δ best | p | wins | Δ last-ten | p |')
P('|---|---|---|---|---|---|')
for lab, tasks in [('High 阶梯', ['T0', 'T1', 'T2', 'T3']),
                   ('Low 阶梯', PHM_EXT[:4]),
                   ('双因素', PHM_EXT[4:])]:
    for other, fn in [('xg_std(512ch)', lambda t, s: lm_path('PHM2009', t, 'xg_std', s)),
                      ('WIDAN', lambda t, s: bl_path('PHM2009', t, 'WIDAN', s)),
                      ('base', lambda t, s: lm_path('PHM2009', t, 'base', s))]:
        cs = []
        for rule in ('best', 'lastten'):
            f = best if rule == 'best' else lastten
            a = [f(lm_path('PHM2009', t, 'cap960', s)) for t in tasks for s in SEEDS]
            b = [f(fn(t, s)) for t in tasks for s in SEEDS]
            m, p, w, n = paired(a, b)
            cs.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
        P('| {}：cap960 − {} | {} |'.format(lab, other, ' | '.join(cs)))
P('')
P('② PU（默认预算）：cap960 − puac_a2_c512（同配置、仅 512→960 通道）与 − xg_std')
P('')
P('| 对比 | Δ best | p | wins | Δ last-ten | p |')
P('|---|---|---|---|---|---|')
for other, fn in [('puac_a2_c512 (512ch, 同配置)',
                   lambda t, s: lm_path('PU', t, 'puac_a2_c512', s)),
                  ('xg_std (512ch, 交付配置)',
                   lambda t, s: lm_path('PU', t, 'xg_std', s)),
                  ('WIDAN', lambda t, s: bl_path('PU', t, 'WIDAN', s)),
                  ('base (窄带)', lambda t, s: lm_path('PU', t, 'base', s))]:
    cs = []
    for rule in ('best', 'lastten'):
        f = best if rule == 'best' else lastten
        a = [f(lm_path('PU', t, 'cap960', s)) for t in PU_TASKS for s in SEEDS]
        b = [f(fn(t, s)) for t in PU_TASKS for s in SEEDS]
        m, p, w, n = paired(a, b)
        cs.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
    P('| cap960 − {} | {} |'.format(other, ' | '.join(cs)))
P('')
P('③ HUST：容量曲线（探针预算 480 步，T0/T4/T9；表 8 口径）')
P('')
P('| 臂 | 通道 | T0 | T4 | T9 | 池化(15) |')
P('|---|---|---|---|---|---|')
for nm, ch in [('sa_lowf512', 512), ('sa_cap768_lowf', 768),
               ('cap960p', 960), ('sa_cap1024', 1024)]:
    pt, allv = [], []
    for t in ['T0', 'T4', 'T9']:
        v = [best(lm_path('HUST', t, nm, s)) for s in SEEDS]
        pt.append(safe_mean(v))
        allv += [x for x in v if x is not None]
    P('| {} | {} | {} | **{:.2f}** |'.format(
        nm, ch, ' | '.join('{:.2f}'.format(x) for x in pt), safe_mean(allv)))
P('')
P('| 对比（n = 15） | Δ best | p | wins | Δ last-ten | p |')
P('|---|---|---|---|---|---|')
for other in ['sa_lowf512', 'sa_cap768_lowf', 'sa_cap1024']:
    cs = []
    for rule in ('best', 'lastten'):
        f = best if rule == 'best' else lastten
        a = [f(lm_path('HUST', t, 'cap960p', s)) for t in ['T0', 'T4', 'T9'] for s in SEEDS]
        b = [f(lm_path('HUST', t, other, s)) for t in ['T0', 'T4', 'T9'] for s in SEEDS]
        m, p, w, n = paired(a, b)
        cs.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
    P('| cap960p − {} | {} |'.format(other, ' | '.join(cs)))
P('')
P('④ HUST 交付预算（2,880 步 + 余弦）：cap960x − 交付配置 sc_steps6x（T0/T4/T9）')
P('')
P('| 对比（n = 15） | Δ best | p | wins | Δ last-ten | p |')
P('|---|---|---|---|---|---|')
for other in ['sc_steps6x', 'sa_lowf512']:
    cs = []
    for rule in ('best', 'lastten'):
        f = best if rule == 'best' else lastten
        a = [f(lm_path('HUST', t, 'cap960x', s)) for t in ['T0', 'T4', 'T9'] for s in SEEDS]
        b = [f(lm_path('HUST', t, other, s)) for t in ['T0', 'T4', 'T9'] for s in SEEDS]
        m, p, w, n = paired(a, b)
        cs.append('{:+.2f} | {:.4f} | {}/{}'.format(m, p, w, n))
    P('| cap960x − {} | {} |'.format(other, ' | '.join(cs)))
P('')

txt = '\n'.join(L) + '\n'
with open(os.path.join(ROOT, 'docs/manuscript', '_s62b_attribution.md'), 'w',
          encoding='utf-8') as fh:
    fh.write(txt)
print(txt)