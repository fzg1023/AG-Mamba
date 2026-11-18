#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.3 Filterbank design validation —— 数值汇总（从逐运行日志重算）。

四组证据：
  A. α 扫描（T0/T9 × 5 种子，480 步探针协议，dc_frac=0.3，512 通道 / ksize64 / f_lo=0.005）
     + dc_frac=0.5 的 α=0 锚点（gw_wide_gabor，Fig. 3 的解析核）
  B. 初始化对照（解析 Gabor / 随机 / 半 Gabor 半随机 / 低 DC）
  C. 低频扩展（f_lo 0.005→0.0025 且首层核 64→128），在 512 与 768 两档上各取一对
  D. 前端容量（512 / 768 / 1024，固定低频扩展前端）
输出：docs/manuscript/_s63_data.md
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

OUT = 'results'
REPORT = os.path.join('docs/manuscript', '_s63_data.md')

TASKS_SWEEP = ['T0', 'T9']
TASKS3 = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2, 3, 4]


def logpath(task, name, seed):
    return _findlog('HUST_{}_AGMamba_fc_{}_seed{}.log'.format(
        task, name, seed))


def series_path(p):
    """逐 epoch 的 tgt_acc（%）。"""
    if not os.path.exists(p):
        return None
    vals = []
    with open(p, 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'tgt_acc:?\s*([0-9.]+)', line)
            if m:
                vals.append(float(m.group(1)) * 100.0)
    return vals or None


def series(task, name, seed):
    return series_path(logpath(task, name, seed))


def wseries(task, tag, seed):
    """参照设计 WIDAN 的日志（命名不同）。"""
    return series_path(_findlog('HUST_{}_WIDAN_{}_seed{}.log'.format(task, tag, seed)))


def collect(tasks, name, seeds=SEEDS):
    """返回 {(task, seed): {'best','last10','final'}}，缺失记 None。"""
    out = {}
    for t in tasks:
        for s in seeds:
            v = series(t, name, s)
            out[(t, s)] = None if v is None else {
                'best': max(v),
                'last10': float(np.mean(v[-10:])),
                'final': v[-1],
            }
    return out


def vec(d, tasks, key, name):
    """按 (task, seed) 取某个统计量的一维数组（缺失记为 nan）。"""
    return np.array([
        np.nan if d[(t, s)] is None else d[(t, s)][key]
        for t in tasks for s in SEEDS], dtype=float)


def ms(x):
    return np.nanmean(x), np.nanstd(x)


def paired(a, b, label_a, label_b):
    """双侧配对 t 检验，a - b。"""
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return (label_a, label_b, np.mean(d), p, int(np.sum(d > 0)), len(d))


L = []
P = L.append

P('# §6.3 数值底稿（由 `summary_63.py` 从 `results_dgfdb/*.log` 重算）\n')
P('口径：best = 逐 epoch `tgt_acc` 的最大值；last10 = 最后 10 次评估的均值；'
  'final = 最后一次评估。单位 %。\n')

# ---------------------------------------------------------------- A. α 扫描
SWEEP = [('gw_sw_a00', '0.0'), ('gw_sw_a05', '0.5'), ('gw_alias1', '1.0'),
         ('gw_sw_a15', '1.5'), ('gw_alias2', '2.0'), ('gw_sw_a30', '3.0')]

P('## A. α 扫描（T0/T9 × 5 种子）\n')
P('| α | 日志名 | T0 best | T9 best | T0 last10 | T9 last10 | 池化(10) best |')
P('|---|---|---|---|---|---|---|')
data = {}
for name, a in SWEEP:
    d = collect(TASKS_SWEEP, name)
    data[name] = d
    b_all = vec(d, TASKS_SWEEP, 'best', name)
    P('| {} | `{}` | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} | '
      '{:.2f} ± {:.2f} |'.format(
          a, name,
          *ms(vec(d, ['T0'], 'best', name)),
          *ms(vec(d, ['T9'], 'best', name)),
          np.nanmean(vec(d, ['T0'], 'last10', name)),
          np.nanmean(vec(d, ['T9'], 'last10', name)),
          *ms(b_all)))
P('')

d_anchor = collect(TASKS_SWEEP, 'gw_wide_gabor')
P('  dc_frac=0.5 的 α=0 锚点（`gw_wide_gabor`）：T0 {:.2f} ± {:.2f}，T9 {:.2f} ± {:.2f}\n'.format(
    *ms(vec(d_anchor, ['T0'], 'best', 'gw_wide_gabor')),
    *ms(vec(d_anchor, ['T9'], 'best', 'gw_wide_gabor'))))

P('### A.1 配对检验（vs α = 2.0，同任务同种子）\n')
P('| 对比 | n | Δ (T0) | p (T0) | wins | Δ (T9) | p (T9) | wins | Δ (池化) | p (池化) |')
P('|---|---|---|---|---|---|---|---|---|---|')
ref = data['gw_alias2']
for name, a in SWEEP:
    if name == 'gw_alias2':
        continue
    r0 = paired(vec(data[name], ['T0'], 'best', name),
                vec(ref, ['T0'], 'best', name), a, '2.0')
    r9 = paired(vec(data[name], ['T9'], 'best', name),
                vec(ref, ['T9'], 'best', name), a, '2.0')
    rp = paired(vec(data[name], TASKS_SWEEP, 'best', name),
                vec(ref, TASKS_SWEEP, 'best', name), a, '2.0')
    P('| α={} vs 2.0 | {} | {:+.2f} | {:.3g} | {}/{} | {:+.2f} | {:.3g} | {}/{} | '
      '{:+.2f} | {:.3g} |'.format(a, r0[5], r0[2], r0[3], r0[4], r0[5],
                                  r9[2], r9[3], r9[4], r9[5], rp[2], rp[3]))
P('')

P('### A.2 配对检验（vs α = 0.0，与 Fig. 11 图注同一基准）\n')
P('| 对比 | Δ (T0, p, wins) | Δ (T9, p, wins) |')
P('|---|---|---|')
anch = data['gw_sw_a00']
for name, a in SWEEP[1:]:
    r0 = paired(vec(data[name], ['T0'], 'best', name),
                vec(anch, ['T0'], 'best', 'gw_sw_a00'), a, '0.0')
    r9 = paired(vec(data[name], ['T9'], 'best', name),
                vec(anch, ['T9'], 'best', 'gw_sw_a00'), a, '0.0')
    P('| α={} vs 0.0 | {:+.2f} ({:.3g}, {}/5) | {:+.2f} ({:.3g}, {}/5) |'.format(
        a, r0[2], r0[3], r0[4], r9[2], r9[3], r9[4]))
P('')

# 单调性：α 与 T9 的相关
al = np.array([float(a) for _, a in SWEEP])
t9 = np.array([np.nanmean(vec(data[n], ['T9'], 'best', n)) for n, _ in SWEEP])
t0 = np.array([np.nanmean(vec(data[n], ['T0'], 'best', n)) for n, _ in SWEEP])
P('  α 与 T9 均值：Pearson r = {:.3f} (p = {:.3f})，Spearman r = {:.3f} (p = {:.3f})'.format(
    *stats.pearsonr(al, t9)[:2], *stats.spearmanr(al, t9)[:2]))
P('  α 与 T0 均值：Pearson r = {:.3f} (p = {:.3f})'.format(*stats.pearsonr(al, t0)[:2]))
P('  T0 极差 = {:.2f} 点（{:.2f}–{:.2f}）；T9 极差 = {:.2f} 点（{:.2f}–{:.2f}）\n'.format(
    t0.max() - t0.min(), t0.min(), t0.max(), t9.max() - t9.min(), t9.min(), t9.max()))

# ------------------------------------------------------- B. 初始化 / DC 对照
INIT = [('gw_wide_gabor', 'analytic Gabor, dc=0.5'),
        ('gw_wide_rand', 'random init, dc=0.5'),
        ('gw_mix', 'half Gabor + half random, dc=0.2'),
        ('gw_dc20', 'analytic Gabor, dc=0.2'),
        ('gw_alias2', 'aliased Gabor α=2, dc=0.3 (delivered DC setting)')]
P('## B. 初始化与 DC 处理对照（T0/T9 × 5 种子，同为 965K 宽频段前端）\n')
P('| 变体 | 配置 | T0 best | T9 best |')
P('|---|---|---|---|')
initd = {}
for name, desc in INIT:
    d = collect(TASKS_SWEEP, name)
    initd[name] = d
    P('| `{}` | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(
        name, desc, *ms(vec(d, ['T0'], 'best', name)),
        *ms(vec(d, ['T9'], 'best', name))))
P('')
P('### B.1 配对检验\n')
P('| 对比 | Δ (T0, p, wins) | Δ (T9, p, wins) |')
P('|---|---|---|')
for x, y in [('gw_wide_gabor', 'gw_wide_rand'), ('gw_wide_gabor', 'gw_mix'),
             ('gw_wide_gabor', 'gw_dc20'), ('gw_alias2', 'gw_wide_gabor')]:
    r0 = paired(vec(initd[x], ['T0'], 'best', x), vec(initd[y], ['T0'], 'best', y), x, y)
    r9 = paired(vec(initd[x], ['T9'], 'best', x), vec(initd[y], ['T9'], 'best', y), x, y)
    P('| {} vs {} | {:+.2f} ({:.3g}, {}/5) | {:+.2f} ({:.3g}, {}/5) |'.format(
        x, y, r0[2], r0[3], r0[4], r9[2], r9[3], r9[4]))
P('')
P('### B.2 混叠 vs 随机弥散（决定性对照，同任务同种子）\n')
for t in TASKS_SWEEP:
    a = vec(initd['gw_alias2'], [t], 'best', 'gw_alias2')
    for y in ['gw_wide_rand', 'gw_mix']:
        rr = paired(a, vec(initd[y], [t], 'best', y), 'gw_alias2', y)
        P('  {}: gw_alias2 − {} = {:+.2f} (p = {:.3g}, {}/5)'.format(
            t, y, rr[2], rr[3], rr[4]))
P('')

# ------------------------------------------------------------- C. 低频扩展
LOWF = [('v4_noL', '512, ksize64, f_lo=0.005'),
        ('sa_lowf512', '512, ksize128, f_lo=0.0025'),
        ('sa_cap768', '768, ksize64, f_lo=0.005'),
        ('sa_cap768_lowf', '768, ksize128, f_lo=0.0025')]
P('## C. 低频扩展对照（f_lo 0.005→0.0025，首层核 64→128；T0/T4/T9 × 5 种子）\n')
P('| 变体 | 配置 | T0 | T4 | T9 | 池化(15) |')
P('|---|---|---|---|---|---|')
lowd = {}
for name, desc in LOWF:
    d = collect(TASKS3, name)
    lowd[name] = d
    P('| `{}` | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | '
      '{:.2f} ± {:.2f} |'.format(
          name, desc,
          *ms(vec(d, ['T0'], 'best', name)),
          *ms(vec(d, ['T4'], 'best', name)),
          *ms(vec(d, ['T9'], 'best', name)),
          *ms(vec(d, TASKS3, 'best', name))))
P('')
P('### C.1 同档配对检验（低频扩展 − 原设置）\n')
P('| 档位 | Δ (T0, p) | Δ (T4, p) | Δ (T9, p) | Δ (池化, p, wins) |')
P('|---|---|---|---|---|')
P('  [交叉校验] `gw_alias2`（T0/T9，tau_mode=normal）vs `v4_noL`（tau_mode=zero）：')
for t in ['T0', 'T9']:
    a = vec(initd['gw_alias2'], [t], 'best', 'gw_alias2')
    b = vec(lowd['v4_noL'], [t], 'best', 'v4_noL')
    rr = paired(a, b, 'gw_alias2', 'v4_noL')
    P('    {}: {:+.2f} (p = {:.3g}, 均值 {:.2f} vs {:.2f})'.format(
        t, rr[2], rr[3], np.nanmean(a), np.nanmean(b)))
P('')
for x, y, tag in [('sa_lowf512', 'v4_noL', '512'),
                  ('sa_cap768_lowf', 'sa_cap768', '768')]:
    r = []
    for t in TASKS3:
        rr = paired(vec(lowd[x], [t], 'best', x), vec(lowd[y], [t], 'best', y), x, y)
        r.append('{:+.2f} ({:.3g})'.format(rr[2], rr[3]))
    rp = paired(vec(lowd[x], TASKS3, 'best', x), vec(lowd[y], TASKS3, 'best', y), x, y)
    P('| {} | {} | {} | {} | {:+.2f} ({:.3g}, {}/15) |'.format(
        tag, r[0], r[1], r[2], rp[2], rp[3], rp[4]))
P('')

# --------------------------------------------------------------- D. 容量
CAP = [('sa_lowf512', '512'), ('sa_cap768_lowf', '768'), ('sa_cap1024', '1024')]
P('## D. 前端容量（固定低频扩展前端，T0/T4/T9 × 5 种子）\n')
P('| 变体 | 前端通道 | T0 | T4 | T9 | 池化(15) |')
P('|---|---|---|---|---|---|')
capd = {}
for name, tag in CAP:
    d = lowd.get(name) or collect(TASKS3, name)
    capd[name] = d
    P('| `{}` | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | '
      '{:.2f} ± {:.2f} |'.format(
          name, tag,
          *ms(vec(d, ['T0'], 'best', name)),
          *ms(vec(d, ['T4'], 'best', name)),
          *ms(vec(d, ['T9'], 'best', name)),
          *ms(vec(d, TASKS3, 'best', name))))
P('')
P('### D.1 配对检验（vs 512）\n')
P('| 对比 | Δ (T0, p) | Δ (T4, p) | Δ (T9, p) | Δ (池化, p, wins) |')
P('|---|---|---|---|---|')
for name, tag in CAP[1:]:
    r = []
    for t in TASKS3:
        rr = paired(vec(capd[name], [t], 'best', name),
                    vec(capd['sa_lowf512'], [t], 'best', 'sa_lowf512'), tag, '512')
        r.append('{:+.2f} ({:.3g})'.format(rr[2], rr[3]))
    rp = paired(vec(capd[name], TASKS3, 'best', name),
                vec(capd['sa_lowf512'], TASKS3, 'best', 'sa_lowf512'), tag, '512')
    P('| {} vs 512 | {} | {} | {} | {:+.2f} ({:.3g}, {}/15) |'.format(
        tag, r[0], r[1], r[2], rp[2], rp[3], rp[4]))
P('')

# ------------------------------------------------- 交付配置在同一协议下的位置
P('## E. 交付配置与探针对照（仅作参照，预算不同）\n')
P('| 变体 | 预算 | T0 best | T9 best |')
P('|---|---|---|---|')
for name, desc in [('gw_alias2', '480 步（探针协议）'),
                   ('sc_steps6x', '2,880 步（交付）')]:
    d = collect(TASKS_SWEEP, name)
    P('| `{}` | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(
        name, desc, *ms(vec(d, ['T0'], 'best', name)),
        *ms(vec(d, ['T9'], 'best', name))))
P('')

# ------------------------------------------- F. 参照设计 WIDAN 的前端拆解
WSE = list(range(10))
P('## F. 参照设计（WIDAN）前端拆解（T0/T9 × 10 种子）\n')
P('| 变体 | T0 best | T9 best |')
P('|---|---|---|')
wd = {}
for tag in ['wid_wavelet', 'wid_rand', 'cap_full', 'cap_half', 'cap_quarter']:
    for t in TASKS_SWEEP:
        vals = []
        for s in WSE:
            v = wseries(t, tag, s)
            vals.append(np.nan if v is None else max(v))
        wd[(tag, t)] = np.array(vals, dtype=float)
    P('| `{}` | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(
        tag, *ms(wd[(tag, 'T0')]), *ms(wd[(tag, 'T9')])))
P('')
P('### F.1 配对检验（n = 10）\n')
P('| 对比 | Δ (T0, p, wins) | Δ (T9, p, wins) |')
P('|---|---|---|')
for x, y in [('wid_wavelet', 'wid_rand'), ('cap_half', 'cap_full'),
             ('cap_quarter', 'cap_full'), ('cap_half', 'cap_quarter')]:
    r0 = paired(wd[(x, 'T0')], wd[(y, 'T0')], x, y)
    r9 = paired(wd[(x, 'T9')], wd[(y, 'T9')], x, y)
    P('| {} vs {} | {:+.2f} ({:.3g}, {}/10) | {:+.2f} ({:.3g}, {}/10) |'.format(
        x, y, r0[2], r0[3], r0[4], r9[2], r9[3], r9[4]))
P('')

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\n[OK] -> {}'.format(REPORT))
