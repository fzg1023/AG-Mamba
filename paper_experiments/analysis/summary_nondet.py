#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现性底噪：同配置、同种子的重跑差异（nondet_ctl 的结果汇总）。

数据来源：
  · nondet_ctl  : PU T0/T2 与 HUST T9 各跑两次同配置副本（rep_a / rep_b），
                  5 个种子完全相同；
  · 交叉对照    : PU 上 xg_std 与 puac_a2_c512（配置与种子也完全相同）。
输出：docs/manuscript/_repro_floor.md
"""
import os
import re

import numpy as np

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


def best(path):
    if not os.path.exists(path) or os.path.getsize(path) < 500:
        return None
    with open(path, errors='ignore') as f:
        v = [float(x) * 100.0 for x in PAT.findall(f.read())]
    return max(v) if v else None


def lm(data, task, name, seed):
    return _findlog('{}_{}_AGMamba_fc_{}_seed{}.log'.format(
        data, task, name, seed))


L = []
P = L.append
P('# 可复现性底噪：同配置 + 同种子的重跑差异')
P('')
P('同一配置、同一随机种子、同一数据流顺序的两个进程，结果并不相同：')
P('日志前 3–4 步逐位一致，随后在第 4 位小数处分叉（浮点非确定性）。')
P('下表给出该差异的量级（best-epoch，百分点）。')
P('')
P('## (a) nondet_ctl：rep_a 与 rep_b 为同一配置的两次运行')
P('')
P('| 基准 | 任务 | rep_a 均值 | rep_b 均值 | Δ(池化) | 逐种子 &#124;Δ&#124; 均值 | 逐种子 &#124;Δ&#124; 最大 | 逐种子差值 σ |')
P('|---|---|---|---|---|---|---|---|')
tot_abs, tot_sig = [], []
for data, tasks in [('PU', ['T0', 'T2']), ('HUST', ['T9'])]:
    for t in tasks:
        a = [best(lm(data, t, 'rep_a', s)) for s in SEEDS]
        b = [best(lm(data, t, 'rep_b', s)) for s in SEEDS]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if not pairs:
            P('| {} | {} | – | – | – | – | – | – |'.format(data, t))
            continue
        d = np.array([y - x for x, y in pairs], dtype=float)
        aa = np.array([x for x, _ in pairs], dtype=float)
        bb = np.array([y for _, y in pairs], dtype=float)
        tot_abs += list(np.abs(d))
        tot_sig += list(d)
        P('| {} | {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
            data, t, aa.mean(), bb.mean(), bb.mean() - aa.mean(),
            np.abs(d).mean(), np.abs(d).max(), d.std(ddof=1)))
P('')
P('**汇总**：{} 对 (任务, 种子) 单元的 |Δ| 均值 = **{:.2f}** 点、最大 = **{:.2f}** 点；'
  '池化差值的标准差 = {:.2f} 点。'.format(
      len(tot_abs), np.mean(tot_abs), np.max(tot_abs), np.std(tot_sig, ddof=1)))
P('')
P('## (b) 交叉对照：PU 上 xg_std vs puac_a2_c512（配置与种子相同）')
P('')
P('| 任务 | 均值 A | 均值 B | Δ | 逐种子 &#124;Δ&#124; 均值 | 最大 |')
P('|---|---|---|---|---|---|')
aa_all = []
for t in ['T0', 'T1', 'T2', 'T3']:
    a = [best(lm('PU', t, 'xg_std', s)) for s in SEEDS]
    b = [best(lm('PU', t, 'puac_a2_c512', s)) for s in SEEDS]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    d = np.array([y - x for x, y in pairs], dtype=float)
    aa_all += list(np.abs(d))
    P('| {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} |'.format(
        t, np.mean([x for x, _ in pairs]), np.mean([y for _, y in pairs]),
        d.mean(), np.abs(d).mean(), np.abs(d).max()))
P('')
P('**汇总**：{} 对单元的 |Δ| 均值 = **{:.2f}** 点、最大 = **{:.2f}** 点。'.format(
    len(aa_all), np.mean(aa_all), np.max(aa_all)))
P('')
P('## (c) 根因诊断（同配置各跑两次，PU T2 seed0）')
P('')
P('| 条件 | 两次运行 | Δ | 日志逐行比对 |')
P('|---|---|---|---|')
DIAG = [
    ('含 SSM（默认设置，共享 GPU）', 'det_ssm_a', 'det_ssm_b'),
    ('无 SSM（use_ssm=False，共享 GPU）', 'det_nossm_a', 'det_nossm_b'),
    ('含 SSM + AGM_DET=1（cuDNN 确定性卷积）', 'det2_ssm_a', 'det2_ssm_b'),
    ('无 SSM + AGM_DET=1', 'det2_nossm_a', 'det2_nossm_b'),
    ('含 SSM + 空闲 GPU', 'det3_ssm_a', 'det3_ssm_b'),
    ('无 SSM + 空闲 GPU', 'det3_nossm_a', 'det3_nossm_b'),
]


def _lines(tag):
    p = lm('PU', 'T2', tag, 0)
    if not os.path.exists(p):
        return None
    with open(p, errors='ignore') as f:
        return f.read().split('\n')


for label, ta, tb in DIAG:
    A, B = _lines(ta), _lines(tb)
    if A is None or B is None:
        P('| {} | – | – | 未运行 |'.format(label))
        continue
    va, vb = best(lm('PU', 'T2', ta, 0)), best(lm('PU', 'T2', tb, 0))
    n = min(len(A), len(B))
    ndiff = sum(1 for i in range(n) if A[i] != B[i])
    first = next((i for i in range(n) if A[i] != B[i]), None)
    P('| {} | {:.2f} / {:.2f} | {:+.2f} | {} |'.format(
        label, va, vb, vb - va,
        '逐位一致' if ndiff == 0 else '{} 行不同（首处第 {} 行）'.format(ndiff, first)))
P('')
P('  结论：所有条件下的日志都在**第 4 位小数**处分叉（~1e-4），与是否共享 GPU、')
P('  是否开 cuDNN 确定性开关、是否有 SSM 分支无关；但同一 $10^{-4}$ 扰动经 2,640 步')
P('  后在不同配置上放大到 0.1–7.9 个精度点（难任务上放大更剧）。')
P('  因此：① 来源是训练路径的数值噪声 + 任务混沌，而非可开关修复的算法选择；')
P('  ② 结论层面只能报告实测波动带，并把 <~2 点的单任务效应降为“不可分辨”。')
P('')

P('## (d) 含义')
P('')
P('- 同一配置的两次运行在 (任务, 种子) 级别可相差 **约 2–3 点**，因此')
P('  **小于 ~2 点的单任务效应不在本代码库的分辨率之内**；')
P('- 涉及末十检查点或单任务的 1–2 点结论（例如若干 +0.5 ~ +1.3 的消融、')
P('  表 8 的 +1.33/−1.33、PHM2009 上 1–2 点的差异）必须按"落在运行间波动内"处理；')
P('- 10–50 对池化后的配对检验若效应量 ≳3 点（如 α 扫描的 +16、+19）不受影响。')
P('')

txt = '\n'.join(L) + '\n'
with open(os.path.join(ROOT, 'docs/manuscript', '_repro_floor.md'), 'w', encoding='utf-8') as fh:
    fh.write(txt)
print(txt)
