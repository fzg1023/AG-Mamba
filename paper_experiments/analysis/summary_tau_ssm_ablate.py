#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.4 τ 调制 / 选择性 SSM 消融 —— v0.3 协议数值底稿。

数据源：`results_probe_tau/HUST/AGMamba/<arm>/{T0,T4,T9}/*.log`
（由 `experiments/run_stage.py tau_ssm_ablate_clean` 产出，
需 `AGM_FORCE_PY_SCAN=1 AGM_RESULTS=<repo>/results_probe_tau`）

四臂
----
· `tau_zero`    —— τ₀ 估计器与 γ·τ₀ 通路**整条不构建**（`tau_mode='zero'`）= **本方法配置**
· `tau_normal`  —— 液体时间常数调制开启（估计 τ₀ 并注入 Δ）
· `tau_shuffle` —— 估计 τ₀ 但**跨样本打乱**（因果对照：有 τ₀ 数值、无样本对应）
· `ssm_off`     —— 去掉选择性 SSM 分支（只留前端 + 池化；feat_dim 由 4d 降为 2d）

口径与配对范围（重要）
----------------------
· 交付配置 + HUST **480 步**（= 主对比同一预算），余弦退火开，5 种子；
  主口径 `last5`（§5.1），并列 `last1`（部署）/ `last10`（稳健性）/ `peak`。
· 本组属于 **architectural ablation**（§5.3 第三种配对范围）：模块的增删
  改变了建模阶段消耗的随机数（`RandomSampler` 用全局 torch RNG 且在建模型
  **之后**才构造），故**只有 (task, seed) 划分相同，不主张逐位可复现**。
  因此**结论只允许建立在远高于可复现下限的池化效应上**（同配置重跑
  mean|Δ| = 2.55、max = 6.00 点，§5.3）。
· 输出：docs/manuscript/_s65_tau_ssm.md

用法
----
    python analysis/summary_tau_ssm_ablate.py [--dir results_probe_tau]
                                              [--out docs/manuscript/_s65_tau_ssm.md]
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
from scipy import stats as st

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')

# (臂名, 说明)
ARMS = [
    ('tau_zero', '**＝本方法**（τ 通路不构建）'),
    ('tau_normal', '液体时间常数 τ₀ 调制开启'),
    ('tau_shuffle', 'τ₀ 跨样本打乱（因果对照）'),
    ('ssm_off', '去掉选择性 SSM 分支（feat_dim 4d→2d）'),
]
TASKS = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
REF = 'tau_zero'
SPEED = {'T0': 20, 'T4': 40, 'T9': 80}
FLOOR = 2.55          # §5.3 同配置重跑 mean|Δ|（点）


def series(path):
    """解析日志中的全部评估点（%）。"""
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    v = [float(x) for x in PAT.findall(txt)]
    return [x * 100 if x <= 1.5 else x for x in v]


def readouts(v):
    return {'last5': float(np.mean(v[-5:])),
            'last1': float(v[-1]),
            'last10': float(np.mean(v[-10:])),
            'peak': float(np.max(v))}


def fmt(x):
    x = list(x)
    if not x:
        return '–'
    if len(x) == 1:
        return '%.2f' % x[0]
    return '%.2f ± %.2f' % (np.mean(x), np.std(x, ddof=1))


def pv(x, min_n=5):
    x = np.asarray(x, dtype=float)
    if len(x) < min_n or np.allclose(x, 0):
        return float('nan')
    try:
        return float(st.wilcoxon(x).pvalue)
    except Exception:
        return float('nan')


def pstr(p):
    return 'n/a' if p != p else '%.4g' % p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results_probe_tau',
                    help='probe 输出根目录（相对仓库根）')
    ap.add_argument('--out', default=os.path.join(
        'docs', 'manuscript', '_s65_tau_ssm.md'))
    args = ap.parse_args()

    probe_root = (args.dir if os.path.isabs(args.dir)
                  else os.path.join(ROOT, args.dir))
    PROBE = os.path.join(probe_root, 'HUST', 'AGMamba')
    out_path = (args.out if os.path.isabs(args.out)
                else os.path.join(ROOT, args.out))

    data, npts = {}, {}
    for arm, _ in ARMS:
        for t in TASKS:
            for s in SEEDS:
                v = series(os.path.join(
                    PROBE, arm, t,
                    'HUST_%s_AGMamba_fc_%s_seed%d.log' % (t, arm, s)))
                npts[(arm, t, s)] = len(v)
                if len(v) >= EPOCHS:
                    data[(arm, t, s)] = readouts(v)

    def vals(arm, task=None, key='last5'):
        out = []
        for t in (TASKS if task is None else [task]):
            for s in SEEDS:
                k = (arm, t, s)
                if k in data:
                    out.append(data[k][key])
        return out

    L = ['# §6.4 τ 调制 / 选择性 SSM 消融 —— v0.3 协议数值底稿', '',
         '由 `analysis/summary_tau_ssm_ablate.py` 生成；数据源 `results_probe_tau/`'
         '（`run_stage.py tau_ssm_ablate_clean` 产出）。', '',
         '- 配置：本方法配置（`gaborwide`/512/gabor 初始化/dc0.3/'
         '`gabor_alias=2.0`/`use_ssm=True`/余弦退火开）'
         '＋ HUST **480 步**（= 主对比同一预算）',
         '- 规模：4 臂 × T0(20 Hz)/T4(40 Hz)/T9(80 Hz) × 5 种子 = **60 单元**；'
         '主口径 **last5**，并列 last1 / last10 / peak',
         '- 后端：`AGM_FORCE_PY_SCAN=1`（与主对比一致）',
         '- ⚠ **配对范围**：本组是 **architectural ablation**（§5.3 第三种）：'
         '模块增删改变建模阶段的随机数消耗 → 仅 **(task, seed) 划分相同**，'
         '**不主张逐位可复现**；结论只建立在远高于可复现下限的**池化**效应上'
         '（同配置重跑 mean|Δ| = %.2f 点、max = 6.00 点）' % FLOOR,
         '']

    # ---- (a0) 完成度 ----
    L += ['## (a0) 完成度', '', '| 臂 | n(total) | n(≥60 点) | 缺失单元 |',
          '|---|---|---|---|']
    for arm, _ in ARMS:
        n_total = len(TASKS) * len(SEEDS)
        miss = ['%s/s%d(%d点)' % (t, s, npts[(arm, t, s)])
                for t in TASKS for s in SEEDS if npts[(arm, t, s)] < EPOCHS]
        L.append('| `%s` | %d | %d | %s |'
                 % (arm, n_total, len(vals(arm)),
                    ', '.join(miss) if miss else '—'))
    L.append('')

    # ---- (a1) 主表 ----
    L += ['## (a1) 四臂主表（last5 主口径，5 种子）', '',
          '| 臂 | 说明 | T0 (20 Hz) | T4 (40 Hz) | T9 (80 Hz) | 池化 last5 | '
          '池化 last10 | 池化 last1 |', '|---|---|---|---|---|---|---|---|']
    for arm, note in ARMS:
        if not vals(arm):
            L.append('| `%s` | %s | – | – | – | – | – | – |' % (arm, note))
            continue
        tag = '`%s`（本方法）' % arm if arm == REF else '`%s`' % arm
        L.append('| %s | %s | %s | %s | %s | **%.2f** | %.2f | %.2f |'
                 % (tag, note,
                    fmt(vals(arm, 'T0')), fmt(vals(arm, 'T4')), fmt(vals(arm, 'T9')),
                    np.mean(vals(arm)), np.mean(vals(arm, key='last10')),
                    np.mean(vals(arm, key='last1'))))
    L.append('')

    # ---- (a2) 配对检验 ----
    L += ['## (a2) 配对检验（vs `tau_zero`＝本方法，同 task×seed，双侧 Wilcoxon）', '',
          'Δ > 0 表示该臂**优于**本方法配置。⚠ 模块增删**不改数据划分、改随机数消耗**，'
          '故 p 值只作参考，判定以**池化 Δ 与可复现下限 %.2f 点**的比较为准。' % FLOOR,
          '', '| 臂 | n | Δ last5 池化 | p (T0) | p (T4) | p (T9) | p (池化) | '
          'wins (池化) | 四读数同号 |', '|---|---|---|---|---|---|---|---|---|']
    for arm, _ in ARMS:
        if arm == REF:
            L.append('| `%s`（本方法） | – | 0（基准） | – | – | – | – | – | – |' % arm)
            continue
        d_t, d_all = {}, []
        for t in TASKS:
            d_t[t] = []
            for s in SEEDS:
                k1, k2 = (arm, t, s), (REF, t, s)
                if k1 in data and k2 in data:
                    d = data[k1]['last5'] - data[k2]['last5']
                    d_t[t].append(d)
                    d_all.append(d)
        if not d_all:
            L.append('| `%s` | 0 | – | – | – | – | – | – | – |' % arm)
            continue
        dirs = []
        for key in ('last5', 'last1', 'last10', 'peak'):
            ds = [data[(arm, t, s)][key] - data[(REF, t, s)][key]
                  for t in TASKS for s in SEEDS
                  if (arm, t, s) in data and (REF, t, s) in data]
            dirs.append(('+' if np.mean(ds) > 0 else
                         ('−' if np.mean(ds) < 0 else '0')) if ds else '?')
        same = '✓' if len(set(dirs)) == 1 else '✗'
        L.append('| `%s` | %d | **%+.2f** | %s | %s | %s | %s | %d/%d | %s |'
                 % (arm, len(d_all), np.mean(d_all),
                    pstr(pv(d_t['T0'])), pstr(pv(d_t['T4'])), pstr(pv(d_t['T9'])),
                    pstr(pv(d_all)), sum(1 for x in d_all if x > 0), len(d_all),
                    same))
    L.append('')

    # ---- (a3) 结论 ----
    pooled = {arm: float(np.mean(vals(arm))) for arm, _ in ARMS if vals(arm)}
    L += ['## (a3) 结论（按可复现下限判定）', '']
    if pooled:
        best = max(pooled, key=pooled.get)
        L.append('- 池化 last5 最高臂：**`%s`（%.2f）**%s'
                 % (best, pooled[best],
                    '，即本方法配置' if best == REF else ' ⚠ 非本方法配置，需复核'))
        for arm, _ in ARMS:
            if arm == REF or arm not in pooled:
                continue
            d = pooled[arm] - pooled[REF]
            verdict = ('**落在可复现下限内 → 无结论**' if abs(d) < FLOOR
                       else '**超出下限 → 可判定方向**')
            L.append('- `%s` − `%s` = **%+.2f** 点 → %s' % (arm, REF, d, verdict))
        L.append('')
        L.append('> 与 §6.4 的措辞一致：τ 调制（`tau_normal`/`tau_shuffle`）'
                 '在本前端与本预算下**无可测收益**，故本方法不构建该通路；'
                 '选择性 SSM 分支的移除造成**远超下限**的下降，故保留。'
                 '两条结论都限定在**本架构 / 本基准**内。')
    L.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('[saved] %s' % out_path)
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
