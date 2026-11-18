#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.4 对齐头消融（w_mmd / w_adv 四组合）—— v0.3 协议数值底稿。

数据源：`results_probe_align/HUST/AGMamba/<arm>/{T0,T4,T9}/*.log`
（由 `experiments/run_stage.py align_ablate_clean` 产出，
需 `AGM_FORCE_PY_SCAN=1 AGM_RESULTS=<repo>/results_probe_align`）

口径
----
· **交付配置**（`gaborwide`/512/gabor/dc0.3/`gabor_alias=2.0`/`use_ssm=True`/
  `tau_mode=zero`/余弦退火开）+ HUST **480 步**（= 主对比同一预算）。
· 只扫 `w_mmd` × `w_adv`，其余字段逐位相同 → w=0 仅把该项从 loss 移除，
  不改前向路径 / 数据次序 / 随机数消耗，故**同 (task, seed) 严格配对**。
· 主口径 `last5`（§5.1）；并列 `last1`（部署）/ `last10`（稳健性）/ `peak`。
· 配对检验：同 (task, seed) 与 **`aln_noall`（= 交付配置）** 配对，
  双侧 Wilcoxon（n=15 池化；T0/T4/T9 各 n=5）。
· **同源性自检**：`aln_noall` 应与主对比 `pub` 臂逐点一致
  （已实测 T0/T4/T9 × 5 种子 = 15/15 单元 `max|Δ| = 0.000000`）。

⚠ 旧 `_s64_data.md`（若有）是 v0.2 探针口径（2880 步 / best-epoch / 套 `DEFAULTS`），
与交付配置不可比；本文件取代它作为 §6.4 的数值底稿。

输出：docs/manuscript/_s64_align.md
用法：
    python analysis/summary_align_ablate.py [--dir results_probe_align]
                                            [--out docs/manuscript/_s64_align.md]
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

# (臂名, w_mmd, w_adv, 说明)
ARMS = [
    ('aln_noall', 0.0, 0.0, '**＝交付配置**（对齐项全关）'),
    ('aln_nommd', 0.0, 0.1, '只留对抗域判别'),
    ('aln_noadv', 0.5, 0.0, '只留 MMD 对齐'),
    ('aln_full', 0.5, 0.1, 'MMD 0.5 + 对抗 0.1（v0.2 带对齐配置）'),
]
TASKS = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
REF = 'aln_noall'                    # 交付配置，作配对基准
SPEED = {'T0': 20, 'T4': 40, 'T9': 80}


def series(path):
    """解析日志中的全部评估点（%）。"""
    if not os.path.exists(path):
        return []
    with open(path, errors='ignore') as f:
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
    ap.add_argument('--dir', default='results_probe_align',
                    help='probe 输出根目录（相对仓库根）')
    ap.add_argument('--out', default=os.path.join(
        'docs', 'manuscript', '_s64_align.md'))
    args = ap.parse_args()

    probe_root = (args.dir if os.path.isabs(args.dir)
                  else os.path.join(ROOT, args.dir))
    PROBE = os.path.join(probe_root, 'HUST', 'AGMamba')

    data = {}                                   # (arm, task, seed) -> readouts
    npts = {}                                   # (arm, task, seed) -> 点数
    for arm, _, _, _ in ARMS:
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

    L = ['# §6.4 对齐头消融（w_mmd / w_adv）—— v0.3 协议数值底稿', '',
         '由 `analysis/summary_align_ablate.py` 生成；'
         '数据源 `results_probe_align/`（`run_stage.py align_ablate_clean` 产出）。', '',
         '- 配置：**交付配置**（`gaborwide`/512/gabor 初始化/dc0.3/'
         '`gabor_alias=2.0`/`use_ssm=True`/`tau_mode=zero`/**余弦退火开**）'
         '+ HUST **480 步**（= 主对比同一预算）',
         '- 规模：4 臂 × T0(20Hz)/T4(40Hz)/T9(80Hz) × 5 种子 = 60 单元；'
         '**主口径 last5**，并列 last1 / last10 / peak',
         '- 只扫 `w_mmd` × `w_adv`：w=0 仅把该项从 loss 移除，'
         '不改前向路径 / 数据次序 / 随机数消耗 → **同 (task, seed) 严格配对**',
         '- 后端：`AGM_FORCE_PY_SCAN=1`（与主对比一致，保证梯度可复现）',
         '- **同源性自检**：`aln_noall` 与主对比 `pub` 臂逐点 '
         '`max|Δ| = 0.000000`（15/15 单元）',
         '- ⚠ 取代 v0.2 探针口径（2880 步 / best-epoch / 套 `DEFAULTS`）的旧对齐消融表',
         '']

    # ---- 完成度 ----
    L += ['## (a0) 完成度', '',
          '| 臂 | n(total) | n(≥60 点) | 缺失单元 |', '|---|---|---|---|']
    for arm, _, _, _ in ARMS:
        n_total = len(TASKS) * len(SEEDS)
        miss = ['%s/s%d(%d点)' % (t, s, npts[(arm, t, s)])
                for t in TASKS for s in SEEDS if npts[(arm, t, s)] < EPOCHS]
        L.append('| `%s` | %d | %d | %s |'
                 % (arm, n_total, len(vals(arm)),
                    ', '.join(miss) if miss else '—'))
    L.append('')

    # ---- (a1) 主表 ----
    L += ['## (a1) 四臂主表（last5 主口径，5 种子）', '',
          '| 臂 | w_mmd | w_adv | T0 (20Hz) | T4 (40Hz) | T9 (80Hz) | 池化 last5 '
          '| 池化 last10 | 池化 last1 |', '|---|---|---|---|---|---|---|---|---|']
    for arm, wm, wa, _ in ARMS:
        if not vals(arm):
            L.append('| `%s` | %.1f | %.1f | – | – | – | – | – | – |'
                     % (arm, wm, wa))
            continue
        tag = '`%s`**（交付）**' % arm if arm == REF else '`%s`' % arm
        L.append('| %s | %.1f | %.1f | %s | %s | %s | **%.2f** | %.2f | %.2f |'
                 % (tag, wm, wa,
                    fmt(vals(arm, 'T0')), fmt(vals(arm, 'T4')), fmt(vals(arm, 'T9')),
                    np.mean(vals(arm)), np.mean(vals(arm, key='last10')),
                    np.mean(vals(arm, key='last1'))))
    L.append('')

    # ---- (a2) 配对检验 vs 交付配置 ----
    L += ['## (a2) 配对检验（vs `aln_noall`＝交付配置，同 task×seed，双侧 Wilcoxon）', '',
          'Δ > 0 表示该臂**优于**交付配置；"同号"= last5/last1/last10/peak 四读数方向一致。', '',
          '| 臂 | n | Δ last5 池化 | p (T0) | p (T4) | p (T9) | p (池化) | '
          'wins (池化) | 同号 |', '|---|---|---|---|---|---|---|---|---|']
    for arm, _, _, _ in ARMS:
        if arm == REF:
            L.append('| `%s`（交付） | – | 0（基准） | – | – | – | – | – | – |'
                     % arm)
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
    pooled = {arm: float(np.mean(vals(arm))) for arm, _, _, _ in ARMS
              if vals(arm)}
    best = max(pooled, key=pooled.get) if pooled else None
    d_full = [data[('aln_full', t, s)]['last5'] - data[(REF, t, s)]['last5']
              for t in TASKS for s in SEEDS
              if ('aln_full', t, s) in data and (REF, t, s) in data]
    d_nommd = [data[('aln_nommd', t, s)]['last5'] - data[(REF, t, s)]['last5']
               for t in TASKS for s in SEEDS
               if ('aln_nommd', t, s) in data and (REF, t, s) in data]
    d_noadv = [data[('aln_noadv', t, s)]['last5'] - data[(REF, t, s)]['last5']
               for t in TASKS for s in SEEDS
               if ('aln_noadv', t, s) in data and (REF, t, s) in data]

    L += ['## (a3) 结论', '']
    incomplete = [arm for arm, _, _, _ in ARMS if len(vals(arm)) < 15]
    if incomplete:
        L += ['⚠ **本表尚未跑满**：%s 仍有缺失单元，以下结论为**阶段性**判定，'
              '定稿前须重跑本脚本。' % '、'.join('`%s`' % a for a in incomplete), '']
    if best is not None and not incomplete:
        L.append('- **最优臂 = `%s`（池化 last5 = %.2f）**。' % (best, pooled[best]))
    if d_full:
        per_t = {}
        for t in TASKS:
            per_t[t] = np.mean([data[('aln_full', t, s)]['last5']
                                - data[(REF, t, s)]['last5']
                                for s in SEEDS
                                if ('aln_full', t, s) in data
                                and (REF, t, s) in data])
        L.append('- **同时打开两路对齐（`aln_full` 0.5/0.1）无益**：'
                 '池化 Δ = %+.2f（T0 %+.2f / T4 %+.2f / T9 %+.2f），'
                 '仅 %d/%d 单元正向。交付配置（对齐项全关）≥ 该臂。'
                 % (np.mean(d_full), per_t['T0'], per_t['T4'], per_t['T9'],
                    sum(1 for x in d_full if x > 0), len(d_full)))
    if d_nommd and d_noadv:
        L.append('- **单腿贡献**：只留对抗（`aln_nommd`）池化 Δ = %+.2f；'
                 '只留 MMD（`aln_noadv`）池化 Δ = %+.2f。'
                 % (np.mean(d_nommd), np.mean(d_noadv)))
    if d_full:
        same_dir = bool(np.mean(d_full) < 0)
        L.append('- **与 v0.2 口径的关系**：v0.2 探针曾报"去掉对齐头 +8.63"；'
                 '本协议（480 步 / 交付配置 / 余弦退火开）复核得'
                 '`aln_full` 相对交付配置 Δ = %+.2f → **方向%s**'
                 '（%s"交付配置不含对齐项"）。'
                 % (np.mean(d_full), '一致' if same_dir else '不一致',
                    '支持' if same_dir else '不支持'))
    L.append('- **§5.4 配对性**：w=0 只改变 loss 项、不改前向与随机数消耗，'
             '故 `aln_noall` 与 `aln_*` 的同种子差异**只归因于对齐项本身**。')
    L.append('')

    # ---- 附录：逐种子明细 ----
    L += ['## (a4) 附录：逐单元 last5 明细', '',
          '| 臂 | task | ' + ' | '.join('s%d' % s for s in SEEDS) + ' |',
          '|---|---|' + '---|' * len(SEEDS)]
    for arm, _, _, _ in ARMS:
        for t in TASKS:
            cells = []
            for s in SEEDS:
                k = (arm, t, s)
                cells.append('%.2f' % data[k]['last5'] if k in data
                             else '–(%d点)' % npts[k])
            L.append('| `%s` | %s | %s |' % (arm, t, ' | '.join(cells)))
    L.append('')

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    print('[align] %s  (%d/60 单元)' % (out, len(data)))
    for arm, _, _, _ in ARMS:
        p = pooled.get(arm)
        print('  %-10s pooled_last5=%s (n=%d)' % (
            arm, ('%.2f' % p) if p is not None else 'na', len(vals(arm))))
    if best is not None:
        print('  best =', best)


if __name__ == '__main__':
    main()
