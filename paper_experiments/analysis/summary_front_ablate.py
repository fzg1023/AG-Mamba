#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.3 前端设计三组消融（初始化 / 低频扩展 / 容量）—— v0.3 协议数值底稿。

数据源：`results_probe_front/HUST/AGMamba/<arm>/{T0,T4,T9}/*.log`
（由 `experiments/run_stage.py frontend_ablate_clean` 产出，
需 `AGM_FORCE_PY_SCAN=1 AGM_RESULTS=<repo>/results_probe_front`）

**参照臂不在本目录**：交付配置的 480 步参照就是 `results_probe_align` 的
`aln_noall`（= 主对比 `pub` 臂，已核 15 单元逐点 max|Δ| = 0.000000）。
本脚本会再次独立核对这条同源性（见 (b) 节），核对失败则整体判定不可用。

五臂
----
· `fr_rand`    —— 首层随机初始化（同形状 Kaiming）
· `fr_mix`     —— 首层一半 Gabor + 一半随机
· `fr_lap`     —— 首层改用 WIDAN 参考实现的 Laplace 族（64 原子 × 核长 128）
· `fr_narrow`  —— 低频扩展的**反向**消融：回到 `f_lo=0.005` + 首层核 64
· `fr_cap768`  —— 前端宽度 512 → 768
· `fr_cap960`  —— 前端宽度 512 → 960（≈3.85 M，WIDAN 3.98 M 量级）

口径与配对范围（重要）
----------------------
· 交付配置 + HUST **480 步**（= 主对比同一预算），余弦退火开，5 种子；
  主口径 `last5`（§5.1），并列 `last1`（部署）/ `last10`（稳健性）/ `peak`。
· 本组属于 **architectural /initialisation ablation**：`front_chan` / `front_ksize`
  改变参数形状，而 Gabor 首层是 `copy_` 写入、随机与混合首层要走 Kaiming
  （**消耗 RNG**），因此下游各层拿到的随机权重也随之不同 —— **不是严格配对**，
  只共享 (task, seed) 划分。故结论只建立在**远高于可复现下限**的池化效应上
  （同配置重跑 mean|Δ| = 2.55、max = 6.00 点，§5.3）。
· 输出：docs/manuscript/_s63_front.md

用法
----
    python analysis/summary_front_ablate.py
    python analysis/summary_front_ablate.py --dir results_probe_front \\
        --ref-dir results_probe_align --ref-arm aln_noall
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
    ('fr_rand', '首层随机初始化（同形状 Kaiming）'),
    ('fr_mix', '首层 50% Gabor + 50% 随机'),
    ('fr_lap', '首层 Laplace 族（WIDAN 参考实现，64 原子 × 核长 128）'),
    ('fr_narrow', '窄带短核档：`f_lo=0.005`、首层核 64'),
    ('fr_cap768', '前端宽度 512 → 768'),
    ('fr_cap960', '前端宽度 512 → 960'),
]
# 参数量（可训练，num_classes=14）：由 logs/_preflight_front_arms.py 实测
PARAMS = {
    'fr_ref': 1374222, 'fr_rand': 1374222, 'fr_mix': 1374222,
    'fr_lap': 1374222, 'fr_narrow': 1370126,
    'fr_cap768': 2610126, 'fr_cap960': 3851550,
}
TASKS = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
SPEED = {'T0': 20, 'T4': 40, 'T9': 80}
FLOOR = 2.55          # §5.3 同配置重跑 mean|Δ|（点）
FLOOR_MAX = 6.00      # §5.3 同配置重跑 max|Δ|（点）


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
    ap.add_argument('--dir', default='results_probe_front')
    ap.add_argument('--ref-dir', default='results_probe_align')
    ap.add_argument('--ref-arm', default='aln_noall')
    ap.add_argument('--pub-dir', default='results',
                    help='主对比目录（用于参照臂同源性自检）')
    ap.add_argument('--out', default=os.path.join(
        'docs', 'manuscript', '_s63_front.md'))
    args = ap.parse_args()

    def absd(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    PROBE = os.path.join(absd(args.dir), 'HUST', 'AGMamba')
    REFD = os.path.join(absd(args.ref_dir), 'HUST', 'AGMamba')
    PUBD = os.path.join(absd(args.pub_dir), 'HUST', 'AGMamba')
    out_path = absd(args.out)

    def logpath(base, arm, t, s):
        return os.path.join(base, arm, t,
                            'HUST_%s_AGMamba_fc_%s_seed%d.log' % (t, arm, s))

    data, npts = {}, {}
    for arm, _ in ARMS:
        for t in TASKS:
            for s in SEEDS:
                v = series(logpath(PROBE, arm, t, s))
                npts[(arm, t, s)] = len(v)
                if len(v) >= EPOCHS:
                    data[(arm, t, s)] = readouts(v)
    ref_raw = {}
    for t in TASKS:
        for s in SEEDS:
            v = series(logpath(REFD, args.ref_arm, t, s))
            ref_raw[(t, s)] = v
            if len(v) >= EPOCHS:
                data[('ref', t, s)] = readouts(v)

    def vals(arm, task=None, key='last5'):
        out = []
        for t in (TASKS if task is None else [task]):
            for s in SEEDS:
                k = (arm, t, s)
                if k in data:
                    out.append(data[k][key])
        return out

    def complete(arm):
        """该臂是否 **15 个单元全部** 达到 EPOCHS 个评估点。

        ⚠ 不完整的臂**不得**报池化值：池化是在“可用单元”上取均值，而各目标的
        难度差很大（T0/T4 高、T9 低），缺了远端目标的“部分池化”会系统性偏高
        （实例：fr_cap960 完成 10/15 时部分池化为 94.98，远高于其真实水平）。
        """
        return all(npts.get((arm, t, s), 0) >= EPOCHS
                   for t in TASKS for s in SEEDS)

    # ---- 同源性自检：参照臂 vs 主对比 pub（逐点） ----
    prov = {'cells': 0, 'max': 0.0, 'short': []}
    for t in TASKS:
        for s in SEEDS:
            vb = series(logpath(PUBD, 'pub', t, s))
            va = ref_raw[(t, s)]
            if len(va) >= EPOCHS and len(vb) >= EPOCHS:
                prov['cells'] += 1
                prov['max'] = max(prov['max'], max(abs(x - y)
                                                   for x, y in zip(va, vb)))
            else:
                prov['short'].append('%s/s%d(%d,%d)' % (t, s, len(va), len(vb)))
    prov_ok = prov['cells'] == len(TASKS) * len(SEEDS) and prov['max'] < 1e-6

    L = ['# §6.3 前端设计消融（初始化 / 低频扩展 / 容量）—— v0.3 协议数值底稿',
         '',
         '由 `analysis/summary_front_ablate.py` 生成；数据源 `%s/`'
         '（`run_stage.py frontend_ablate_clean` 产出）。' % args.dir, '',
         '- 配置：**交付配置**（`gaborwide`/512/gabor 初始化/`f_lo=0.0025`/'
         '首层核 128/dc0.3/`gabor_alias=2.0`/无对齐头/`tau_mode=zero`/余弦退火开）'
         '，每臂只覆盖该臂列出的字段',
         '- 规模：%d 臂 × T0(20 Hz)/T4(40 Hz)/T9(80 Hz) × 5 种子 = **%d 单元**；'
         'HUST **480 步**（= 主对比同一预算）；主口径 **last5**，'
         '并列 last1 / last10 / peak' % (len(ARMS), len(ARMS) * 15),
         '- 后端：`AGM_FORCE_PY_SCAN=1`（与主对比一致）',
         '- ⚠ **配对范围**：本组改的是前端**结构与初始化** —— `front_chan`/'
         '`front_ksize` 改变参数形状，随机/混合首层要走 Kaiming（消耗 RNG），'
         '故下游各层的随机权重也随之不同 → **不是严格配对**，'
         '只共享 (task, seed) 划分；结论只建立在远高于可复现下限的**池化**效应上'
         '（同配置重跑 mean|Δ| = %.2f、max = %.2f 点）' % (FLOOR, FLOOR_MAX),
         '- ⚠ 本文件**取代** `_s63_data.md`（v0.2 探针：best-epoch + 2880 步 + '
         '带对齐头 + `tau_mode=normal` + dc 0.5 + 核 64/`f_lo` 0.005 + '
         '关余弦退火；来源 `results_dgfdb/` 已删除）；W3 已禁止其回流正文',
         '']

    # ---- (a0) 完成度 ----
    L += ['## (a0) 完成度', '', '| 臂 | n(total) | n(≥60 点) | 缺失/偏短单元 |',
          '|---|---|---|---|']
    for arm, _ in ARMS:
        n_total = len(TASKS) * len(SEEDS)
        miss = ['%s/s%d(%d点)' % (t, s, npts[(arm, t, s)])
                for t in TASKS for s in SEEDS if npts[(arm, t, s)] < EPOCHS]
        L.append('| `%s` | %d | %d | %s |'
                 % (arm, n_total, len(vals(arm)),
                    ', '.join(miss) if miss else '—'))
    L.append('| `ref` (%s = 本方法) | %d | %d | %s |'
             % (args.ref_arm, len(TASKS) * len(SEEDS), len(vals('ref')),
                ', '.join('%s/s%d' % (t, s) for t in TASKS for s in SEEDS
                          if ('ref', t, s) not in data) or '—'))
    L.append('')

    # ---- (a1) 主表 ----
    L += ['## (a1) 主表（last5 主口径，5 种子）', '',
          '| 臂 | 说明 | 可训练参数 | T0 (20 Hz) | T4 (40 Hz) | T9 (80 Hz) | '
          '池化 last5 | 池化 last10 | 池化 last1 |',
          '|---|---|---|---|---|---|---|---|---|']
    L.append('| **本方法**（交付配置） | 参照 `%s`（%s） | %.2f M | %s | %s | %s | '
             '**%.2f** | %.2f | %.2f |'
             % (args.ref_arm, args.ref_dir, PARAMS['fr_ref'] / 1e6,
                fmt(vals('ref', 'T0')), fmt(vals('ref', 'T4')),
                fmt(vals('ref', 'T9')), np.mean(vals('ref')),
                np.mean(vals('ref', key='last10')),
                np.mean(vals('ref', key='last1'))))
    for arm, note in ARMS:
        if not vals(arm):
            L.append('| `%s` | %s | %.2f M | – | – | – | – | – | – |'
                     % (arm, note, PARAMS.get(arm, float('nan')) / 1e6))
            continue
        if complete(arm):
            pooled_cols = ('**%.2f** | %.2f | %.2f'
                           % (np.mean(vals(arm)),
                              np.mean(vals(arm, key='last10')),
                              np.mean(vals(arm, key='last1'))))
        else:
            pooled_cols = '– *[不完整，池化不可比]* | – | –'
        L.append('| `%s` | %s | %.2f M | %s | %s | %s | %s |'
                 % (arm, note, PARAMS.get(arm, float('nan')) / 1e6,
                    fmt(vals(arm, 'T0')), fmt(vals(arm, 'T4')),
                    fmt(vals(arm, 'T9')), pooled_cols))
    L.append('')
    L.append('> 参数量由 `logs/_preflight_front_arms.py` 在交付配置下实测'
             '（`num_classes=14`、可训练参数、含位置编码与分类头）；'
             '`fr_cap960` 的 3.85 M 即"容量对齐 WIDAN（3.98 M）"档。')
    L.append('')

    # ---- (a2) 配对检验 ----
    L += ['## (a2) 配对检验（vs 本方法配置，同 task×seed，双侧 Wilcoxon）', '',
          'Δ > 0 表示该臂**优于**本方法配置。⚠ 本组**不是严格配对**（见文件头），'
          '故 p 值只作参考，判定以**池化 Δ 与可复现下限 %.2f 点**的比较为准。' % FLOOR,
          '', '| 臂 | n | Δ last5 池化 | p (T0) | p (T4) | p (T9) | p (池化) | '
          'wins (池化) | 四读数同号 |', '|---|---|---|---|---|---|---|---|---|']
    for arm, _ in ARMS:
        d_t, d_all = {}, []
        for t in TASKS:
            d_t[t] = []
            for s in SEEDS:
                k1, k2 = (arm, t, s), ('ref', t, s)
                if k1 in data and k2 in data:
                    d = data[k1]['last5'] - data[k2]['last5']
                    d_t[t].append(d)
                    d_all.append(d)
        if not d_all:
            L.append('| `%s` | 0 | – | – | – | – | – | – | – |' % arm)
            continue
        dirs = []
        for key in ('last5', 'last1', 'last10', 'peak'):
            ds = [data[(arm, t, s)][key] - data[('ref', t, s)][key]
                  for t in TASKS for s in SEEDS
                  if (arm, t, s) in data and ('ref', t, s) in data]
            dirs.append(('+' if np.mean(ds) > 0 else
                         ('−' if np.mean(ds) < 0 else '0')) if ds else '?')
        same = '✓' if len(set(dirs)) == 1 else '✗'
        L.append('| `%s` | %d | **%+.2f** | %s | %s | %s | %s | %d/%d | %s |'
                 % (arm, len(d_all), np.mean(d_all), pstr(pv(d_t['T0'])),
                    pstr(pv(d_t['T4'])), pstr(pv(d_t['T9'])), pstr(pv(d_all)),
                    sum(1 for x in d_all if x > 0), len(d_all), same))
    L.append('')

    # ---- (a3) 结论 ----
    pooled = {arm: float(np.mean(vals(arm))) for arm, _ in ARMS
              if vals(arm) and complete(arm)}
    L += ['## (a3) 结论（按可复现下限判定）', '']
    if pooled and vals('ref'):
        refp = float(np.mean(vals('ref')))
        best = max(pooled, key=pooled.get)
        L.append('- 本方法池化 last5 = **%.2f**；本组最高臂：**`%s`（%.2f）**%s'
                 % (refp, best, pooled[best],
                    '？⚠ 高于本方法，需复核是否超下限' if pooled[best] > refp
                    else '（未超过本方法的 %.2f）' % refp))
        for arm, _ in ARMS:
            if arm not in pooled:
                continue
            d = pooled[arm] - refp
            verdict = ('**落在可复现下限内 → 无可测差异**' if abs(d) < FLOOR
                       else '**超出下限 → 可判定方向**')
            L.append('- `%s` − 本方法 = **%+.2f** 点 → %s' % (arm, d, verdict))
        L.append('')
        L.append('> 判读模板（与 §6.3/§6.4 措辞一致）：')
        L.append('> · 初始化：若 `fr_rand`/`fr_mix` 未超下限，只能写'
                 '"在该混叠前端下，解析 Gabor 首层先验**未检出**精度优势"，'
                 '不得写成"初始化无关"；')
        L.append('> · 低频扩展：若 `fr_narrow` 未超下限，§4.1/§4.2.2 的'
                 '"低频扩展有效性"必须改写为**覆盖率论证**（把最慢转速轴频谐波'
                 '纳入频带），不得留作精度主张；')
        L.append('> · 容量：容量臂参数量已列在 (a1)，若未超下限则支持'
                 '"容量不是本方法的杠杆"，同时与`_s66_efficiency.md`的效率主张一致。')
    L.append('')

    # ---- (b) 同源性自检 ----
    L += ['## (b) 参照臂同源性自检（`%s` vs 主对比 `pub`）' % args.ref_arm, '',
          '- 单元数：**%d/%d**（T0/T4/T9 × 5 种子）'
          % (prov['cells'], len(TASKS) * len(SEEDS)),
          '- 逐点最大差：**max|Δ| = %.6f** → %s'
          % (prov['max'], '✅ 一致（参照臂 = 主对比臂）' if prov_ok
             else '❌ **不一致，本文件不可用**'),
          '- 依据：参照臂与主对比 `pub` 同为交付配置 + 480 步 + 余弦退火 + '
          '同种子 + 确定性初始化，故应逐点一致。']
    if prov['short']:
        L.append('- ⚠ 缺失/偏短：%s' % ', '.join(prov['short']))
    L.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('[saved] %s' % out_path)
    print('ref provenance: cells=%d max|d|=%.6f ok=%s'
          % (prov['cells'], prov['max'], prov_ok))
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
