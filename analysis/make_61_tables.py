#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文用的 **HUST 表数据**（`docs/manuscript/_<ds>_tables.md`）。

⚠ 这是本脚本 2026-10-01 的重写；原版（v0.2 稿时代）已归档到
   `docs/legacy/make_61_tables_v0.2-era.py`。重写原因 = 原版的三个致命问题：

   1. **数据源是中间报表**：原版解析 `HUST_all_methods_report.md`
      （其注释写着"引用集数值来自 HUST_all_methods_report.md，日志已不在仓库"）——
      报表一改口径，本脚本就**静默**产出错数字。新版**只读 `results/`**。
   2. **口径是 best-epoch(peak)**：协议 `docs/COMPARE_PROTOCOL.md` §5.1 已把主口径
      改为 `last5`，并明确 peak 偏置随方法变化（0.85–6.89 分）、**跨方法不可比**。
      新版主列 = `last5`，并列 `last1`/`last10`，peak 单列留档。
   3. **引用了已消失的臂**：原版用 `AGMamba_fc_sc_steps6x`、`WIDAN_long`、`WIDAN_std`
      —— 现行（P-统一）只有 `unif`/`pub`，2880 步臂的日志已不在仓库
      （`results/_archive/` 内亦无）。**故 v0.2 的"预算对比（480 vs 2880）"不再生成**；
      P-统一已把所有方法预算固定为 480 步，该对比的语义本身也已被取代。
      预算曲线请走 `analysis/exp_e6_efficiency.py` 单独出表。

产出三张表
----------
* `<DS>-T1` 十任务主对比：方法/臂 × 任务（`last5` 主口径，5 种子 mean ± std）+ 族 + 平均
* `<DS>-T2` 关键对手：AG-Mamba 两臂 vs 主口径 Top-N 基线（含逐任务 Δ）
* `<DS>-T3` 三口径稳健性 + 配对检验（AG-Mamba `unif` / `pub` vs 全部基线）

统计
----
同时给 **Wilcoxon 符号秩**（协议口径，与 `analysis/summary_compare.py` 一致）
与 **配对 t 检验**（v0.2 稿表题里写的那个），避免静默改口径。

用法
----
    python analysis/make_61_tables.py                 # 默认 HUST
    python analysis/make_61_tables.py PU --top 5
输出
----
    docs/manuscript/_<ds 小写>_tables.md
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats as st

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from agmamba import paths as _P                       # noqa: E402
from make_all_methods_report import (                 # noqa: E402
    READOUTS, SEEDS, EPOCHS, read_series, SPEEDS)

OURS = 'AGMamba'
MAIN = 'last5'
READOUT_COLS = ('last5', 'last1', 'last10', 'peak')

# 与 experiments/run_cell.METHOD_MODELS 的顺序一致（顺序即批次执行顺序）
METHODS = ['CNNC', 'ADN', 'IEDGNet', 'DGNIS', 'WhiteningNet', 'CCN', 'CCDG',
           'CDDG', 'DDGFD', 'AGMamba', 'UDTL', 'WIDAN', 'MRA_SNN', 'DANN',
           'AGG', 'MMD', 'CORAL', 'Triplet_loss', 'Mixup', 'MLDG', 'DAEL']
# 只有这些方法需要跑第二个臂（见 run_compare.ARMS_MAY_DIFFER）
ARMS_MAY_DIFFER = {OURS}
# 机制族编号（对应论文 Table 2 的族号）
FAMILY = {
    'AGG': '1', 'MRA_SNN': '1',
    'DANN': '2', 'ADN': '2', 'UDTL': '2', 'DAEL': '2', 'IEDGNet': '2',
    'MMD': '3', 'CORAL': '3', 'DGNIS': '3', 'WIDAN': '3',
    'WhiteningNet': '4', 'CCN': '4', 'CCDG': '4', 'CDDG': '4', 'DDGFD': '4',
    'Triplet_loss': '5', 'CNNC': '5',
    'Mixup': '6', 'MLDG': '6',
}
# 模型参数量（模型属性，与训练协议无关；见论文 §5.4）
PARAMS = {OURS: '1.37M', 'WIDAN': '3.85M'}


def arms_of(method):
    return ['unif'] + (['pub'] if method in ARMS_MAY_DIFFER else [])


def label(method, arm):
    return method if method not in ARMS_MAY_DIFFER else '%s(%s)' % (method, arm)


def collect(ds, tasks):
    """{(method, arm): {(task, seed): {readout: value}}}；只收满 EPOCHS 点的单元。"""
    out = {}
    for method in METHODS:
        for arm in arms_of(method):
            d = {}
            for t in tasks:
                for s in SEEDS:
                    v = read_series(_P.log_path(ds, t, method, s, arm=arm))
                    if len(v) < EPOCHS:
                        continue
                    d[(t, s)] = {ro: fn(v) for ro, fn in READOUTS.items()}
            if d:
                out[(method, arm)] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='?', default='HUST')
    ap.add_argument('--top', type=int, default=4,
                    help='关键对手表里列出的基线个数（按主口径排序）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ds = a.dataset
    tasks = ['T%d' % i for i in range(10)] if ds == 'HUST' \
        else ['T%d' % i for i in range(4)]
    spd = SPEEDS.get(ds, {})

    data = collect(ds, tasks)
    if (OURS, 'unif') not in data:
        raise SystemExit('缺少 %s 的 unif 臂结果' % OURS)
    missing = [m for m in METHODS
               if all((m, arm) not in data for arm in arms_of(m))]
    if missing:
        print('[warn] 无满 %d 点单元、已跳过的方法: %s'
              % (EPOCHS, ', '.join(missing)))

    base_keys = [k for k in data if k[0] != OURS]
    ag_keys = [(OURS, arm) for arm in arms_of(OURS) if (OURS, arm) in data]

    def mean_of(key, ro):
        return float(np.mean([v[ro] for v in data[key].values()]))

    def paired(key_a, key_b, ro):
        common = sorted(set(data[key_a]) & set(data[key_b]))
        x = np.array([data[key_a][c][ro] for c in common])
        y = np.array([data[key_b][c][ro] for c in common])
        return x, y, common

    def tests(key_a, key_b, ro):
        """(p_wilcoxon, p_ttest, wins, n)。样本过少/全等 -> nan。"""
        x, y, common = paired(key_a, key_b, ro)
        n = len(common)
        if n < 5 or np.allclose(x, y):
            return float('nan'), float('nan'), int((x > y).sum()), n
        return (float(st.wilcoxon(x, y).pvalue),
                float(st.ttest_rel(x, y).pvalue), int((x > y).sum()), n)

    def fmt_p(p):
        return 'n/a' if p != p else '%.4g' % p

    L = ['# %s 论文表数据（**P-统一协议**，主口径 `%s`）' % (ds, MAIN), '',
         '由 `analysis/make_61_tables.py %s` 生成；'
         '**数据源 = `results/` 直读**（不再解析任何中间报表）。' % ds, '',
         '## 口径与"与原版的差异"（务必先读）', '',
         '- 主口径 **`last5`**（每单元末 5 个评估点均值，§5.1）；并列 `last1`（部署）'
         '与 `last10`（稳健）；`peak` 单列留档（**跨方法不可比**，不进正文）',
         '- 只统计**满 %d 个评估点**的单元（半截日志=中断残留，一律剔除）' % EPOCHS,
         '- 单元 = 任务 × 种子，满配 `n = %d`' % (5 * len(tasks)),
         '- 配对检验同时给 **Wilcoxon 符号秩**（协议口径）与 **配对 t 检验**'
         '（v0.2 稿表题口径）；两者不一致时以 Wilcoxon 为准',
         '- ⚠ **v0.2 的"决定性对比（480 vs 2880 步预算）"已不再生成**：'
         '`AGMamba_fc_sc_steps6x` / `WIDAN_long` 臂的日志已不在仓库'
         '（`results/_archive/` 内亦无），且 P-统一把所有方法预算固定为 480 步，'
         '该对比的语义已被取代。预算曲线请走 `analysis/exp_e6_efficiency.py`', '']

    # ---------------- T1 ----------------
    order = sorted(data, key=lambda k: -mean_of(k, MAIN))
    L += ['## %s-T1　主对比（%d 方法/臂 × %d 任务，`%s` 主口径，5 种子 mean ± std）'
          % (ds, len(order), len(tasks), MAIN), '',
          '| 方法 | 族 | ' + ' | '.join(
              '%s<br>%sHz' % (t, spd[t]) if t in spd else t for t in tasks)
          + ' | 平均 | n |',
          '|' + '---|' * (len(tasks) + 4)]
    for key in order:
        cells = []
        for t in tasks:
            vs = [v[MAIN] for (tt, _), v in data[key].items() if tt == t]
            cells.append('%.2f ± %.2f'
                         % (np.mean(vs), np.std(vs, ddof=1)) if vs else '—')
        L.append('| %s | %s | %s | **%.2f** | %d |'
                 % (label(*key), FAMILY.get(key[0], '–'), ' | '.join(cells),
                    mean_of(key, MAIN), len(data[key])))
    L.append('')

    # ---------------- T2 ----------------
    top = sorted(base_keys, key=lambda k: -mean_of(k, MAIN))[:a.top]
    L += ['## %s-T2　关键对手（主口径 Top-%d 基线）' % (ds, a.top), '',
          '| 方法 | 参数 | **%s（主）** | last1 | last10 | peak（留档） '
          '| Δ_%s vs AGMamba(unif) | p(Wilcoxon) | wins |' % (MAIN, MAIN),
          '|' + '---|' * 9]
    for key in ag_keys + top:
        is_ours = key[0] == OURS
        if is_ours:
            d_txt, p_txt, w_txt = '—', '—', '—'
        else:
            d_txt = '%+.2f' % (mean_of((OURS, 'unif'), MAIN) - mean_of(key, MAIN))
            pw, _, wins, n = tests((OURS, 'unif'), key, MAIN)
            p_txt, w_txt = fmt_p(pw), '%d/%d' % (wins, n)
        L.append('| %s | %s | **%.2f** | %.2f | %.2f | %.2f | %s | %s | %s |'
                 % (label(*key), PARAMS.get(key[0], '–'), mean_of(key, MAIN),
                    mean_of(key, 'last1'), mean_of(key, 'last10'),
                    mean_of(key, 'peak'), d_txt, p_txt, w_txt))
    L += ['', '### 逐任务 Δ（AGMamba(unif) − 对手，`%s` 主口径）' % MAIN, '',
          '| 对手 | ' + ' | '.join(tasks) + ' | 全部 |',
          '|' + '---|' * (len(tasks) + 2)]
    for key in top:
        cells = []
        for t in tasks:
            diffs = [data[(OURS, 'unif')][(t, s)][MAIN] - data[key][(t, s)][MAIN]
                     for s in SEEDS
                     if (t, s) in data[(OURS, 'unif')] and (t, s) in data[key]]
            cells.append('%+.2f' % np.mean(diffs) if diffs else '—')
        L.append('| %s | %s | **%+.2f** |'
                 % (label(*key), ' | '.join(cells),
                    mean_of((OURS, 'unif'), MAIN) - mean_of(key, MAIN)))
    L.append('')

    # ---------------- T3 ----------------
    for arm in arms_of(OURS):
        key_a = (OURS, arm)
        if key_a not in data:
            continue
        L += ['## %s-T3%s　三口径稳健性 + 配对检验（AGMamba `%s` vs 各基线）'
              % (ds, '' if arm == 'unif' else 'b', arm), '',
              'Δ = 同 (任务, 种子) 配对的均值差；`p_W` = Wilcoxon，`p_t` = 配对 t。',
              '',
              '`同号` 的定义与 `analysis/summary_compare.py` **保持一致**：'
              '`last5`/`last1`/`last10`/`peak` 四个 Δ **全部同向**才算"是"'
              '（peak 不进正文，但它反号提示"读数对口径敏感"，必须如实标出）。',
              '',
              '| 基线 | Δ %s | Δ last1 | Δ last10 | Δ peak | 同号(4口径) | '
              'p_W, p_t, wins | 判定 |' % MAIN, '|' + '---|' * 8]
        keys_sorted = sorted(base_keys,
                             key=lambda k: -(mean_of(key_a, MAIN) - mean_of(k, MAIN)))
        for key in keys_sorted:
            deltas = {ro: mean_of(key_a, ro) - mean_of(key, ro)
                      for ro in READOUT_COLS}
            same = all(deltas[ro] * deltas[MAIN] > 0 for ro in READOUT_COLS)
            pw, pt, wins, n = tests(key_a, key, MAIN)
            verdict = '胜 (p<0.05)' if pw == pw and pw < 0.05 else (
                '不显著' if pw == pw else 'n/a')
            L.append('| %s | **%+.2f** | %+.2f | %+.2f | %+.2f | %s | '
                     '%s, %s, %d/%d | %s |'
                     % (label(*key), deltas[MAIN], deltas['last1'],
                        deltas['last10'], deltas['peak'],
                        '是' if same else '**否**', fmt_p(pw), fmt_p(pt),
                        wins, n, verdict))
        L.append('')

    out = a.out or os.path.join(ROOT, 'docs', 'manuscript',
                                '_%s_tables.md' % ds.lower())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('[tables] %s' % out)
    print('  方法/臂 %d 个；主口径 %s；产出 T1/T2/T3' % (len(order), MAIN))
    for key in order[:6]:
        print('   top  %-18s %s=%.2f' % (label(*key), MAIN, mean_of(key, MAIN)))


if __name__ == '__main__':
    main()
