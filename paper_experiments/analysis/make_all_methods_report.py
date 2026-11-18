#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 **统一协议（P-统一）** 汇总全方法基准表。

**职责定位（2026-10-01 起）**
----------------------------
本模块同时扮演两个角色：

1. **工具库（主要）**：提供全库共用的读数原语 —— `read_series`（日志解析）、
   `READOUTS`（last5/last1/last10/peak）、`SEEDS`、`EPOCHS`、`SPEEDS`。
   `analysis/make_61_tables.py` 直接 import 这些，保证**跨脚本口径逐位一致**。
2. **临时速览 CLI（次要）**：默认**只打印到 stdout**，需要时用 `--out` 显式落盘。

> 为什么要这样限制落盘：本模块早期版本默认写仓库根的
> `<DS>_all_methods_report.md`，与 `docs/manuscript/_<ds>_tables.md` 内容
> 高度重叠 —— 两个事实源一旦口径漂移就会静默出错数字（已有前车之鉴：
> 旧版 `make_61_tables.py` 读的就是那个中间报表）。
> **规范性表格一律以 `docs/manuscript/` 下由 `make_61_tables.py` 生成的产物为准。**

与历史 `analysis/summary_*_all.py` 的区别（重要）
------------------------------------------------
那些脚本是 **best-epoch(peak) 口径**，且读的是归档前的旧日志目录。协议
`docs/COMPARE_PROTOCOL.md` §5.1 已明确：**peak 偏置随方法变化（0.85–6.89 分），
跨方法不可比**，只能留档、不进论文。故本模块按现行协议输出：

  · 主口径 `last5`（末 5 个评估点均值）—— 进论文
  · `last1`（部署口径）、`last10`（稳健性口径）—— 进论文
  · `peak` 单独一列标注「留档·不可跨方法比」

数据源：`results/<DS>/<METHOD>/<ARM>/<TASK>/*.log`（与 run_compare 同一解析规则），
只统计**满 EPOCHS(60) 个评估点**的单元（半截日志不计，防中断残留污染）。

用法
----
    python analysis/make_all_methods_report.py HUST            # 打印到 stdout
    python analysis/make_all_methods_report.py PU --out /tmp/pu.md   # 显式落盘
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

from agmamba import paths as _P                      # noqa: E402

PAT_ACC = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
# 显示名：只有 AGMamba 有两个差异臂，需要区分
ARMS_MAY_DIFFER = {'AGMamba'}
# HUST 任务名 -> 目标域转速（表头用）；非 HUST 数据集留空
SPEEDS = {
    'HUST': {'T%d' % i: s for i, s in enumerate(
        [20, 25, 30, 35, 40, 60, 65, 70, 75, 80])},
    'PU': {},
    'PHM2009': {},
}


def read_series(path):
    """读一个单元的全部评估点（%）。文件不存在/过短返回 []。"""
    if not os.path.exists(path):
        return []
    with open(path, errors='ignore') as fh:
        txt = fh.read()
    v = [float(x) for x in PAT_ACC.findall(txt)]
    return [x * 100 if x <= 1.5 else x for x in v]


READOUTS = {
    'last5': lambda v: float(np.mean(v[-5:])),
    'last1': lambda v: float(v[-1]),
    'last10': lambda v: float(np.mean(v[-10:])),
    'peak': lambda v: float(np.max(v)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--readout', default='last5', choices=list(READOUTS),
                    help='主口径列（默认 last5，§5.1）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    ds = a.dataset
    root = os.path.join(_P.RESULTS, ds)
    if not os.path.isdir(root):
        raise SystemExit('结果目录不存在: %s' % root)

    # 任务列表：从各方法的目录里取（形如 T0…T9）
    tasks = set()
    units = defaultdict(dict)          # (method, arm) -> (task, seed) -> path
    for method in sorted(os.listdir(root)):
        pm = os.path.join(root, method)
        if not os.path.isdir(pm):
            continue
        for arm in sorted(os.listdir(pm)):
            pa = os.path.join(pm, arm)
            if not os.path.isdir(pa):
                continue
            if arm == 'pub' and method not in ARMS_MAY_DIFFER:
                continue                       # 与 unif 等价的臂不重复计入
            for task in sorted(os.listdir(pa)):
                pt = os.path.join(pa, task)
                if not os.path.isdir(pt) or not re.match(r'^T\d+$', task):
                    continue
                tasks.add(task)
                for seed in SEEDS:
                    f = os.path.join(
                        pt, '%s_%s_%s_fc_%s_seed%d.log' % (ds, task, method,
                                                           arm, seed))
                    if os.path.exists(f):
                        units[(method, arm)][(task, seed)] = f
    tasks = sorted(tasks, key=lambda t: int(t[1:]))

    main_ro = a.readout
    per = {}                            # (method, arm) -> {task: (mu, sd, n)}
    agg = {}                            # (method, arm) -> {ro: mean across units}
    for key, files in units.items():
        by_task = defaultdict(list)
        for (task, seed), f in files.items():
            v = read_series(f)
            if len(v) < EPOCHS:          # 完整性闸门
                continue
            by_task[task].append({ro: fn(v) for ro, fn in READOUTS.items()})
        if not by_task:
            continue
        per[key] = {}
        ro_pool = defaultdict(list)
        for task, lst in by_task.items():
            vals = [d[main_ro] for d in lst]
            per[key][task] = (float(np.mean(vals)),
                              float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              len(vals))
            for ro in READOUTS:
                ro_pool[ro].extend(d[ro] for d in lst)
        agg[key] = {ro: float(np.mean(ro_pool[ro])) for ro in READOUTS}
        agg[key]['n_units'] = sum(len(l) for l in by_task.values())

    if not per:
        raise SystemExit('没有任何满 %d 个评估点的单元' % EPOCHS)

    def label(key):
        m, arm = key
        return m if arm == 'unif' and m not in ARMS_MAY_DIFFER \
            else '%s(%s)' % (m, arm)

    keys = sorted(per, key=lambda k: -np.mean([v[0] for v in per[k].values()]))
    spd = SPEEDS.get(ds, {})
    L = ['# %s 全方法基准（**P-统一协议**，%d 任务 LODO × 5 种子）'
         % (ds, len(tasks)), '',
         '临时速览输出（stdout）。**规范性表格请看** `docs/manuscript/_%s_tables.md`'
         '（由 `analysis/make_61_tables.py` 生成）；两者共用本模块的读数原语，'
         '口径逐位一致。' % ds.lower(), '',
         '## 口径（与旧报表的关键差异）', '',
         '- **主口径 = `%s`**（每个单元末 k 个评估点均值，k=5 的选择依据见'
         ' `docs/COMPARE_PROTOCOL.md` §5.1）' % main_ro,
         '- 并列给出 `last1`（部署口径，最保守）与 `last10`（稳健性口径）；'
         '三者**同向**才可下结论',
         '- `peak`（best-epoch）**不用于跨方法比较**（偏置随方法变化 0.85–6.89 分），'
         '仅在本表留档',
         '- 单元须**满 %d 个评估点**才计入（半截日志=中断残留，一律剔除）' % EPOCHS,
         '- 预算：60 epoch × iters_per_epoch，batch 64，每 epoch 评估一次',
         '- 数值 = 5 种子均值 ± 标准差（**跨种子**）；`n` = 计入的单元数，'
         '满配为 %d（任务数 × 5）' % (5 * len(tasks)), '',
         '## 主口径 `%s` —— 逐任务（mean ± std / 5 种子）' % main_ro, '',
         '| 方法 | ' + ' | '.join(
             '%s<br>%sHz' % (t, spd[t]) if t in spd else t for t in tasks)
         + ' | 平均 | n |',
         '|' + '---|' * (len(tasks) + 3)]
    for k in keys:
        cells = []
        for t in tasks:
            if t in per[k]:
                mu, sd, _ = per[k][t]
                cells.append('%.2f ± %.2f' % (mu, sd))
            else:
                cells.append('—')
        avg = np.mean([per[k][t][0] for t in tasks if t in per[k]])
        L.append('| %s | %s | **%.2f** | %d |'
                 % (label(k), ' | '.join(cells), avg, agg[k]['n_units']))

    L += ['', '## 三口径 + peak 对照（跨任务平均）', '',
          '| 方法 | **%s（主）** | last1（部署） | last10（稳健） '
          '| peak（留档·不可比） | n |' % main_ro,
          '|---|---|---|---|---|---|']
    for k in keys:
        g = agg[k]
        L.append('| %s | **%.2f** | %.2f | %.2f | %.2f | %d |'
                 % (label(k), g[main_ro], g['last1'], g['last10'],
                    g['peak'], g['n_units']))

    out = a.out
    if out:
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L) + '\n')
        print('[report] %s  (%d 个 (方法,臂), %d 个任务)'
              % (out, len(per), len(tasks)))
    else:
        # 默认只打印：规范性表格请用 analysis/make_61_tables.py（写 docs/manuscript/）
        print('\n'.join(L))
    for k in keys:
        print('  %-16s %s=%.2f  n=%d'
              % (label(k), main_ro, agg[k][main_ro], agg[k]['n_units']))


if __name__ == '__main__':
    main()
