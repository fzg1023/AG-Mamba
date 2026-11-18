# -*- coding: utf-8 -*-
"""计算“两种报告口径”下的主表数值（best-epoch=oracle 与 last-ten=固定预算末端）。

日志可能分布在 results_dgfdb / results_pu / results_phm2009 / results_widan / results_udtl，
其中原始八任务协议下 WIDAN/UDTL 的文件名不含方法名。
输出：docs/manuscript/_two_rule_tables.md
"""
import os
import re

import numpy as np
# --- 结果路径：统一走 agmamba.paths ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
RD = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'docs/manuscript', '_two_rule_tables.md')
PAT = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEEDS = [0, 1, 2, 3, 4]
DIRS = [RD, os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results'),
        os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results')]
L = []


def find(data, task, arm, seed):
    """候选路径（顺序即优先级）：
      1. {data}_{task}_{arm}_seedK.log  —— results_dgfdb 下的所有臂；
      2. {task}_{arm}_seedK.log        —— results_pu / results_phm2009（无数据集前缀）；
      3. {data}_{task}_seedK.log       —— results_widan / results_udtl（文件名不含方法名）。
    """
    return _findlog('{}_{}_{}_seed{}.log'.format(data, task, arm, seed))


def parse(p):
    if p is None or not os.path.exists(p):
        return None
    with open(p, errors='ignore') as f:
        v = [float(x) * 100 for x in PAT.findall(f.read())]
    if not v:
        return None
    return max(v), (float(np.mean(v[-10:])) if len(v) >= 10 else float(np.mean(v)))


def table(title, data, tasks, methods):
    L.append('\n## ' + title)
    L.append('| Method | ' + ' | '.join('{} best / last'.format(t) for t in tasks) +
             ' | Best (pooled) | Last-ten (pooled) | Best − last |')
    L.append('|' + '---|' * (len(tasks) + 4))
    out = []
    for label, arm in methods:
        pb, pl, allb, alll, miss = [], [], [], [], 0
        for t in tasks:
            bs, ls = [], []
            for s in SEEDS:
                r = parse(find(data, t, arm, s))
                if r is None:
                    miss += 1
                    continue
                bs.append(r[0])
                ls.append(r[1])
            if bs:
                pb.append(np.mean(bs))
                pl.append(np.mean(ls))
                allb.extend(bs)
                alll.extend(ls)
        if not allb:
            L.append('| {} | {} |'.format(label, ' | '.join(['(no logs)'] * len(tasks))))
            print('  {:26s} (no logs)'.format(label))
            continue
        cells = ['{:.2f} / {:.2f}'.format(b, l) for b, l in zip(pb, pl)]
        L.append('| {} | {} | **{:.2f} ± {:.2f}** | **{:.2f} ± {:.2f}** | {:+.2f} |'.format(
            label, ' | '.join(cells), np.mean(allb), np.std(allb, ddof=1),
            np.mean(alll), np.std(alll, ddof=1), np.mean(allb) - np.mean(alll)))
        out.append((label, np.mean(allb), np.mean(alll), miss))
    for label, b, l, m in out:
        print('  {:26s} best {:.2f}  last-ten {:.2f}  drop {:+.2f}{}'.format(
            label, b, l, b - l, '' if m == 0 else '  [{} missing]'.format(m)))


PU_M = ['UDTL', 'Mixup', 'DANN', 'WIDAN', 'CORAL', 'MMD', 'AGG', 'Triplet_loss', 'MLDG', 'CCN',
        'CDDG', 'DDGFD', 'CCDG', 'DAEL', 'DGNIS', 'ADN', 'IEDGNet', 'CNNC', 'WhiteningNet']
PHM_M = ['WIDAN', 'DANN', 'AGG', 'Mixup', 'MMD', 'CORAL', 'MLDG', 'Triplet_loss', 'CCN', 'DAEL',
         'ADN', 'DGNIS', 'CDDG', 'CCDG', 'DDGFD', 'UDTL', 'IEDGNet', 'CNNC', 'WhiteningNet']

print('=== PU ===')
table('PU bearing, four tasks, best-epoch (oracle) vs last-ten (deployment rule)',
      'PU', ['T0', 'T1', 'T2', 'T3'],
      [('**AG-Mamba (xg_std)**', 'AGMamba_fc_xg_std'),
       ('AG-Mamba (xg_6x)', 'AGMamba_fc_xg_6x'),
       ('AG-Mamba base', 'AGMamba_fc_base')] + [(m, m) for m in PU_M])

print('\n=== PHM2009 ===')
table('PHM 2009 gearbox, four tasks, best-epoch (oracle) vs last-ten (deployment rule)',
      'PHM2009', ['T0', 'T1', 'T2', 'T3'],
      [('**AG-Mamba (xg_std)**', 'AGMamba_fc_xg_std'),
       ('AG-Mamba (xg_6x)', 'AGMamba_fc_xg_6x'),
       ('AG-Mamba base', 'AGMamba_fc_base')] + [(m, m) for m in PHM_M])

print('\n=== HUST ===')
table('HUST ten tasks, best-epoch (oracle) vs last-ten (deployment rule)',
      'HUST', ['T{}'.format(i) for i in range(10)],
      [('**AG-Mamba (delivered)**', 'AGMamba_fc_sc_steps6x'),
       ('WIDAN (480)', 'WIDAN'), ('WIDAN (2,880, matched)', 'WIDAN_long'),
       ('DANN (480)', 'DANN'), ('UDTL (480)', 'UDTL'), ('MRA-SNN (480)', 'MRA_SNN')])

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n[OK] ->', OUT)
