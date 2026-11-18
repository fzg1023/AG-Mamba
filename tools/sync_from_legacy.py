# -*- coding: utf-8 -*-
"""把旧仓库的结果日志增量同步进本仓库的规范布局。

用途
----
1) 首次迁移后补齐遗漏（tuned / lm / p2c / tsne 等目录）；
2) 旧仓库仍在跑批次时，可反复执行以同步新产生的日志（已存在则跳过）。

规范目标路径：
    results/<DATASET>/<METHOD>/<ARM>/<TASK>/<DS>_<TASK>_<METHOD>[_fc_<ARM>][_<EXTRA>]_seed<k>.log

用法
----
    python tools/sync_from_legacy.py [旧仓库路径]
默认旧仓库路径为 ~/FaultDia/DG-for-RMFD-main。
"""
import os
import re
import shutil
import sys

_OS = os
_HERE = _OS.path.dirname(_OS.path.abspath(__file__))
_ROOT = _OS.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from agmamba import paths as P  # noqa: E402

LEGACY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.expanduser('~'), 'FaultDia', 'DG-for-RMFD-main')

# 目录名 -> (数据集, 算法)。None 表示从文件名解析。
DIR_HINT = {
    'results_dgfdb': (None, None),
    'results_dgfdb_tuned': (None, None),
    'results_dgfdb_tuned2': (None, None),
    'results_lm': (None, 'AGMamba'),
    'results_pu': ('PU', None),
    'results_phm2009': ('PHM2009', None),
    'results_widan': (None, 'WIDAN'),
    'results_udtl': (None, 'UDTL'),
    'results_p2c_pu': ('PU', 'P2C'),
    'results_p2c_phm2009': ('PHM2009', 'P2C'),
    'results_tsne': (None, 'AGMamba'),
}
DATASETS = ['HUST', 'PU', 'PHM2009', 'UO']
TASK_RX = r'T\d|[LM]\d'
PAT_A = re.compile(r'^(%s)_(%s)_(.+)_seed(\d+)\.log$' % ('|'.join(DATASETS),
                                                         TASK_RX))
PAT_B = re.compile(r'^(%s)_(.+)_seed(\d+)\.log$' % TASK_RX)
PAT_C = re.compile(r'^(%s)_(%s)(?:_(.+))?_seed(\d+)\.log$'
                   % ('|'.join(DATASETS), TASK_RX))
SEED_RX = re.compile(r'_seed(\d+)\.log$')


def canonical(ds, task, method, arm, fname):
    """按目录信息重建规范文件名（与 _normalize_names2 同规则）。"""
    toks = SEED_RX.sub('', fname).split('_')
    if toks and toks[0] in DATASETS:
        toks = toks[1:]
    while toks and re.match(r'^[TLM]\d$', toks[0]):
        toks = toks[1:]

    def strip_head(ts, name):
        nt = name.split('_')
        while len(ts) >= len(nt) and ts[:len(nt)] == nt:
            ts = ts[len(nt):]
        return ts

    toks = strip_head(toks, method)
    if arm and arm != 'main':
        toks = strip_head(toks, arm)
    extra = '_'.join(t for t in toks if t)
    out = '{}_{}_{}'.format(ds, task, method)
    if arm and arm != 'main':
        out += '_fc_{}'.format(arm)
    if extra and extra not in (method, arm):
        out += '_' + extra
    return out + '_seed{}.log'.format(SEED_RX.search(fname).group(1))


def split_method_arm(rest):
    if '_fc_' in rest:
        m, a = rest.split('_fc_', 1)
        return m, (a or 'main')
    return rest, 'main'


added = skipped = unparsed = 0
per_dir = {}
for dname, (ds_hint, meth_hint) in DIR_HINT.items():
    src_dir = os.path.join(LEGACY, dname)
    if not os.path.isdir(src_dir):
        continue
    for f in sorted(os.listdir(src_dir)):
        if not f.endswith('.log'):
            continue
        ds = task = meth = arm = None
        m = PAT_A.match(f)
        if m:
            ds, task, rest, seed = m.groups()
            meth, arm = split_method_arm(rest)
        elif ds_hint is not None:
            m = PAT_B.match(f)
            if m:
                task, rest, seed = m.groups()
                meth, arm = split_method_arm(rest)
                ds = ds_hint
        elif meth_hint is not None:
            m = PAT_C.match(f)
            if m:
                ds, task, extra, seed = m.groups()
                meth, arm = meth_hint, (extra or 'main')
        if not (ds and task and meth):
            unparsed += 1
            continue
        name = canonical(ds, task, meth, arm, f)
        dst = os.path.join(P.RESULTS, ds, meth, arm or 'main', task, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(src_dir, f), dst)
        added += 1
        per_dir[dname] = per_dir.get(dname, 0) + 1

print('同步完成：新增 {}，已存在跳过 {}，无法解析 {}'.format(added, skipped,
                                                            unparsed))
for k in sorted(per_dir):
    print('   {:<26s} +{}'.format(k, per_dir[k]))
P._INDEX = None
print('\n当前布局：')
P.describe()
