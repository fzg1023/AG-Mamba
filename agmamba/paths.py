# -*- coding: utf-8 -*-
"""结果与权重的目录规范（唯一入口）。

布局
----
    results/<DATASET>/<METHOD>/<ARM>/<TASK>/<原文件名>.log
    weights/<DATASET>/<METHOD>/<ARM>/<原文件名>.pt|.npz

* METHOD 算法名（AGMamba / WIDAN / UDTL / …）
* ARM    该算法的一个配置（变体臂）；基线只有一个配置时用 ``main``
* TASK   任务标识（T0…T9，或 L1/M3 等额外协议条件）

用法
----
写（训练脚本）::

    from agmamba.paths import log_path
    logfile = log_path('HUST', 'T0', 'AGMamba', seed, arm='xg2_std')

读（分析脚本）::

    from agmamba.paths import find
    p = find('HUST_T0_AGMamba_fc_xg2_std_seed0.log')   # 按文件名递归定位

``find`` 在 ``RESULTS``（及可选的 ``AGM_RESULTS_EXTRA``）下按**文件名**建立索引；
``_archive/`` 留档目录会被显式跳过。文件名在规范布局中唯一，因此查找无歧义。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)          # 仓库根
RESULTS = os.environ.get('AGM_RESULTS') or os.path.join(ROOT, 'results')
WEIGHTS = os.environ.get('AGM_WEIGHTS') or os.path.join(ROOT, 'weights')
# 可选：额外搜索根（冒号分隔），用于在别处挂载的历史结果目录；默认无
EXTRA_ROOTS = [p for p in os.environ.get('AGM_RESULTS_EXTRA', '').split(':') if p]
# 被淘汰的实验版本留档于此，不参与查找（见 docs/PROTOCOL.md §8.3）
ARCHIVE_NAME = '_archive'

_INDEX = None


def log_path(dataset, task, method, seed, arm='main',
             filename=None):
    """返回日志的规范路径（并创建目录）。"""
    if filename is None:
        if arm in (None, '', 'main'):
            filename = '{}_{}_{}_seed{}.log'.format(dataset, task, method,
                                                    seed)
        else:
            filename = '{}_{}_{}_fc_{}_seed{}.log'.format(dataset, task,
                                                          method, arm, seed)
    d = os.path.join(RESULTS, dataset, method, arm or 'main', str(task))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def weights_dir(dataset, method, arm='main'):
    d = os.path.join(WEIGHTS, dataset, method, arm or 'main')
    os.makedirs(d, exist_ok=True)
    return d


def _build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}
    for root in [RESULTS] + EXTRA_ROOTS:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [x for x in dn if x != ARCHIVE_NAME]
            for f in fn:
                idx.setdefault(f, os.path.join(dp, f))
    _INDEX = idx
    return idx


def find(filename, refresh=False):
    """按文件名定位日志；找不到返回原样拼接的路径（便于报错可见）。"""
    if refresh:
        global _INDEX
        _INDEX = None
    idx = _build_index()
    if filename in idx:
        return idx[filename]
    return os.path.join(RESULTS, filename)


def iter_logs(dataset=None, method=None, arm=None):
    """遍历规范布局下的日志路径。"""
    base = os.path.join(RESULTS, dataset) if dataset else RESULTS
    if method:
        base = os.path.join(base, method)
        if arm:
            base = os.path.join(base, arm)
    for dp, dn, fn in os.walk(base):
        dn[:] = [x for x in dn if x != ARCHIVE_NAME]
        for f in sorted(fn):
            if f.endswith('.log'):
                yield os.path.join(dp, f)


def describe():
    """打印布局概览（数据集 -> 算法 -> 臂 及日志数）。"""
    import collections
    cnt = collections.Counter()
    for dp, dn, fn in os.walk(RESULTS):
        logs = [f for f in fn if f.endswith('.log')]
        if not logs:
            continue
        rel = os.path.relpath(dp, RESULTS).split(os.sep)
        key = '/'.join(rel[:3])
        cnt[key] += len(logs)
    print('RESULTS = {}'.format(RESULTS))
    print('WEIGHTS = {}'.format(WEIGHTS))
    for k in sorted(cnt):
        print('   {:<44s} {:>5d}'.format(k, cnt[k]))
    print('   合计 {} 个日志'.format(sum(cnt.values())))


if __name__ == '__main__':
    describe()
