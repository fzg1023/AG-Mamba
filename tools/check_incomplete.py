# -*- coding: utf-8 -*-
"""体检：找出**未跑满**的结果日志（进程被中断时留下的半截文件）。

背景
----
`run_compare.py` 的跳过条件是"解析出的评估点 >= 10"，目的是容忍历史格式差异。
但断电/被杀进程会留下**只有十几个评估点**的半截日志：它满足 >=10，
会被当成已完成而永久跳过，且 last10 会取自错误的检查点区间
（应有 60 个评估点，last10 = 第 51~60 个）。

本脚本按 `tgt_acc` / `tgt_train_acc` 计数（与 run_compare.parse_accs 同口径），
列出所有评估点数 != 期望值（默认 60）的日志。

用法
----
    python tools/check_incomplete.py --dataset HUST
    python tools/check_incomplete.py --dataset HUST --delete   # 删掉半截日志
"""
import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agmamba import paths as _P        # noqa: E402

STEP = re.compile(r'=+\s*Step\s+(\d+)\s*=+')
ACC = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
SEED = re.compile(r'_seed(\d+)\.log$')


def count_points(logfile):
    """返回该日志的评估点数（0 表示空/损坏）。"""
    if not os.path.exists(logfile):
        return 0
    n = 0
    with open(logfile, errors='ignore') as f:
        for line in f:
            if ACC.search(line):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--expect', type=int, default=60,
                    help='期望的评估点数（协议 = EPOCHS = 60）')
    ap.add_argument('--delete', action='store_true',
                    help='删除未跑满的日志（下次运行会自动重跑）')
    a = ap.parse_args()

    root = os.path.join(_P.RESULTS, a.dataset)
    if not os.path.isdir(root):
        print('目录不存在: %s' % root)
        return

    bad, ok, empty = [], 0, []
    for dp, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x != _P.ARCHIVE_NAME]
        for f in sorted(fn):
            if not f.endswith('.log'):
                continue
            p = os.path.join(dp, f)
            n = count_points(p)
            rel = os.path.relpath(p, root).replace('\\', '/')
            if n == 0:
                empty.append((rel, os.path.getsize(p)))
            elif n != a.expect:
                bad.append((rel, n))
            else:
                ok += 1

    print('===== %s 日志体检（期望评估点 %d）=====' % (a.dataset, a.expect))
    print('完整 : %d' % ok)
    print('半截 : %d' % len(bad))
    print('空/坏: %d' % len(empty))
    if bad:
        print('\n--- 半截（评估点不足，会被 run_compare 误判为已完成）---')
        for rel, n in bad:
            print('  %-3d 个点  %s' % (n, rel))
    if empty:
        print('\n--- 空/坏文件 ---')
        for rel, sz in empty:
            print('  %d bytes  %s' % (sz, rel))

    if a.delete:
        for rel, _ in bad + [(r, 0) for r, _ in empty]:
            os.remove(os.path.join(root, rel))
        print('\n已删除 %d 个文件（下次运行会自动重跑）'
              % (len(bad) + len(empty)))


if __name__ == '__main__':
    main()
