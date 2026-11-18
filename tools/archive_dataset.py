#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按数据集归档并清空 results/ 与 weights/ 下的全部内容。

与其他归档脚本的关系
--------------------
- `tools/archive_agmamba_results.py`：只针对 AG-Mamba **家族**（跨数据集挑目录）。
- 本脚本：针对**某个数据集**在 results/weights 下的**全部**方法。

用法
----
    python tools/archive_dataset.py --dataset PHM2009            # 预演
    python tools/archive_dataset.py --dataset PHM2009 --apply    # 执行

产物
----
    results/_archive/<STAMP>_<dataset>/results.tar.gz + manifest.txt
    weights/_archive/<STAMP>_<dataset>/weights.tar.gz + manifest.txt

`_archive` 被 `agmamba/paths.py::find()` 显式跳过，不参与查找。
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

from agmamba import paths as _P  # noqa: E402

STAMP = '2026-09-28_preclean'


def human(n):
    for u in ('B', 'K', 'M', 'G'):
        if n < 1024:
            return '%.1f%s' % (n, u)
        n /= 1024.0
    return '%.1fT' % n


def scan(root, dataset):
    d = os.path.join(root, dataset)
    if not os.path.isdir(d):
        return []
    out = []
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if not os.path.isdir(p):
            continue
        nf = sz = 0
        for dp, _, fn in os.walk(p):
            for f in fn:
                nf += 1
                sz += os.path.getsize(os.path.join(dp, f))
        out.append((os.path.join(dataset, e), nf, sz))
    return out


def make_archive(root, dataset, entries, outdir, tarname, label):
    os.makedirs(outdir, exist_ok=True)
    tarpath = os.path.join(outdir, tarname)
    with open(os.path.join(outdir, 'manifest.txt'), 'w') as f:
        f.write('# %s 归档清单\n# 时间: %s\n# 根目录: %s\n# 数据集: %s\n'
                '# 条目数: %d\n\n'
                % (label, time.strftime('%F %T'), root, dataset, len(entries)))
        for name, nf, sz in entries:
            f.write('%-44s files=%-6d size=%s\n' % (name, nf, human(sz)))
    lst = os.path.join(outdir, '_filelist.txt')
    with open(lst, 'w') as f:
        f.write('\n'.join(n for n, _, _ in entries) + '\n')
    subprocess.run(['tar', '-czf', tarpath, '-C', root, '-T', lst], check=True)
    os.remove(lst)
    return tarpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    ds = a.dataset
    r_ent = scan(_P.RESULTS, ds)
    w_ent = scan(_P.WEIGHTS, ds)

    tot_r = sum(s for _, _, s in r_ent)
    tot_w = sum(s for _, _, s in w_ent)
    print('数据集 %s' % ds)
    print('  results/%s : %d 个目录, %d 文件, %s'
          % (ds, len(r_ent), sum(n for _, n, _ in r_ent), human(tot_r)))
    print('  weights/%s : %d 个目录, %d 文件, %s'
          % (ds, len(w_ent), sum(n for _, n, _ in w_ent), human(tot_w)))
    print('\n--- 将归档并从活动目录移除 ---')
    for name, nf, sz in r_ent:
        print('  results/%-40s files=%-5d %s' % (name, nf, human(sz)))
    for name, nf, sz in w_ent:
        print('  weights/%-40s files=%-5d %s' % (name, nf, human(sz)))

    if not a.apply:
        print('\n[预演] 未改动任何文件。加 --apply 执行。')
        return

    if not r_ent and not w_ent:
        print('无可归档内容。')
        return

    if r_ent:
        od = os.path.join(_P.RESULTS, _P.ARCHIVE_NAME, '%s_%s' % (STAMP, ds))
        t = make_archive(_P.RESULTS, ds, r_ent, od, 'results.tar.gz',
                         '%s %s results' % (ds, STAMP))
        n = len(subprocess.run(['tar', '-tzf', t], check=True,
                               capture_output=True, text=True).stdout.splitlines())
        assert n >= len(r_ent), 'results 归档异常'
        print('  归档 results -> %s (%d members, %s)'
              % (t, n, human(os.path.getsize(t))))
    if w_ent:
        od = os.path.join(_P.WEIGHTS, _P.ARCHIVE_NAME, '%s_%s' % (STAMP, ds))
        t = make_archive(_P.WEIGHTS, ds, w_ent, od, 'weights.tar.gz',
                         '%s %s weights' % (ds, STAMP))
        n = len(subprocess.run(['tar', '-tzf', t], check=True,
                               capture_output=True, text=True).stdout.splitlines())
        assert n >= len(w_ent), 'weights 归档异常'
        print('  归档 weights -> %s (%d members, %s)'
              % (t, n, human(os.path.getsize(t))))

    for root, ents in ((_P.RESULTS, r_ent), (_P.WEIGHTS, w_ent)):
        for name, _, _ in ents:
            subprocess.run(['rm', '-rf', os.path.join(root, name)], check=True)
        # 清掉空的 <数据集> 层
        d = os.path.join(root, ds)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    print('\n完成。释放 %s (results) + %s (weights)'
          % (human(tot_r), human(tot_w)))


if __name__ == '__main__':
    main()
