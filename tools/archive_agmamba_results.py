#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归档并清空 AG-Mamba 家族的结果与权重（其余基线保持不动）。

用法
----
    python tools/archive_agmamba_results.py            # 预演，不改动任何文件
    python tools/archive_agmamba_results.py --apply    # 执行归档 + 清空

归属判定（见 docs/PROTOCOL.md §7）
------------------------------
AG-Mamba 家族 = `AGMamba`、旧方法名 `LM_DGNet`，以及历史上把"臂名"直接放在
method 层的扁平布局臂（`arch_*` / `mod_*` / `sweep_*` / `deep6_*` / `incept_*` /
`resnet_*` / `se_*` / `sep_*` / `tcn_*` / `base_lr*` / `impact_lr*` /
`tauprojv2` / `v4mod` / `ours` / `ctrl`）。

判定用两条**互不依赖**的证据，避免误删基线：
  1. 文件名含 `_AGMamba_` / `_LM_DGNet_`；
  2. 日志内容含 AG-Mamba **独有**字段 `tau0:` 或 `| delta:`
     （WIDAN 系列的日志是 `loss_total | ce | jmmd | lam`，两者不重叠）。

归档产物
--------
    results/_archive/<STAMP>/results.tar.gz + manifest.txt
    weights/_archive/<STAMP>/weights.tar.gz + manifest.txt

`_archive` 被 `agmamba/paths.py` 的 `find()` 显式跳过，不参与查找。
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

STAMP = '2026-09-28_preclean_agmamba'
RESULTS = _P.RESULTS
WEIGHTS = _P.WEIGHTS
ARCH = _P.ARCHIVE_NAME

HEAD = 512 * 1024
CONTENT_MARKS = (b'tau0:', b'| delta:', b"'front_mode'", b"'tau_estimator'")
NAME_MARKS = ('AGMamba', 'LM_DGNet')
NAME_PREFIX_AG = ('arch_',)     # 39B 的失败运行无内容可判，按臂名前缀认定


def _logs(d):
    out = []
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x != ARCH]
        for f in fn:
            if f.endswith('.log'):
                out.append(os.path.join(dp, f))
    return out


def is_ag_dir(d):
    """返回 (是否AG, 日志数, 字节数)。"""
    base = os.path.basename(os.path.normpath(d))
    force = base.startswith(NAME_PREFIX_AG)
    logs = _logs(d)
    if not logs:
        return force, 0, 0
    n_ag = 0
    for p in logs:
        nm = os.path.basename(p)
        if force or any(k in nm for k in NAME_MARKS):
            n_ag += 1
            continue
        try:
            with open(p, 'rb') as fh:
                if any(m in fh.read(HEAD) for m in CONTENT_MARKS):
                    n_ag += 1
        except OSError:
            pass
    size = sum(os.path.getsize(p) for p in logs)
    return (n_ag == len(logs)), len(logs), size


def scan_results():
    ag, keep = [], []
    for ds in sorted(os.listdir(RESULTS)):
        pds = os.path.join(RESULTS, ds)
        if not os.path.isdir(pds) or ds.startswith('_'):
            continue
        for e in sorted(os.listdir(pds)):
            pe = os.path.join(pds, e)
            if not os.path.isdir(pe):
                continue
            ok, n, sz = is_ag_dir(pe)
            (ag if ok else keep).append(('%s/%s' % (ds, e), n, sz))
    return ag, keep


def scan_weights():
    ag = []
    for ds in sorted(os.listdir(WEIGHTS)):
        pds = os.path.join(WEIGHTS, ds)
        if not os.path.isdir(pds) or ds.startswith('_'):
            continue
        for e in sorted(os.listdir(pds)):
            pe = os.path.join(pds, e)
            if not os.path.isdir(pe):
                continue
            if any(k in e for k in NAME_MARKS):
                nf = sum(len(fn) for _, _, fn in os.walk(pe))
                sz = sum(os.path.getsize(os.path.join(dp, f))
                         for dp, _, fn in os.walk(pe) for f in fn)
                ag.append(('%s/%s' % (ds, e), nf, sz))
    return ag


def human(n):
    for u in ('B', 'K', 'M', 'G'):
        if n < 1024:
            return '%.1f%s' % (n, u)
        n /= 1024.0
    return '%.1fT' % n


def make_archive(root, entries, outdir, tarname, label):
    """打包 entries（相对 root 的路径）为 tar.gz，返回 (tar路径, manifest路径)。"""
    os.makedirs(outdir, exist_ok=True)
    tarpath = os.path.join(outdir, tarname)
    manpath = os.path.join(outdir, 'manifest.txt')
    lines = ['# %s 归档清单' % label,
             '# 时间: %s' % time.strftime('%F %T'),
             '# 根目录: %s' % root,
             '# 条目数: %d' % len(entries), '']
    for name, nf, sz in entries:
        lines.append('%-44s files=%-6d size=%s' % (name, nf, human(sz)))
    with open(manpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    lst = os.path.join(outdir, '_filelist.txt')
    with open(lst, 'w') as f:
        f.write('\n'.join(name for name, _, _ in entries) + '\n')
    subprocess.run(['tar', '-czf', tarpath, '-C', root, '-T', lst], check=True)
    os.remove(lst)
    return tarpath, manpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正执行（默认只预演）')
    a = ap.parse_args()

    print('扫描 %s ...' % RESULTS)
    ag_r, keep_r = scan_results()
    ag_w = scan_weights()

    tot_r = sum(sz for _, _, sz in ag_r)
    tot_w = sum(sz for _, _, sz in ag_w)
    print('  AG-Mamba 结果目录: %d 个, 日志 %d 个, %s'
          % (len(ag_r), sum(n for _, n, _ in ag_r), human(tot_r)))
    print('  保留的基线目录  : %d 个, 日志 %d 个'
          % (len(keep_r), sum(n for _, n, _ in keep_r)))
    print('  AG-Mamba 权重目录: %d 个, 文件 %d 个, %s'
          % (len(ag_w), sum(n for _, n, _ in ag_w), human(tot_w)))
    print('\n--- 将归档并从活动目录移除（AG-Mamba，前 12 条）---')
    for name, n, sz in ag_r[:12]:
        print('  results/%-40s files=%-5d %s' % (name, n, human(sz)))
    if len(ag_r) > 12:
        print('  ... 其余 %d 条见 manifest' % (len(ag_r) - 12))
    print('\n--- 保留（示例，前 12 条）---')
    for name, n, sz in keep_r[:12]:
        print('  results/%-40s files=%-5d %s' % (name, n, human(sz)))

    if not a.apply:
        print('\n[预演] 未改动任何文件。加 --apply 执行。')
        return

    rdir = os.path.join(RESULTS, ARCH, STAMP)
    wdir = os.path.join(WEIGHTS, ARCH, STAMP)
    print('\n归档结果 -> %s' % rdir)
    rtar, _ = make_archive(RESULTS, ag_r, rdir, 'results.tar.gz', 'AG-Mamba results')
    print('归档权重 -> %s' % wdir)
    wtar, _ = make_archive(WEIGHTS, ag_w, wdir, 'weights.tar.gz', 'AG-Mamba weights')

    # 校验：tar 内成员数必须 >= 条目数（每个目录至少 1 个成员）
    for t, ents, label in ((rtar, ag_r, 'results'), (wtar, ag_w, 'weights')):
        n = len(subprocess.run(['tar', '-tzf', t], check=True,
                               capture_output=True, text=True).stdout.splitlines())
        assert n >= len(ents), '%s 归档异常: %d members < %d entries' % (
            label, n, len(ents))
        print('  校验 %-8s %s (%d members, %s)'
              % (label, os.path.basename(t), n, human(os.path.getsize(t))))

    print('\n删除活动目录 ...')
    for name, _, _ in ag_r:
        subprocess.run(['rm', '-rf', os.path.join(RESULTS, name)], check=True)
    for name, _, _ in ag_w:
        subprocess.run(['rm', '-rf', os.path.join(WEIGHTS, name)], check=True)
    # 清掉空的 <数据集> 层（weights 全部属于 AG-Mamba 时）
    for root in (WEIGHTS,):
        for ds in os.listdir(root):
            p = os.path.join(root, ds)
            if os.path.isdir(p) and ds != ARCH and not os.listdir(p):
                os.rmdir(p)
    print('完成。')
    print('  释放 %s（results） + %s（weights）'
          % (human(tot_r), human(tot_w)))


if __name__ == '__main__':
    main()
