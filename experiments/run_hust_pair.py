#!/usr/bin/python
# -*- coding:utf-8 -*-
"""HUST far-pair LODO with TRAINED models (source-restricted, volume-matched).

Fidelity: calls run_compare.run_one() -- the paper own single-cell entry -- after
monkeypatching run_cell.TASKS['HUST'][task].  Arm = 'unif' (the paper main arm).
NOTHING in the repo is modified; every output goes to results_hustpair/ so the probe
can never be swept into the paper _cmp_* summaries.

levels
  near<k> / far<k> : k source domains chosen by speed ratio (near = k smallest,
                     far = k largest).  Same k  =>  same n_src, same volume per
                     source per epoch, same steps (480).  The ONLY difference is
                     WHERE the sources sit on the speed axis.
  all9             : the paper leave-one-out (already in results/, not re-run here)

usage
  python experiments/run_hust_pair.py --run --targets 0,9 --levels near4,far4 \
         --methods AGMamba,WIDAN,CORAL,MMD --seeds 0,1,2
  python experiments/run_hust_pair.py --report
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault('AGM_FORCE_PY_SCAN', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.HUST_bearing import SPEEDS as HSP          # noqa: E402
import run_cell as RC                                     # noqa: E402
import run_compare as RCM                                 # noqa: E402

OUT = os.path.join(ROOT, 'results_hustpair')
EPOCHS = 60


def csv_file(methods, seeds):
    """per-process CSV: concurrent writers must not share one file"""
    return os.path.join(OUT, '_units_%s_s%s.csv'
                        % (methods.replace(',', '+'), seeds.replace(',', '')))
COLS = ['method', 'arm', 'level', 'target', 'task', 'seed', 'n_src', 'src_ids',
        'n_ckpt', 'last1', 'last5', 'last10', 'peak', 'seconds', 'log']


def ratio(i, t):
    return max(HSP[i], HSP[t]) / min(HSP[i], HSP[t])


def sources(level, t):
    order = sorted([i for i in range(10) if i != t], key=lambda i: ratio(i, t))
    if level == 'all9':
        return order
    if level.startswith('v'):
        # explicit speed list, e.g. 'v65+80' -> the two source speeds closest to 65 / 80 Hz
        out = []
        for x in level[1:].split('+'):
            v = float(x)
            i = min(range(10), key=lambda j: abs(HSP[j] - v))
            if i not in out:
                out.append(i)
        return sorted(out)
    if level.startswith('belowN') or level.startswith('aboveN'):
        k = int(level[6:])
        lo = level.startswith('belowN')
        side = [i for i in range(10) if i != t
                and ((HSP[i] < HSP[t]) if lo else (HSP[i] > HSP[t]))]
        side.sort(key=lambda i: ratio(i, t))
        return side[:k]           # k SMALLEST ratios on ONE side (NEAREST)
    if level.startswith('below') or level.startswith('above'):
        k = int(level[5:])
        lo = level.startswith('below')
        side = [i for i in range(10) if i != t
                and ((HSP[i] < HSP[t]) if lo else (HSP[i] > HSP[t]))]
        side.sort(key=lambda i: ratio(i, t))
        return side[-k:]          # k largest ratios on ONE side (FARTHEST)
    k = int(level.replace('near', '').replace('far', ''))
    return order[:k] if level.startswith('near') else order[-k:]


def logpath(method, arm, level, t, seed):
    d = os.path.join(OUT, method, level, 'T%d' % t)
    return os.path.join(d, 'HUST_T%d_%s_fc_%s_seed%d.log' % (t, method, level, seed))


def accs(path):
    if not os.path.exists(path) or os.path.getsize(path) < 200:
        return []
    try:
        return RCM.parse_accs(path) or []
    except Exception:
        return []


def row_from(method, arm, level, t, seed, src, seconds, lp):
    a = [v for _, v in accs(lp)]
    if not a:
        return None
    return dict(method=method, arm=arm, level=level, target=HSP[t], task='T%d' % t,
                seed=seed, n_src=len(src), src_ids='|'.join(str(i) for i in src),
                n_ckpt=len(a), last1=round(a[-1], 4),
                last5=round(sum(a[-5:]) / len(a[-5:]), 4),
                last10=round(sum(a[-10:]) / len(a[-10:]), 4),
                peak=round(max(a), 4), seconds=round(seconds, 1), log=lp)


def append_row(row, path):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def do_run(a):
    import torch
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT, exist_ok=True)
    cells = [(m, lv, int(t), int(s))
             for m in a.methods.split(',') for lv in a.levels.split(',')
             for t in a.targets.split(',') for s in a.seeds.split(',')]
    cf = csv_file(a.methods, a.seeds)
    print('### %d cells   arm=%s   device=%s   AGM_FORCE_PY_SCAN=%s'
          % (len(cells), a.arm, dev, os.environ.get('AGM_FORCE_PY_SCAN')), flush=True)
    for i, (m, lv, t, s) in enumerate(cells, 1):
        lp = logpath(m, a.arm, lv, t, s)
        got = accs(lp)
        if a.resume and len(got) >= EPOCHS:
            print('[skip %d/%d] %s %s T%d s%d (%d pts)'
                  % (i, len(cells), m, lv, t, s, len(got)), flush=True)
            continue
        src = sources(lv, t)
        RC.TASKS['HUST']['T%d' % t] = [list(src), [t]]       # <-- the only intervention
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        t0 = time.time()
        print('[%s] %d/%d RUN %s %s T%d(=%dHz) seed%d  n_src=%d src=%s'
              % (time.strftime('%H:%M:%S'), i, len(cells), m, lv, t, HSP[t], s,
                 len(src), src), flush=True)
        try:
            RCM.run_one(m, a.arm, 'HUST', 'T%d' % t, s, dev, lp)
        except Exception as e:
            print('    FAIL: %s' % e, flush=True)
        row = row_from(m, a.arm, lv, t, s, src, time.time() - t0, lp)
        if row:
            append_row(row, cf)
            print('    -> n=%d last5=%.2f (%.0fs)' % (row['n_ckpt'], row['last5'],
                                                      row['seconds']), flush=True)
        else:
            print('    -> NO LOG (%s)' % lp, flush=True)
    print('### END %s' % time.strftime('%H:%M:%S'), flush=True)


def ref_all9(method, arm, t, seeds):
    """the paper leave-one-out cells (already on disk)"""
    out = []
    for s in seeds:
        p = os.path.join(ROOT, 'results', 'HUST', method, arm, 'T%d' % t,
                         'HUST_T%d_%s_fc_%s_seed%d.log' % (t, method, arm, s))
        a = [v for _, v in accs(p)]
        if a:
            out.append(sum(a[-5:]) / len(a[-5:]))
    return out


def do_report(a):
    import glob
    paths = sorted(glob.glob(os.path.join(OUT, '_units*.csv')))
    if not paths:
        print('no _units*.csv yet in %s' % OUT)
        return
    rows = []
    for _p in paths:
        rows += list(csv.DictReader(open(_p, encoding='utf-8', newline='')))
    print('CSVs: %s' % [os.path.basename(x) for x in paths])
    seeds = sorted({int(r['seed']) for r in rows})
    print('=' * 96)
    print('HUST far-pair probe  (trained models)   cells=%d  seeds=%s  readout=last5'
          % (len(rows), seeds))
    print('=' * 96)
    print('%-10s %-6s %6s %8s %7s %8s %8s' %
          ('method', 'level', 'tgt', 'mean', 'sd', 'n', 'n_src'))
    keys = sorted({(r['method'], r['level'], int(r['target'])) for r in rows},
                  key=lambda k: (k[0], k[2], k[1]))
    for m, lv, tg in keys:
        v = [float(r['last5']) for r in rows
             if r['method'] == m and r['level'] == lv and int(r['target']) == tg]
        ns = {r['n_src'] for r in rows
              if r['method'] == m and r['level'] == lv and int(r['target']) == tg}
        sd = (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
        print('%-10s %-6s %6d %8.2f %7.2f %8d %8s'
              % (m, lv, tg, sum(v) / len(v), sd, len(v), ','.join(sorted(ns))))
    print()
    print('-' * 96)
    print('PAIRED  near4 - far4  (same method, target, seed)')
    idx = {(r['method'], r['level'], int(r['target']), int(r['seed'])): float(r['last5'])
           for r in rows}
    for m in sorted({r['method'] for r in rows}):
        for tg in sorted({int(r['target']) for r in rows}):
            d = [idx[(m, 'near4', tg, s)] - idx[(m, 'far4', tg, s)]
                 for s in seeds if (m, 'near4', tg, s) in idx and (m, 'far4', tg, s) in idx]
            if d:
                print('  %-10s T%-3d  delta=%+7.2f   n=%d   %s'
                      % (m, tg, sum(d) / len(d), len(d),
                         ' '.join('%+.1f' % x for x in d)))
    print()
    print('-' * 96)
    print('PAIRED  below2 - above2   (k=2, ratio-matched, ONLY the side differs)')
    for m in sorted({r['method'] for r in rows}):
        for tg in sorted({int(r['target']) for r in rows}):
            d = [idx[(m, 'below2', tg, s)] - idx[(m, 'above2', tg, s)]
                 for s in seeds
                 if (m, 'below2', tg, s) in idx and (m, 'above2', tg, s) in idx]
            if d:
                print('  %-10s T%-3d  delta=%+7.2f   n=%d   %s'
                      % (m, tg, sum(d) / len(d), len(d),
                         ' '.join('%+.1f' % x for x in d)))
    print()
    print('-' * 96)
    print('REFERENCE all9 (paper LODO, already on disk, `unif` arm)')
    for m in sorted({r['method'] for r in rows}):
        for tg in sorted({int(r['target']) for r in rows}):
            t = HSP.index(tg)
            v = ref_all9(m, a.arm, t, seeds)
            if v:
                print('  %-10s T%-3d  all9=%.2f  (n=%d, %s)'
                      % (m, tg, sum(v) / len(v), len(v),
                         ' '.join('%.1f' % x for x in v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--arm', default='unif')
    ap.add_argument('--methods', default='AGMamba,WIDAN,CORAL,MMD')
    ap.add_argument('--targets', default='0,9')
    ap.add_argument('--levels', default='near4,far4')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    if a.report:
        do_report(a)
    elif a.run:
        do_run(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
