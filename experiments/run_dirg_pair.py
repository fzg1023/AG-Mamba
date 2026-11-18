#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIRG limited-source probe (multi-target): does adding a FAR source help?

Auxiliary experiment. Outputs go only to results_dirg_pair/ and logs/dirg_pair*.

Levels are generated automatically per target from the ratio max/min:
    near1 = the closest single source domain
    near2 = the two closest
    far2  = the two farthest
    far1  = the farthest single
    all4  = all four (anchor)

`run_dirg.run_one` computes iters from len(srcs[0]) only, so steps is ALWAYS
360 and 50 samples/source-domain/epoch regardless of n_src -> arms with the
same k are directly comparable (no steps_override needed).
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault('AGM_FORCE_PY_SCAN', '1')

import experiments.run_dirg as RD
from datasets.DIRG_bearing import DOMAINS, SPEEDS

PAIRROOT = os.path.join(_ROOT, 'results_dirg_pair')
LOAD = 1000
CH = 0
LEVEL_ORDER = ['all4', 'near2', 'far2', 'near1', 'far1']
COLS = ['method', 'arm', 'level', 'target', 'task', 'seed', 'n_src', 'src_ids',
        'n_ckpt', 'last1', 'last5', 'last10', 'peak', 'seconds', 'log']


def ratio(i, t):
    return max(SPEEDS[i], SPEEDS[t]) / min(SPEEDS[i], SPEEDS[t])


def levels_for(t):
    pool = [i for i in range(len(SPEEDS)) if i != t]
    order = sorted(pool, key=lambda i: ratio(i, t))
    return {'all4': order, 'near2': order[:2], 'far2': order[-2:],
            'near1': order[:1], 'far1': order[-1:]}


def sources_of(level, t):
    """level -> sorted source-domain ids.  's<idx>' = one source,
    'v<i>+<j>' = explicit set, otherwise the family from levels_for()."""
    if level.startswith('v'):
        return sorted(int(x) for x in level[1:].split('+'))
    if level.startswith('s') and level[1:].isdigit():
        return [int(level[1:])]
    return list(levels_for(t)[level])


def cell_csv(methods, seeds, targets, levels):
    return os.path.join(PAIRROOT, '_units_%s_s%s_T%s_L%s.csv'
                        % (methods.replace(',', '+'), seeds.replace(',', ''),
                           targets.replace(',', '+'), levels.replace(',', '+')))


def append_row(row, path):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def do_run(a):
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    targets = [int(x) for x in a.targets.split(',') if x.strip()]
    levels = [l.strip() for l in a.levels.split(',') if l.strip()]
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(PAIRROOT, exist_ok=True)
    cf = cell_csv(a.methods, a.seeds, a.targets, a.levels)
    done = set()
    if a.resume and os.path.exists(cf):
        with open(cf, newline='') as fh:
            for r in csv.DictReader(fh):
                if int(r.get('n_ckpt') or 0) >= 60:
                    done.add((r['method'], r['level'], r['target'], r['seed']))
    cells = [(m, t, lv, s) for m in methods for t in targets
             for lv in levels for s in seeds]
    print('### %d cells (done=%d) targets=%s levels=%s seeds=%s dev=%s'
          % (len(cells), len(done), targets, levels, seeds, dev), flush=True)
    for i, (m, t, lv, s) in enumerate(cells, 1):
        if (m, lv, str(t), s) in done:
            print('[skip] %s %s T%d seed%d' % (m, lv, t, s), flush=True)
            continue
        src = [x for x in sources_of(lv, t) if x != t]
        if not src:
            print('[skip] %s %s T%d seed%d : empty source set' % (m, lv, t, s), flush=True)
            continue
        task = [list(src), [t]]
        ld = os.path.join(PAIRROOT, m, 'T%d' % t, lv)
        os.makedirs(ld, exist_ok=True)
        lf = os.path.join(ld, 'DIRG_T%d_%s_%s_seed%d.log' % (t, m, lv, s))
        print('[%s] %d/%d RUN %s %s T%d(=%dHz) seed%d n_src=%d src=%s ratios=%s'
              % (time.strftime('%H:%M:%S'), i, len(cells), m, lv, t, SPEEDS[t], s,
                 len(src), [DOMAINS[x] for x in src],
                 [round(ratio(x, t), 3) for x in src]), flush=True)
        t0 = time.time()
        try:
            acc = RD.run_one(m, 'T%d_%s' % (t, lv), task, LOAD, s, dev, lf, None, CH)
            st = RD.summarize(acc)
        except Exception as e:
            print('  !! FAILED %s %s T%d seed%d : %r' % (m, lv, t, s, e), flush=True)
            continue
        dt = time.time() - t0
        row = dict(method=m, arm='unif', level=lv, target=SPEEDS[t], task='T%d' % t,
                   seed=s, n_src=len(src), src_ids='|'.join(str(x) for x in src),
                   n_ckpt=st['n'], last1='%.4f' % st['last1'],
                   last5='%.4f' % st['last5'], last10='%.4f' % st['last10'],
                   peak='%.4f' % st['peak'], seconds='%.1f' % dt, log=lf)
        append_row(row, cf)
        print('[%s] DONE %s %s T%d seed%d last5=%.2f n=%d (%.0fs)'
              % (time.strftime('%H:%M:%S'), m, lv, t, s, st['last5'], st['n'], dt),
              flush=True)


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(PAIRROOT, '_units_*.csv'))):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                if int(r.get('n_ckpt') or 0) >= 60:
                    rows.append(r)
    seen = {}
    for r in rows:
        seen[(r['method'], r['task'], r['level'], r['seed'])] = r
    return list(seen.values())


def do_report(a):
    rows = load_rows()
    methods = sorted({r['method'] for r in rows}) or ['AGMamba']
    tasks = sorted({r['task'] for r in rows}, key=lambda s: int(s[1:]))
    for tk in tasks:
        t = int(tk[1:])
        table = levels_for(t)
        print('=' * 104)
        print('DIRG %s (target %d Hz)   n_src per level: %s'
              % (tk, SPEEDS[t], {k: len(v) for k, v in table.items()}))
        for lv in LEVEL_ORDER:
            print('   %-6s = %-22s ratios %s'
                  % (lv, '|'.join(DOMAINS[x] for x in table[lv]),
                     '/'.join('%.3f' % ratio(x, t) for x in table[lv])))
        print('=' * 104)
        print('%-9s %8s %8s %8s %8s %8s %3s'
              % ('method', 'all4', 'near2', 'far2', 'near1', 'far1', 'n'))
        for m in methods:
            cells = []
            for lv in ['all4', 'near2', 'far2', 'near1', 'far1']:
                v = [float(r['last5']) for r in rows
                     if r['method'] == m and r['task'] == tk and r['level'] == lv]
                cells.append('%8.2f' % np.mean(v) if v else '%8s' % '-')
            n = max([len([r for r in rows if r['method'] == m and r['task'] == tk
                          and r['level'] == lv]) for lv in LEVEL_ORDER] or [0])
            print('%-9s %s %3d' % (m, ' '.join(cells), n))
        idx = {(r['method'], r['level'], r['seed']): float(r['last5']) for r in rows
               if r['task'] == tk}
        print()
        print('PAIRED (same method, same seed, %s):' % tk)
        for m in methods:
            for la, lb, tag in (('near1', 'far1', 'distance  k=1'),
                                ('near2', 'far2', 'distance  k=2'),
                                ('all4', 'near2', 'MORE-SOURCES (add 2 far)')):
                seeds_here = sorted({k[2] for k in idx if k[0] == m})
                d = [idx[(m, la, s)] - idx[(m, lb, s)] for s in seeds_here
                     if (m, la, s) in idx and (m, lb, s) in idx]
                if d:
                    print('  %-9s %-9s - %-9s (=%-28s) delta=%+7.2f n=%d  %s'
                          % (m, la, lb, tag, float(np.mean(d)), len(d),
                             ' '.join('%+.1f' % x for x in d)))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--methods', default='AGMamba')
    ap.add_argument('--targets', default='0,2')
    ap.add_argument('--levels', default='all4,near2,far2,near1,far1')
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
