#!/usr/bin/python
# -*- coding:utf-8 -*-
"""P3: does per-domain spectral pre-whitening generalise beyond DIRG?

P2/P2b (doc section 12) showed, on DIRG: +15.83 pooled over all five LODO
targets, 22/25 seed-paired comparisons positive, t=7.4, while a GLOBAL filter
controls at -13.64 (10/10 negative).  This runner repeats the `none` vs
`tgtmean` comparison on the other benchmarks.

Protocol is taken verbatim from experiments/run_cell.py:
    task  = RC.TASKS[dataset][task_name]
    srcs, tar, num_cls = RC.load_data(dataset, task)
    steps = 60 * ceil(len(srcs[0]) / 64)          <- do NOT change
so the `none` arm reproduces the delivered results bit-for-bit.

The whitening is injected by PATCHING the dataset module's `data_load`, which is
called once per (domain, num_samples) -- i.e. per domain.  No target label is
used anywhere: the target domain's filter is computed from its own (unlabelled)
windows.

Arms
  none     : untouched loader (in-batch reference; must match delivered numbers)
  tgtmean  : x' = irfft(rfft(x - mean) * H_d),  H_d = 1/sqrt(Sbar_d) normalised

Usage
  python experiments/run_prep.py --dataset PHM2009 --tasks T0 --seeds 0 \
      --arms none,tgtmean --steps 60          # smoke
  python experiments/run_prep.py --dataset PHM2009 --tasks T0,T2 --seeds 0,1,2 \
      --arms none,tgtmean --resume
  python experiments/run_prep.py --dataset PHM2009 --tasks T0,T1,T2,T3 \
      --arms none,tgtmean --report
"""
import argparse
import importlib
import logging
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault('AGM_FORCE_PY_SCAN', '1')

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import experiments.run_cell as RC                               # noqa: E402
from utils.DatasetClass import InfiniteDataLoader                # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

OUTROOT = os.path.join(_ROOT, 'results_prep')
METHOD = 'AGMamba'
TAG = ''                    # e.g. '_gap' -> results_prep/<DS>_gap/  (P4)
MODULES = {'PHM2009': 'datasets.PHM_2009_Gearbox',
           'PU': 'datasets.PU_bearing',
           'HUST': 'datasets.HUST_bearing'}


# ---------------------------------------------------------------- whitening
def spec_filter(X, eps=1e-8, beta=1.0):
    """H_d = (Sbar_d / mean(Sbar_d)) ** (-beta/2).

    beta=1.0 reproduces the P2/P3 `tgtmean` arm EXACTLY (same 1/sqrt code path).
    beta=0.0 collapses to H == 1 (per-window centring only == `none` up to the
    downstream per-sample mean-std normalisation) -> built-in validity control.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    W = np.hanning(X.shape[1])
    S = (np.abs(np.fft.rfft(Xc * W, axis=1)) ** 2).mean(axis=0)
    S = S / S.mean()
    if beta == 1.0:
        return 1.0 / np.sqrt(S + eps)
    return (S + eps) ** (-0.5 * float(beta))


def apply_filter(X, H):
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.fft.irfft(np.fft.rfft(Xc, axis=1) * H, n=X.shape[1], axis=1)


_RAW_CACHE = {}


def parse_arm(spec):
    """'none' | 'tgtmean' | 'tgtmean@0.5'  ->  (name, beta)"""
    if '@' in spec:
        name, b = spec.split('@', 1)
        return name.strip(), float(b)
    return spec.strip(), 1.0


def install(data_name, arm, beta=1.0):
    """Patch <dataset module>.data_load according to arm."""
    mod = importlib.import_module(MODULES[data_name])
    orig = getattr(mod, '_orig_data_load', None) or mod.data_load
    mod._orig_data_load = orig

    def raw(*a, **k):
        key = (data_name, repr(a), repr(sorted(k.items())))
        if key not in _RAW_CACHE:
            _RAW_CACHE[key] = orig(*a, **k)
            print('    [load] %s %s' % (data_name, key[1]), flush=True)
        return _RAW_CACHE[key]

    if arm == 'none':
        mod.data_load = raw
        return
    if arm != 'tgtmean':
        raise SystemExit('unknown arm %r' % arm)

    def whiten(*a, **k):
        data, label = raw(*a, **k)
        X = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in data])
        H = spec_filter(X, beta=beta)
        Xw = apply_filter(X, H)
        return [[v.reshape(-1, 1).astype(np.float32) for v in Xw], label]

    mod.data_load = whiten


# ---------------------------------------------------------------- one unit
def run_one(arm, data_name, task_name, seed, device, logfile, steps_override):
    RC.setup_seed(seed)
    task = RC.TASKS[data_name][task_name]
    srcs_raw, tar_raw, n_cls = RC.load_data(data_name, task)
    srcs = [RC.ResampleDataset(s) for s in srcs_raw]
    tar = RC.ResampleDataset(tar_raw)

    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = steps_override or (RC.EPOCHS * iters)
    ckpt = iters

    mod = importlib.import_module(RC.METHOD_MODULES.get(METHOD, METHOD))
    cfg = RC.make_config(mod, METHOD, n_cls, steps, ckpt, device, n_src=len(srcs))
    cfg.n_domains = len(srcs)
    print('[cfg] %s %s arm=%s seed=%d steps=%d ckpt=%d n_src=%d w_mmd=%g w_adv=%g'
          % (data_name, task_name, arm, seed, cfg.steps, ckpt, len(srcs),
             getattr(cfg, 'w_mmd', float('nan')),
             getattr(cfg, 'w_adv', float('nan'))), flush=True)
    model = getattr(mod, RC.METHOD_MODELS[METHOD])(cfg)

    logger = logging.getLogger('%s_%s_%s_seed%d' % (data_name, arm, task_name, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    train_loaders = [InfiniteDataLoader(s, RC.BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=RC.BATCH_SIZE, shuffle=False,
                             num_workers=0)
    res = model.train_model(zip(*train_loaders), [test_loader], logger)
    acc = np.asarray(res['acces'], dtype=float)
    return acc[:, 0] * 100.0 if acc.size else np.array([])


def summarize(acc):
    if acc.size == 0:
        return dict(n=0, last1=float('nan'), last5=float('nan'),
                    last10=float('nan'), peak=float('nan'))
    return dict(n=int(acc.size), last1=float(acc[-1]),
                last5=float(acc[-5:].mean()), last10=float(acc[-10:].mean()),
                peak=float(acc.max()))


def runroot(ds):
    return os.path.join(OUTROOT, ds + TAG)


def summarize_csv(ds, arms, tasks):
    import csv as _csv
    p = os.path.join(runroot(ds), '_summary.csv')
    if not os.path.exists(p):
        print('no summary yet:', p)
        return
    rows = [r for r in _csv.DictReader(open(p, encoding='utf-8'))
            if int(r.get('n_eval') or 0) >= 60]
    print('\n%s\nP3  %s  --  last5\n%s' % ('=' * 74, ds, '=' * 74))
    print('%-10s %8s %8s %8s %6s   %s'
          % ('arm', 'mean', 'std', 'median', 'n', 'per-task'))
    for a in arms:
        sel = [r for r in rows if r['arm'] == a]
        if not sel:
            continue
        v = np.array([float(r['last5']) for r in sel])
        cells = []
        for t in tasks:
            vv = [float(r['last5']) for r in sel if r['task'] == t]
            cells.append('%s=%s' % (t, ('%.2f' % np.mean(vv)) if vv else '-'))
        print('%-10s %8.2f %8.2f %8.2f %6d   %s'
              % (a, v.mean(), v.std(), np.median(v), len(v), ' '.join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PHM2009', choices=list(MODULES))
    ap.add_argument('--tasks', default='T0,T2')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--arms', default='none,tgtmean')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--tag', default='',
                    help="suffix for the output dir, e.g. '_gap' -> "
                         "results_prep/<DS>_gap/ (keeps results_prep/<DS>/ clean)")
    ap.add_argument('--override', default='',
                    help='semicolon-separated overrides of RC.TASKS, '
                         'e.g. "T3=[0,4]>[12];T9=[0,4]>[9]"  '
                         '(sources > target)')
    args = ap.parse_args()

    global TAG
    TAG = args.tag

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    if args.override:
        for spec in args.override.split(';'):
            spec = spec.strip()
            if not spec:
                continue
            name, rhs = spec.split('=', 1)
            lhs, tgt = rhs.split('>')

            def _ids(s):
                return [int(x) for x in s.replace('[', ' ').replace(']', ' ')
                        .replace(',', ' ').split() if x.strip()]

            _t = name.strip()
            _S, _G = _ids(lhs), _ids(tgt)
            RC.TASKS[args.dataset][_t] = [_S, _G]
            print('[override] %s %s -> src %s (k=%d)  tgt %s'
                  % (args.dataset, _t, _S, len(_S), _G), flush=True)
            if _t not in tasks:
                tasks.append(_t)

    if args.report:
        summarize_csv(args.dataset, arms, tasks)
        return

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('P3: per-domain spectral pre-whitening on %s%s'
          % (args.dataset, ('  [tag=%s]' % TAG) if TAG else ''))
    for t in tasks:
        s, g = RC.TASKS[args.dataset][t]
        print('  %-4s src %s -> tgt %s' % (t, s, g))
    print('  arms=%s seeds=%s device=%s steps=%s'
          % (arms, seeds, device, args.steps if args.steps else '60*ceil(N/64)'))
    print('  out: %s\n' % runroot(args.dataset), flush=True)

    root = runroot(args.dataset)
    os.makedirs(root, exist_ok=True)
    sump = os.path.join(root, '_summary.csv')
    new = not os.path.exists(sump)
    fsum = open(sump, 'a', encoding='utf-8', newline='')
    if new:
        fsum.write('dataset,arm,task,seed,n_eval,last1,last5,last10,peak,'
                   'seconds,log\n')
    import csv as _csv
    try:
        done = {(r['dataset'], r['arm'], r['task'], r['seed'])
                for r in _csv.DictReader(open(sump, encoding='utf-8'))
                if int(r.get('n_eval') or 0) >= 60}
    except Exception:                                            # noqa: BLE001
        done = set()

    for arm_spec in arms:
        arm, beta = parse_arm(arm_spec)
        install(args.dataset, arm, beta)
        for tname in tasks:
            for seed in seeds:
                key = (args.dataset, arm_spec, tname, str(seed))
                if args.resume and key in done and args.steps is None:
                    print('[skip] %s %s seed%d' % (arm_spec, tname, seed),
                          flush=True)
                    continue
                d = os.path.join(root, arm_spec, tname)
                os.makedirs(d, exist_ok=True)
                log = os.path.join(d, '%s_%s_%s_seed%d.log'
                                   % (args.dataset, tname,
                                      arm_spec.replace('@', '_b'), seed))
                t0 = time.time()
                try:
                    acc = run_one(arm_spec, args.dataset, tname, seed, device,
                                  log, args.steps)
                    st = summarize(acc)
                    dt = time.time() - t0
                    np.savetxt(log + '_acc.csv', acc, fmt='%.4f')
                    print('[%s] %-11s %-4s seed%d  last5=%.2f  n=%d (%.0fs)'
                          % (time.strftime('%H:%M:%S'), arm_spec, tname, seed,
                             st['last5'], st['n'], dt), flush=True)
                    fsum.write('%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%s\n'
                               % (args.dataset, arm_spec, tname, seed, st['n'],
                                  st['last1'], st['last5'], st['last10'],
                                  st['peak'], dt, os.path.relpath(log, _ROOT)))
                    fsum.flush()
                except Exception as e:                            # noqa: BLE001
                    import traceback
                    print('[FAIL] %s %s seed%d : %s'
                          % (arm_spec, tname, seed, str(e)[:300]), flush=True)
                    traceback.print_exc()
    fsum.close()
    print('\n### P3 ALL DONE @ %s' % time.strftime('%F %T'), flush=True)
    summarize_csv(args.dataset, arms, tasks)


if __name__ == '__main__':
    main()
