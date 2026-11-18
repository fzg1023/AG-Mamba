#!/usr/bin/python
# -*- coding:utf-8 -*-
"""P2: does per-domain spectral pre-whitening ("tgtmean") help the REAL model?

P1 (doc section 11) showed, with a linear probe, that subtracting the TARGET
domain's own mean spectrum is the single best cheap preprocessing on all four
datasets -- DIRG +13.4 -- and that a global (source-pooled) correction is worth
exactly +0.0.

This runner tests the waveform-domain analogue of that operation on the real
AG-Mamba pipeline, DIRG only:

    H_d(f) = 1 / sqrt( Sbar_d(f) )        (per-domain matched pre-whitening)
    x' = irfft( rfft(x - mean) * H_d )    then the usual per-sample mean-std

`Sbar_d` is the average power spectrum of domain d's OWN windows -- for the
target that is unlabelled data, which the DG protocol gives us. No target label
is used anywhere.

Arms
  none     : identical to experiments/run_dirg.py  (in-batch reference)
  tgtmean  : per-domain matched pre-whitening      <- the candidate
  srcglob  : one global filter from the pooled source windows (control;
             P1's `srcmean` analogue, expected ~0)

Everything is written under results_dirg_prep/ -- the delivered paper results
in results_dirg/ are never touched.

Usage
  python experiments/run_dirg_prep.py --load 1000 --ch 0 \
      --tasks T0,T4 --seeds 0,1,2,3,4 --arms none,tgtmean
  python experiments/run_dirg_prep.py --load 1000 --ch 0 --report
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
import pandas as pd                                             # noqa: E402
import torch                                                    # noqa: E402

import experiments.run_cell as RC                               # noqa: E402
from datasets.DIRG_bearing import (                             # noqa: E402
    DIRG_DG, DOMAINS, CLASSES, num_cls, data_load, available_domains)
from datasets.SequenceDatasets import dataset                    # noqa: E402
from utils.DatasetClass import InfiniteDataLoader                # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

OUTROOT = os.path.join(_ROOT, 'results_dirg_prep')
TAG = ''                # e.g. '_b050' -> results_dirg_prep/L1000_ch0_b050/
NTR = 50
NTE = 50
METHOD = 'AGMamba'


# ---------------------------------------------------------------- whitening
def spec_filter(X, eps=1e-8, beta=1.0):
    """H(f) = (Sbar(f)/mean(Sbar)) ** (-beta/2).

    beta=1.0 reproduces the P2/P2b arm EXACTLY (same 1/sqrt code path).
    beta=0.0 -> H == 1 (centring only) -- a validity control.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    W = np.hanning(X.shape[1])
    S = (np.abs(np.fft.rfft(Xc * W, axis=1)) ** 2).mean(axis=0)
    S = S / S.mean()
    if beta == 1.0:
        return 1.0 / np.sqrt(S + eps)
    return (S + eps) ** (-0.5 * float(beta))


def parse_arm(spec):
    """'none' | 'tgtmean' | 'tgtmean@0.5'  ->  (name, beta)"""
    if '@' in spec:
        name, b = spec.split('@', 1)
        return name.strip(), float(b)
    return spec.strip(), 1.0


def apply_filter(X, H):
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.fft.irfft(np.fft.rfft(Xc, axis=1) * H, n=X.shape[1], axis=1)


def get_raw(dom, load, ch, n, mode):
    d, y = data_load(dom, load, n, mode=mode, ch=ch)
    return (np.stack([np.asarray(a, dtype=np.float64).ravel() for a in d]),
            np.asarray(y))


def build_split(task, load, ch, arm, transforms, beta=1.0):
    src_ids, tgt_ids = task
    tgt = tgt_ids[0]
    raw_s = {d: get_raw(d, load, ch, NTR, 'head') for d in src_ids}
    raw_t = get_raw(tgt, load, ch, NTE, 'tail')

    if arm == 'tgtmean':
        for d in list(src_ids) + [tgt]:
            X, y = raw_s[d] if d in raw_s else raw_t
            H = spec_filter(X, beta=beta)
            if d in raw_s:
                raw_s[d] = (apply_filter(X, H), y)
            else:
                raw_t = (apply_filter(X, H), y)
    elif arm == 'srcglob':
        allsrc = np.concatenate([raw_s[d][0] for d in src_ids], axis=0)
        H = spec_filter(allsrc)
        raw_s = {d: (apply_filter(raw_s[d][0], H), raw_s[d][1]) for d in src_ids}
        raw_t = (apply_filter(raw_t[0], H), raw_t[1])
    elif arm != 'none':
        raise SystemExit('unknown arm %r' % arm)

    def mk(X, y, key):
        df = pd.DataFrame({'data': [x.reshape(-1, 1).astype(np.float32)
                                    for x in X], 'label': y})
        return dataset(list_data=df, transform=transforms[key])

    srcs = [mk(raw_s[d][0], raw_s[d][1], 'train') for d in src_ids]
    tar = mk(raw_t[0], raw_t[1], 'test')
    return srcs, tar


# ---------------------------------------------------------------- one unit
def run_one(arm, task_name, task, load, ch, seed, device, logfile, steps_override,
            transforms, beta=1.0):
    RC.setup_seed(seed)
    srcs_raw, tar_raw = build_split(task, load, ch, arm, transforms, beta)
    srcs = [RC.ResampleDataset(s) for s in srcs_raw]
    tar = RC.ResampleDataset(tar_raw)

    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = steps_override or (RC.EPOCHS * iters)
    checkpoint_freq = iters

    mod = importlib.import_module(RC.METHOD_MODULES.get(METHOD, METHOD))
    cfg = RC.make_config(mod, METHOD, num_cls, steps, checkpoint_freq, device,
                         n_src=len(srcs))
    cfg.n_domains = len(srcs)
    print('[cfg] arm=%s task=%s seed=%d steps=%d ckpt=%d n_src=%d '
          'w_mmd=%g w_adv=%g'
          % (arm, task_name, seed, cfg.steps, cfg.checkpoint_freq, len(srcs),
             getattr(cfg, 'w_mmd', float('nan')),
             getattr(cfg, 'w_adv', float('nan'))), flush=True)
    model = getattr(mod, RC.METHOD_MODELS[METHOD])(cfg)

    logger = logging.getLogger('%s_%s_seed%d' % (arm, task_name, seed))
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
    acc = acc[:, 0] * 100.0 if acc.size else np.array([])
    return acc


def summarize(acc):
    if acc.size == 0:
        return dict(n=0, last1=float('nan'), last5=float('nan'),
                    last10=float('nan'), peak=float('nan'))
    return dict(n=int(acc.size), last1=float(acc[-1]),
                last5=float(acc[-5:].mean()), last10=float(acc[-10:].mean()),
                peak=float(acc.max()))


def runroot(load, ch):
    return os.path.join(OUTROOT, 'L%d_ch%d%s' % (load, ch, TAG))


def summarize_csv(load, ch, arms, tasks, seeds):
    import csv as _csv
    p = os.path.join(runroot(load, ch), '_summary.csv')
    if not os.path.exists(p):
        print('no summary yet:', p)
        return
    rows = [r for r in _csv.DictReader(open(p, encoding='utf-8'))
            if int(r.get('n_eval') or 0) >= 60]
    print('\n%s\nP2  DIRG L%d Ch%d  --  last5 (pooled over task x seed)\n%s'
          % ('=' * 74, load, ch, '=' * 74))
    print('%-10s %8s %8s %8s %6s   %s' % ('arm', 'mean', 'std', 'median', 'n',
                                          'per-task'))
    for a in arms:
        sel = [r for r in rows if r['arm'] == a]
        if not sel:
            continue
        v = np.array([float(r['last5']) for r in sel])
        pt = ' '.join('%s=%.2f' % (t, np.mean([float(r['last5']) for r in sel
                                               if r['task'] == t]) or float('nan'))
                      for t in tasks
                      if any(r['task'] == t for r in sel))
        print('%-10s %8.2f %8.2f %8.2f %6d   %s'
              % (a, v.mean(), v.std(), np.median(v), len(v), pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000)
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--tasks', default='T0,T4')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--arms', default='none,tgtmean')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--tag', default='',
                    help="suffix for the output dir, e.g. '_b050' -> "
                         "results_dirg_prep/L1000_ch0_b050/")
    args = ap.parse_args()

    global TAG
    TAG = args.tag

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    tasks_want = [t.strip() for t in args.tasks.split(',') if t.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    ids, sps = available_domains(args.load, ch=args.ch)
    tasks_all = {}
    for k, tgt in enumerate(ids):
        tasks_all['T%d' % k] = [[d for d in ids if d != tgt], [tgt]]
    tasks = {t: tasks_all[t] for t in tasks_want}

    if args.report:
        summarize_csv(args.load, args.ch, arms, tasks_want, seeds)
        return

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = DIRG_DG([[0], [0]], load=args.load, ch=args.ch).data_transforms
    print('P2: per-domain spectral pre-whitening on DIRG')
    print('  domains: %s' % ', '.join(DOMAINS[i] for i in ids))
    for t, (s, g) in tasks.items():
        print('  %-4s src %s -> tgt %s' % (t, [DOMAINS[i] for i in s],
                                           DOMAINS[g[0]]))
    print('  arms=%s seeds=%s device=%s steps=%s'
          % (arms, seeds, device, args.steps if args.steps else '60*ceil(N_src/64)'))
    print('  out: %s\n' % OUTROOT, flush=True)

    root = runroot(args.load, args.ch)
    os.makedirs(root, exist_ok=True)
    sump = os.path.join(root, '_summary.csv')
    new = not os.path.exists(sump)
    fsum = open(sump, 'a', encoding='utf-8', newline='')
    if new:
        fsum.write('load,ch,arm,task,seed,n_eval,last1,last5,last10,peak,'
                   'seconds,log\n')
    import csv as _csv
    try:
        done = {(r['load'], r['arm'], r['task'], r['seed'])
                for r in _csv.DictReader(open(sump, encoding='utf-8'))
                if int(r.get('n_eval') or 0) >= 60}
    except Exception:                                            # noqa: BLE001
        done = set()

    for arm_spec in arms:
        arm, beta = parse_arm(arm_spec)
        for tname, (src_ids, tgt_ids) in tasks.items():
            for seed in seeds:
                key = (str(args.load), arm_spec, tname, str(seed))
                if args.resume and key in done and args.steps is None:
                    print('[skip] %s %s seed%d' % (arm_spec, tname, seed),
                          flush=True)
                    continue
                d = os.path.join(root, arm_spec, tname)
                os.makedirs(d, exist_ok=True)
                log = os.path.join(d, 'DIRG_%s_%s_L%d_seed%d.log'
                                   % (tname, arm_spec.replace('@', '_b'),
                                      args.load, seed))
                t0 = time.time()
                try:
                    # NOTE: pass the BASE arm name (no @beta) -- build_split
                    # dispatches on it; beta travels as a separate argument.
                    acc = run_one(arm, tname, [src_ids, tgt_ids], args.load,
                                  args.ch, seed, device, log, args.steps,
                                  transforms, beta)
                    st = summarize(acc)
                    dt = time.time() - t0
                    np.savetxt(log + '_acc.csv', acc, fmt='%.4f')
                    print('[%s] %-11s %-4s seed%d  last5=%.2f  n=%d (%.0fs)'
                          % (time.strftime('%H:%M:%S'), arm_spec, tname, seed,
                             st['last5'], st['n'], dt), flush=True)
                    fsum.write('%d,%d,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%s\n'
                               % (args.load, args.ch, arm_spec, tname, seed,
                                  st['n'], st['last1'], st['last5'],
                                  st['last10'], st['peak'], dt,
                                  os.path.relpath(log, _ROOT)))
                    fsum.flush()
                except Exception as e:                            # noqa: BLE001
                    import traceback
                    print('[FAIL] %s %s seed%d : %s'
                          % (arm_spec, tname, seed, str(e)[:300]), flush=True)
                    traceback.print_exc()
    fsum.close()
    print('\n### P2 ALL DONE @ %s' % time.strftime('%F %T'), flush=True)
    summarize_csv(args.load, args.ch, arms, tasks_want, seeds)


if __name__ == '__main__':
    main()
