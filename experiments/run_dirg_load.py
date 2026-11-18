#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 附加试验：**域 = 不同载荷、转速固定** 的 LODO 跨域对比（轴转置）。

与 `experiments/run_dirg.py`（域 = 转速、载荷固定）互为镜像。
⚠ 附加试验，不参与论文；产物只写 `results_dirg/` 与 `logs/dirg_*`。

为什么单开一个文件：`run_dirg.py` 的 `--load` 是标量、`build_tasks` 按转速枚举；
本文件把两个轴对调，且**不改动任何现有文件**（避免影响已完成的臂）。

数据约束（来自 npz 实测）：
    L0     -> 100/200/300/400/500   5 档转速
    L1000  -> 100/200/300/400/500
    L1400  -> 100/200/300/400
    L1800  -> 100/200/300
=> 同转速下可用的载荷档数：100/200/300 Hz 各 **4** 个（4 折 LODO）；
   400 Hz **3** 个（3 折）；500 Hz 只有 **2** 个（仅 1 源域，不是真正的 DG，默认不跑）。

任务命名：`T{速度}_{留出的载荷}`，例如 `T300_0`、`T300_1400`。
协议与 run_dirg 完全一致：每域每类 head-50 训练 / tail-50 测试、
steps = 60*ceil(N_src/64)（4 源域 -> 360）、每单元 60 个评估点、主口径 last5。

用法：
  # 冒烟
  python experiments/run_dirg_load.py --speeds 300 --methods AGMamba --seeds 0 --steps 6
  # 正式：3 个速度 x 4 折 x 5 种子 = 60 单元
  python experiments/run_dirg_load.py --speeds 100,200,300 \
      --methods AGMamba,AGG,MMD,CORAL,DANN,WIDAN,MRA_SNN --seeds 0,1,2,3,4
  # 只看已有结果
  python experiments/run_dirg_load.py --speeds 100,200,300 --report
"""
import argparse
import csv
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
import pandas as pd                                             # noqa: E402

import experiments.run_cell as RC                               # noqa: E402
from datasets.DIRG_bearing import (                             # noqa: E402
    SPEEDS, CLASSES, LOAD_TAG, num_cls, data_load, _load_all)
from datasets.SequenceDatasets import dataset                   # noqa: E402
from datasets.sequence_aug import Compose, Reshape, Normalize, Retype  # noqa: E402
from utils.DatasetClass import InfiniteDataLoader                # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

OUTROOT = os.path.join(_ROOT, 'results_dirg')
ALL_LOADS = [0, 1000, 1400, 1800]
SEED_DEFAULT = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- 任务表
def speed_index(speed):
    speed = int(speed)
    if speed not in SPEEDS:
        raise SystemExit('未知转速 %s（可选 %s）' % (speed, SPEEDS))
    return SPEEDS.index(speed)


def available_loads(speed, ch=0, fs=51200):
    """返回该转速下 7 类齐全的载荷档列表（升序）。"""
    z = _load_all(ch, fs)
    out = []
    for ld in ALL_LOADS:
        tag = LOAD_TAG[ld]
        if all('%d_%s_%s' % (int(speed), c, tag) in z.files for c in CLASSES):
            out.append(ld)
    return out


def build_tasks(speeds, ch=0, fs=51200, min_src=1, drop_l0=False):
    """{task_name: [source_loads, [target_load], speed]}，外加清单。"""
    tasks, report = {}, []
    for sp in speeds:
        sp = int(sp)
        loads = available_loads(sp, ch=ch, fs=fs)
        if drop_l0:
            loads = [l for l in loads if l != 0]
        report.append((sp, loads))
        if len(loads) - 1 < min_src:
            continue
        for tgt in loads:
            srcs = [ld for ld in loads if ld != tgt]
            tasks['T%d_%d' % (sp, tgt)] = [srcs, [tgt], sp]
    return tasks, report


# ---------------------------------------------------------------- 加载器
class DirgLoadDG(object):
    """转速固定、载荷为域的 DG 数据集（接口同 DIRG_DG）。"""

    def __init__(self, source_loads, target_load, speed, norm_type='mean-std',
                 num_train_samples=50, num_test_samples=50, ch=0, fs=51200,
                 sig=1024, lp_hz=0):
        self.source_loads = list(source_loads)
        self.target_load = int(target_load)
        self.speed = int(speed)
        self.ch = ch
        self.fs = int(fs)
        self.sig = int(sig)
        self.lp_hz = float(lp_hz)
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.data_transforms = {
            'train': Compose([Reshape(), Normalize(norm_type), Retype()]),
            'test': Compose([Reshape(), Normalize(norm_type), Retype()]),
        }

    def _one(self, load, n, mode):
        ld = data_load(speed_index(self.speed), load, n, mode=mode, ch=self.ch,
                       fs=self.fs, sig=self.sig, lp_hz=self.lp_hz)
        df = pd.DataFrame({'data': ld[0], 'label': ld[1]})
        return dataset(list_data=df, transform=self.data_transforms[
            'train' if mode == 'head' else 'test'])

    def train_test_data_split(self):
        srcs = []
        for i, ld in enumerate(self.source_loads):
            ds = self._one(ld, self.num_train_samples, 'head')
            print('  source_%d (L%d @ %dHz): %d' % (
                i + 1, ld, self.speed, len(ds)), flush=True)
            srcs.append(ds)
        tar = self._one(self.target_load, self.num_test_samples, 'tail')
        print('  target  (L%d @ %dHz): %d' % (
            self.target_load, self.speed, len(tar)), flush=True)
        return srcs, tar, num_cls


# ---------------------------------------------------------------- 训练
def run_one(method, task_name, task, seed, device, logfile, steps_override,
            ch=0, fs=51200, sig=1024, lp=0):
    """与 run_dirg.run_one 同构，只把"载荷=域"的加载器换进来。"""
    src_loads, tgt_loads, speed = task[0], task[1], task[2]
    RC.setup_seed(seed)
    srcs_raw, tar_raw, n_cls = DirgLoadDG(
        src_loads, tgt_loads[0], speed, ch=ch, fs=fs, sig=sig,
        lp_hz=lp).train_test_data_split()
    srcs = [RC.ResampleDataset(s, length=int(sig)) for s in srcs_raw]
    tar = RC.ResampleDataset(tar_raw, length=int(sig))

    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = steps_override or (RC.EPOCHS * iters)
    checkpoint_freq = iters

    mod = importlib.import_module(RC.METHOD_MODULES.get(method, method))
    cfg = RC.make_config(mod, method, n_cls, steps, checkpoint_freq, device,
                         n_src=len(srcs))
    cfg.n_domains = len(srcs)
    cfg.data_length = int(sig)
    print('[cfg] method=%s nc=%d steps=%d ckpt=%d n_src=%d | speed=%dHz ch=%d\n'
          '       domains=%s -> target L%d | ids=%s' % (
              method, cfg.num_classes, cfg.steps, cfg.checkpoint_freq,
              len(srcs), speed, ch + 1, src_loads, tgt_loads[0],
              [LOAD_TAG[l] for l in src_loads]), flush=True)

    model = getattr(mod, RC.METHOD_MODELS[method])(cfg)
    logger = logging.getLogger('{}_{}_seed{}'.format(method, task_name, seed))
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


# ---------------------------------------------------------------- 路径
def runroot(fs=51200, sig=1024, tag_extra='', lp=0, arm='LOADAX'):
    tag = str(arm) + '_ch0'
    if int(fs) != 51200:
        tag += '_fs%d' % int(fs)
    if int(sig) != 1024:
        tag += '_sig%d' % int(sig)
    if int(lp):
        tag += '_lp%d' % int(lp)
    if tag_extra:
        tag += '_%s' % tag_extra
    return os.path.join(OUTROOT, tag)


# ---------------------------------------------------------------- 汇总
def report(tasks, methods, seeds, fs=51200, sig=1024, tag_extra='', lp=0, arm='LOADAX'):
    csvp = os.path.join(runroot(fs, sig, tag_extra, lp, arm), '_summary.csv')
    if not os.path.exists(csvp):
        print('无汇总文件:', csvp)
        return
    rows = list(csv.DictReader(open(csvp, encoding='utf-8')))
    print('=' * 78)
    print('DIRG 载荷轴 LODO 汇总（主口径 last5；池化 = 任务 x 种子均值）')
    print('=' * 78)
    print('%-9s %8s %8s %8s %8s %6s' % ('method', 'last5', 'last1', 'last10', 'peak', 'n'))

    def mean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float('nan')

    for m in methods:
        rs = [r for r in rows if r['method'] == m and
              r.get('kind', 'lodo') == 'lodo' and float(r.get('n_eval') or 0) >= RC.EPOCHS]
        if not rs:
            print('%-9s %8s %8s %8s %8s %6d' % (m, '-', '-', '-', '-', 0))
            continue
        print('%-9s %8.2f %8.2f %8.2f %8.2f %6d'
              % (m, mean([float(r['last5']) for r in rs]),
                 mean([float(r['last1']) for r in rs]),
                 mean([float(r['last10']) for r in rs]),
                 mean([float(r['peak']) for r in rs]), len(rs)))
    print()
    print('逐任务（last5，跨种子均值）:')
    hdr = sorted({r['task'] for r in rows if r.get('kind', 'lodo') == 'lodo'})
    print('%-9s' % 'method' + ''.join('%14s' % t for t in hdr))
    for m in methods:
        cells = []
        for t in hdr:
            v = [float(r['last5']) for r in rows if r['method'] == m and r['task'] == t
                 and r.get('kind', 'lodo') == 'lodo' and float(r.get('n_eval') or 0) >= RC.EPOCHS]
            cells.append('%14s' % ('%.1f' % mean(v) if v else '-'))
        print('%-9s' % m + ''.join(cells))
    print('随机线 = %.2f' % (100.0 / num_cls))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speeds', default='100,200,300',
                    help='固定转速列表（逗号分隔，可选 100/200/300/400/500）')
    ap.add_argument('--methods', default='AGMamba')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEED_DEFAULT))
    ap.add_argument('--ch', type=int, default=0)
    ap.add_argument('--fs', type=int, default=51200)
    ap.add_argument('--sig', type=int, default=1024)
    ap.add_argument('--lp', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--arm', default='LOADAX')
    ap.add_argument('--exclude-l0', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    speeds = [int(x) for x in args.speeds.split(',') if x.strip()]
    methods = [x for x in args.methods.split(',') if x]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]

    allt, rep = build_tasks(speeds, ch=args.ch, fs=args.fs, drop_l0=args.exclude_l0)
    print('载荷覆盖情况（转速 -> 可用载荷档）:')
    for sp, loads in rep:
        print('  %3d Hz : %s  -> %d 档载荷 = %d 折 LODO' % (
            sp, [LOAD_TAG[l] for l in loads], len(loads), len(loads)))
    if not allt:
        raise SystemExit('没有可跑的任务')

    tasks = allt
    if args.report:
        report(tasks, methods, seeds, fs=args.fs, sig=args.sig,
               tag_extra=args.tag, lp=args.lp)
        return

    root = runroot(args.fs, args.sig, args.tag, args.lp, args.arm)
    os.makedirs(root, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if dev == 'cuda' and not torch.cuda.is_available():
        dev = 'cpu'

    done = {}
    csvp = os.path.join(root, '_summary.csv')
    if args.resume and os.path.exists(csvp):
        for r in csv.DictReader(open(csvp, encoding='utf-8')):
            done[(r['method'], r['task'], r['seed'])] = r

    print('=' * 78)
    print('DIRG 载荷轴 (同转速/异载荷)  speed=%s ch=%d : %d 任务 x %d 方法 x %d 种子 = %d 单元'
          % (speeds, args.ch, len(tasks), len(methods), len(seeds),
             len(tasks) * len(methods) * len(seeds)))
    print('=' * 78)
    t00 = time.time()
    for method in methods:
        for tname in sorted(tasks):
            task = tasks[tname]
            for seed in seeds:
                old = done.get((method, tname, str(seed)))
                if (old is not None and int(old.get('n_eval') or 0) >= RC.EPOCHS
                        and (args.steps is None or old.get('steps') == str(args.steps))):
                    print('[skip] %s %s seed%d' % (method, tname, seed), flush=True)
                    continue
                d = os.path.join(root, method, tname)
                os.makedirs(d, exist_ok=True)
                log = os.path.join(d, 'DIRGLOAD_%s_%s_seed%d.log' % (tname, method, seed))
                t0 = time.time()
                acc = run_one(method, tname, task, seed, dev, log, args.steps,
                              ch=args.ch, fs=args.fs, sig=args.sig, lp=args.lp)
                s = summarize(acc)
                print('[%s] %-9s %-12s seed%d  last5=%.2f last1=%.2f n=%d (%.0fs)'
                      % (time.strftime('%H:%M:%S'), method, tname, seed,
                         s['last5'], s['last1'], s['n'], time.time() - t0), flush=True)
                with open(csvp, 'a', encoding='utf-8', newline='') as f:
                    w = csv.writer(f, lineterminator='\n')
                    if f.tell() == 0:
                        w.writerow(['speed_axis', 'axis', 'method', 'task', 'seed',
                                    'kind', 'steps', 'n_eval', 'last1', 'last5',
                                    'last10', 'peak', 'seconds', 'log'])
                    w.writerow([int(tname.split('_')[0][1:]), 'load', method, tname, seed,
                                'lodo', args.steps or '', s['n'], '%.4f' % s['last1'],
                                '%.4f' % s['last5'], '%.4f' % s['last10'],
                                '%.4f' % s['peak'], '%.1f' % (time.time() - t0), log])
    print('### ALL DONE @ %s  (%.0f min)' % (time.strftime('%F %T'),
                                             (time.time() - t00) / 60.0))
    report(tasks, methods, seeds, fs=args.fs, sig=args.sig,
           tag_extra=args.tag, lp=args.lp)


if __name__ == '__main__':
    main()
