#!/usr/bin/python
# -*- coding:utf-8 -*-
"""§6.6 SNR 噪声网格：**训练干净、测试加噪**的鲁棒性探针。

设计（与手稿 Fig. 8d 面板里写明的一致）
--------------------------------------
- 每个单元 (method, task, seed)：训练用**干净**数据，与主协议**同一路径**
  （`load_data` + `make_config` + `METHOD_MODELS[m](cfg)`，HUST 480 步、种子一致）。
- 评估集复制成 5 份并**一次性**传给 `train_model` 的 test loaders：
  `clean / SNR 10 / 5 / 0 / −5 dB`。逐窗加性高斯白噪声，噪声标准差
  = 该窗 RMS / 10^(SNR/20)，即相对每窗能量定义 SNR。
- 每个 checkpoint 的 `result['acces']` 直接给出 5 列精度（**不依赖各方法的日志格式**
  —— 各基线 `train_model` 的日志写法不统一，靠日志解析会出错）。第 0 列 = 干净集，
  与主口径 `last5` 可比；后 4 列 = 各 SNR。
- ⚠ **噪声实现必须与方法无关**：噪声実現只由 (ds_tag, window_index, snr, seed) 决定，
  三种方法评估的是**同一组含噪测试窗**（否则方法间差异会混入不同的噪声抽样）。
- 方法集：`AGMamba`（交付配置，即本方法）+ 本基准最强的两个基线 `WIDAN` / `MRA_SNN`。
- 目标：HUST `T0`(20 Hz) / `T4`(40 Hz) / `T9`(80 Hz)；种子 0/1/2。

用法
----
    AGM_FORCE_PY_SCAN=1 python experiments/run_snr_grid.py
    AGM_FORCE_PY_SCAN=1 python experiments/run_snr_grid.py --steps 8 --tasks T0 --seeds 0   # 冒烟
    AGM_RESULTS=... 默认写到 results_probe_snr/

输出
----
    results_probe_snr/<DS>/<METHOD>/snr/<TASK>/<DS>_<TASK>_<METHOD>_fc_snr_seed<S>.log
    logs/_s66_snr.json          每个单元的 5 列 last5（供 summary_snr.py 汇总）
由 `analysis/summary_snr.py` 汇总成 `docs/manuscript/_s66_snr.md`。
"""
import argparse
import importlib
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['AGM_FORCE_PY_SCAN'] = '1'

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments'))

import run_cell as RC                                         # noqa: E402
from utils.DatasetClass import InfiniteDataLoader             # noqa: E402

SNRS = [None, 10.0, 5.0, 0.0, -5.0]      # None = 干净
DATA = 'HUST'
METHODS = ['AGMamba', 'WIDAN', 'MRA_SNN']
TASKS = ['T0', 'T4', 'T9']
SEEDS = [0, 1, 2]
OUT_JSON = os.path.join(_ROOT, 'logs', '_s66_snr.json')


class NoisyView(torch.utils.data.Dataset):
    """在**样本级**加逐窗 AWGN；噪声只依赖 (tag, i, snr, seed)，与方法无关。"""

    def __init__(self, ds, snr_db, tag, seed=0):
        self.ds, self.snr, self.tag, self.seed = ds, snr_db, tag, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.snr is None:
            return x, y
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        key = abs(hash((self.tag, int(i), float(self.snr), int(self.seed)))) \
            % (2 ** 32)
        rng = np.random.default_rng(key)
        rms = float(np.sqrt(np.mean(a ** 2))) + 1e-12
        sig = rms / (10.0 ** (self.snr / 20.0))
        out = a + rng.normal(0.0, sig, a.shape)
        return torch.from_numpy(out.astype(np.float32)).reshape(
            np.shape(x)), y


def arm_dir():
    root = os.environ.get('AGM_RESULTS') or os.path.join(_ROOT,
                                                         'results_probe_snr')
    return root


def logpath(method, task, seed, step_tag=''):
    d = os.path.join(arm_dir(), DATA, method, 'snr', task)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, '%s_%s_%s_fc_snr%s_seed%d.log'
                        % (DATA, task, method, step_tag, seed))


def accs_path():
    return OUT_JSON


def load_accs():
    if os.path.exists(accs_path()):
        with open(accs_path(), encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_accs(d):
    tmp = accs_path() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, accs_path())


def run_cell(method, task, seed, steps_override, quick):
    from utils.DatasetClass import InfiniteDataLoader as IDL
    RC.setup_seed(seed)
    tlist = RC.TASKS[DATA][task]
    srcs, tar, num_cls = RC.load_data(DATA, tlist)
    srcs = [RC.ResampleDataset(s) for s in srcs]
    tar = RC.ResampleDataset(tar)
    tag = '%s_%s_s%d' % (DATA, task, seed)

    loaders = [DataLoader(tar, batch_size=RC.BATCH_SIZE, shuffle=False,
                          num_workers=0)]
    for snr in SNRS[1:]:
        loaders.append(DataLoader(NoisyView(tar, snr, tag, seed),
                                  batch_size=RC.BATCH_SIZE, shuffle=False,
                                  num_workers=0))

    mod = importlib.import_module(RC.METHOD_MODULES.get(method, method))
    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = steps_override or RC.EPOCHS * iters
    ckpt = 1 if quick else iters
    cfg = RC.make_config(mod, method, num_cls, steps, ckpt, 'cuda',
                         n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = getattr(mod, RC.METHOD_MODELS[method])(cfg)

    import logging
    lg = logging.getLogger('snr_%s_%s_%d' % (method, task, seed))
    lg.handlers = []
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fh = logging.FileHandler(logpath(method, task, seed), mode='w',
                             encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    lg.addHandler(fh)

    train_loaders = [IDL(s, RC.BATCH_SIZE) for s in srcs]
    train_iter = zip(*train_loaders)
    t0 = time.time()
    res = model.train_model(train_iter, loaders, lg)
    acc = np.asarray(res['acces'], dtype=float) * 100.0      # (n_ckpt, 1+4)
    dt = time.time() - t0

    arr = np.array(res['acces'][-5:], dtype=float) * 100.0   # last5 行
    means = arr.mean(axis=0)
    return means, dt, len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None,
                    help='覆盖步数（冒烟用，如 8）')
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--quick', action='store_true',
                    help='checkpoint_freq=1（冒烟用）')
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(',') if t]
    seeds = [int(s) for s in a.seeds.split(',') if s]
    methods = [m for m in a.methods.split(',') if m]
    store = load_accs()
    print('[snr] dir=%s  methods=%s tasks=%s seeds=%s steps=%s'
          % (arm_dir(), methods, tasks, seeds,
             a.steps or 'protocol(%d)' % (RC.EPOCHS * 8)), flush=True)
    print('[snr] snr levels = %s (None = clean)' % SNRS, flush=True)

    t_all = time.time()
    for task in tasks:
        for method in methods:
            for seed in seeds:
                key = '%s|%s|%s|%d' % (DATA, method, task, seed)
                if key in store and not a.steps:
                    print('[skip] %s  -> %s' % (key, np.round(store[key], 2)),
                          flush=True)
                    continue
                try:
                    means, dt, n = run_cell(method, task, seed, a.steps,
                                            a.quick)
                except Exception as e:
                    print('[FAIL] %s : %s' % (key, str(e)[:150]), flush=True)
                    continue
                if not a.steps:
                    store[key] = [float(v) for v in means]
                    save_accs(store)
                print('[%s] %-22s n_ckpt=%3d  clean=%.2f  '
                      'snr10=%.2f snr5=%.2f snr0=%.2f snr-5=%.2f  (%.0fs)'
                      % (time.strftime('%H:%M:%S'), key, n, means[0],
                         means[1], means[2], means[3], means[4], dt),
                      flush=True)
    print('[snr] done %.0fs -> %s' % (time.time() - t_all, OUT_JSON),
          flush=True)


if __name__ == '__main__':
    main()
