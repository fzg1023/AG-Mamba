#!/usr/bin/python
# -*- coding:utf-8 -*-
"""在 DG-for-RMFD 的数据上运行 DGFDBenchmark 的 9 个方法（统一 2560 长度、跨工况 4 任务、5 种子）。

用法:
  # 冒烟测试单个组合
  python run_dgfdb.py --method CNNC --dataset PHM2009 --task T0 --seed 0
  # 全部 9 方法 x 2 数据集 x 4 任务 x 5 种子
  python run_dgfdb.py --all
"""
import os
import sys
import argparse
import importlib
import logging
import random
import time

# 仓库根入 path（本项目按目录归位后，入口脚本位于 experiments/ 之下）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import PHM2009_DG, PU_DG, UO_DG, HUST_DG
from utils.DatasetClass import InfiniteDataLoader

# 可复现模式开关（诊断用，默认关闭，不影响任何既有结果）：
#   AGM_DET=1 时强制 cuDNN 使用确定性卷积算法，用于检验"同配置同种子两次运行"
#   是否能逐位一致（见 run_final_eval.py 的 det_ctl / det_ctl2 阶段）。
#   注意：该模式可能与既有日志不可比（算法不同），仅用于诊断；若要作为正式配置
#   使用，必须整批重跑同一比较。
if os.environ.get('AGM_DET', '0') == '1':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


METHOD_MODULES = {'ADN': 'baselines.ADN', 'CCDG': 'baselines.CCDG', 'CCN': 'baselines.CCN', 'CDDG': 'baselines.CDDG', 'CNNC': 'baselines.CNNC', 'DANN': 'baselines.DANN', 'DDGFD': 'baselines.DDGFD', 'DGNIS': 'baselines.DGNIS', 'IEDGNet': 'baselines.IEDGNet', 'MRA_SNN': 'baselines.MRA_SNN', 'UDTL': 'baselines.UDTL', 'WIDAN': 'baselines.WIDAN', 'WhiteningNet': 'baselines.WhiteningNet',
                  # origeight 家族（原仓库 8 方法；DANN 已有独立模块）
                  'AGG': 'baselines.origeight', 'MMD': 'baselines.origeight',
                  'CORAL': 'baselines.origeight',
                  'Triplet_loss': 'baselines.origeight',
                  'Mixup': 'baselines.origeight',
                  'MLDG': 'baselines.origeight',
                  'DAEL': 'baselines.origeight',
                  'AGMamba': 'agmamba.method'}

METHOD_MODELS = {
    'CNNC': 'CNNC',
    'ADN': 'DGDNN',
    'IEDGNet': 'IEDGNet',
    'DGNIS': 'DGNIS',
    'WhiteningNet': 'WhitenNet',
    'CCN': 'CCN',
    'CCDG': 'CCDG',
    'CDDG': 'CDDG',
    'DDGFD': 'DDGFD',
    'AGMamba': 'AGMamba',
    'UDTL': 'UDTL',
    'WIDAN': 'WIDAN',
    'MRA_SNN': 'MRA_SNN',
    'DANN': 'DANN',
    # origeight 家族（类名在 baselines/origeight.py 中；带下划线避免与方法名冲突）
    'AGG': 'AGG',
    'MMD': 'MMD_',
    'CORAL': 'CORAL_',
    'Triplet_loss': 'Triplet_',
    'Mixup': 'Mixup_',
    'MLDG': 'MLDG_',
    'DAEL': 'DAEL',
}

# SGD 优化器需要更高的学习率；Adagrad 优化器标准 lr=0.01，0.001 会欠拟合
LR_OVERRIDE = {
    'CCDG': 0.02,
    'DDGFD': 0.02,
    'ADN': 0.01,      # Adagrad
    'IEDGNet': 0.01,  # Adagrad
    'CCN': 0.01,      # Adagrad
    'AGMamba': 0.001,
    'CNNC': 0.002,        # 快扫调优
    'DGNIS': 0.002,       # 快扫调优
    'WhiteningNet': 0.002,  # 快扫调优
    'CDDG': 0.002,        # 快扫调优
    'UDTL': 0.002,        # 快扫调优
    'WIDAN': 0.002,       # 快扫调优（PHM2009 用；PU 保持原结果 1e-3）
    'MRA_SNN': 0.005,     # SNN 替代梯度需要较大 lr（原版 0.01）
    # origeight 家族：与原 utils/train.py 一样共用单一 lr；沿用 DANN 移植值 1e-3
    'AGG': 0.001, 'MMD': 0.001, 'CORAL': 0.001, 'Triplet_loss': 0.001,
    'Mixup': 0.001, 'MLDG': 0.001, 'DAEL': 0.001,
}

TASKS = {
    'PHM2009': {
        'T0': [[4, 8, 12], [0]],
        'T1': [[0, 8, 12], [4]],
        'T2': [[0, 4, 12], [8]],
        'T3': [[0, 4, 8], [12]],
    },
    'PU': {
        'T0': [[1, 2, 3], [0]],
        'T1': [[0, 2, 3], [1]],
        'T2': [[0, 1, 3], [2]],
        'T3': [[0, 1, 2], [3]],
    },
    'UO': {
        # 4 个转速域: 0=Low(33-57Hz) 1=MidLow(57-66Hz) 2=MidHigh(66-76Hz) 3=High(76-102Hz)
        # 见 datasets/UO_bearing.py 的分箱说明
        'T0': [[1, 2, 3], [0]],
        'T1': [[0, 2, 3], [1]],
        'T2': [[0, 1, 3], [2]],
        'T3': [[0, 1, 2], [3]],
    },
    'HUST': {
        # 10 个恒定转速域: 0..9 = 20/25/30/35/40/60/65/70/75/80 Hz
        # LODO: 9 源域 -> 1 目标域, 见 datasets/HUST_bearing.py
        'T0': [[1, 2, 3, 4, 5, 6, 7, 8, 9], [0]],
        'T1': [[0, 2, 3, 4, 5, 6, 7, 8, 9], [1]],
        'T2': [[0, 1, 3, 4, 5, 6, 7, 8, 9], [2]],
        'T3': [[0, 1, 2, 4, 5, 6, 7, 8, 9], [3]],
        'T4': [[0, 1, 2, 3, 5, 6, 7, 8, 9], [4]],
        'T5': [[0, 1, 2, 3, 4, 6, 7, 8, 9], [5]],
        'T6': [[0, 1, 2, 3, 4, 5, 7, 8, 9], [6]],
        'T7': [[0, 1, 2, 3, 4, 5, 6, 8, 9], [7]],
        'T8': [[0, 1, 2, 3, 4, 5, 6, 7, 9], [8]],
        'T9': [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9]],
    },
}

# HUST 每个(域,类)的源域训练 / 目标域测试样本数
HUST_NUM_TRAIN = 50
HUST_NUM_TEST = 50


SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 64
DATA_LENGTH = 1024
EPOCHS = 60


# 可复现性开关（P-统一协议的前提）
#   cuDNN 默认按启发式挑卷积算法（含非确定性实现 + 原子累加），同一
#   (method, task, seed) 两次运行会得到不同结果 —— 对"跨方法公平对比"是致命的。
#   置 True 后固定 cudnn.deterministic / 关闭 benchmark，使每个单元可复现。
DETERMINISTIC = True


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        # 须在 CUDA 上下文建立前生效；首次调用前设好即可
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResampleDataset(Dataset):
    """把 DG-for-RMFD 的 (1,1024) 样本重采样到 (1,2560)。"""
    def __init__(self, ds, length=DATA_LENGTH):
        self.ds = ds
        self.length = length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        seq, label = self.ds[i]
        x = torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0)
        if x.shape[-1] != self.length:
            x = F.interpolate(x, size=self.length, mode='linear', align_corners=False)
        return x[0], label


def load_data(data_name, task):
    """返回 (srcs, tar, num_cls)，srcs 为源域数据集列表（可变数量）。"""
    if data_name == 'PHM2009':
        src1, src2, src3, tar, num_cls = PHM2009_DG(
            task, 'mean-std', 200, 50).train_test_data_split()
        return [src1, src2, src3], tar, num_cls
    elif data_name == 'UO':
        src1, src2, src3, tar, num_cls = UO_DG(
            task, 'mean-std', 200, 50).train_test_data_split()
        return [src1, src2, src3], tar, num_cls
    elif data_name == 'HUST':
        return HUST_DG(task, 'mean-std', HUST_NUM_TRAIN,
                       HUST_NUM_TEST).train_test_data_split()
    else:
        src1, src2, src3, tar, num_cls = PU_DG(
            task, 'mean-std', 200, 50).train_test_data_split()
        return [src1, src2, src3], tar, num_cls


def make_config(mod, method, num_cls, steps, checkpoint_freq, device, n_src=3):
    cfg = mod.configs
    cfg.dataset_type = 'bearing'
    cfg.device = device
    cfg.use_cuda = (device == 'cuda')
    cfg.num_classes = num_cls
    cfg.batch_size = BATCH_SIZE
    # 可选：steps_override 优先（批次 B 训练预算实验用）
    so = getattr(cfg, 'steps_override', None)
    cfg.steps = int(so) if so else steps
    cfg.checkpoint_freq = checkpoint_freq
    cfg.data_length = DATA_LENGTH
    cfg.dim_feature = 448
    # 源域数量必须来自**真实划分**。9 个移植基线（ADN/CCN/CDDG/CNNC/DGNIS/
    # IEDGNet/WhiteningNet）内部用 `len(configs.datasets_src)` 当 num_domains，
    # 而域判别类损失要求 标签数 == logits 行数 == n_src * batch_size。
    # 这里原来硬编码 ['s1','s2','s3']：在 3 源域的 PHM2009/PU 上恰好正确，
    # 但在 HUST(9 源域) 上会让标签只有 3*batch 而行数有 9*batch →
    # ADN 实测 `Expected input batch_size (576) to match target batch_size (192)`
    # 直接崩溃（CCN/CDDG/DGNIS 同因）。故改为按 n_src 生成。
    cfg.datasets_src = ['s%d' % (i + 1) for i in range(max(1, int(n_src)))]
    cfg.datasets_tgt = ['t']
    cfg.schedule = [EPOCHS // 3 * checkpoint_freq, 2 * EPOCHS // 3 * checkpoint_freq]
    cfg.cos = False
    # 调度模式必须显式重置：cfg 是 mod.configs 这个**模块级单例**，
    # 上一个个体的 sched_mode 会泄漏进来（曾导致 run_compare 的 pub 臂
    # 被 unif 臂污染成 'step'）。置空 = 由方法自行按 use_scheduler 推断。
    cfg.sched_mode = ''
    cfg.use_domain_weight = False
    cfg.lr = LR_OVERRIDE.get(method, 0.001)
    return cfg


def run_one(method, data_name, task_name, seed, device, logfile):
    setup_seed(seed)
    task = TASKS[data_name][task_name]
    srcs, tar, num_cls = load_data(data_name, task)
    srcs = [ResampleDataset(s) for s in srcs]
    tar = ResampleDataset(tar)

    iters_per_epoch = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
    steps = EPOCHS * iters_per_epoch
    checkpoint_freq = iters_per_epoch

    mod = importlib.import_module(METHOD_MODULES.get(method, method))
    cfg = make_config(mod, method, num_cls, steps, checkpoint_freq, device,
                      n_src=len(srcs))
    cfg.n_domains = len(srcs)          # 供域判别器等使用（HUST 为 9）
    model_cls = getattr(mod, METHOD_MODELS[method])
    model = model_cls(cfg)

    logger = logging.getLogger('{}_{}_seed{}'.format(method, data_name, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    train_loaders = [InfiniteDataLoader(s, BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_iter = zip(*train_loaders)

    result = model.train_model(train_iter, [test_loader], logger)
    acces = np.array(result['acces'])
    best = float(acces[:, 0].max()) if acces.size else float('nan')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='PHM2009',
                        choices=['PHM2009', 'PU', 'UO', 'HUST'])
    parser.add_argument('--datasets', type=str, default='PHM2009,PU',
                         help='--all 模式下要跑的数据集列表,逗号分隔,如 UO 或 PHM2009,PU,UO')
    parser.add_argument('--task', type=str, default='T0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from agmamba.paths import RESULTS as outdir
    os.makedirs(outdir, exist_ok=True)

    if args.all:
        data_names_all = [d.strip() for d in args.datasets.split(',') if d.strip()]
        summary = {}
        t0 = time.time()
        for data_name in data_names_all:
            for task in TASKS[data_name]:
                for method in METHOD_MODELS:
                    accs = []
                    for seed in SEEDS:
                        print('[{}] RUN {} / {} / {} / seed{} ...'.format(
                            time.strftime('%H:%M:%S'), data_name, task, method, seed), flush=True)
                        from agmamba.paths import log_path as _lp
                        logfile = _lp(data_name, task, method, seed)
                        try:
                            acc = run_one(method, data_name, task, seed, device, logfile)
                            accs.append(acc * 100.0)
                            print('[{}]   seed{} -> {:.2f}'.format(
                                time.strftime('%H:%M:%S'), seed, acc * 100.0), flush=True)
                        except Exception as e:
                            accs.append(float('nan'))
                            print('[{}]   seed{} -> FAIL ({})'.format(
                                time.strftime('%H:%M:%S'), seed, e), flush=True)
                    accs = np.array(accs)
                    summary[(data_name, task, method)] = accs
                    mean = np.nanmean(accs) if not np.all(np.isnan(accs)) else float('nan')
                    std = np.nanstd(accs) if not np.all(np.isnan(accs)) else float('nan')
                    print('[{}] DONE {} {} {} -> mean {:.2f} +- {:.2f} (elapsed {:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), data_name, task, method, mean, std,
                        time.time() - t0), flush=True)

        # 输出汇总
        for data_name in data_names_all:
            print('\n================ {} SUMMARY (mean+-std %) ================'.format(data_name))
            print('task\t' + '\t'.join(METHOD_MODELS.keys()))
            for task in ['T0', 'T1', 'T2', 'T3']:
                vals = []
                for method in METHOD_MODELS:
                    a = summary.get((data_name, task, method))
                    if a is None or np.all(np.isnan(a)):
                        vals.append('FAIL')
                    else:
                        vals.append('{:.2f}$\\pm${:.2f}'.format(np.nanmean(a), np.nanstd(a)))
                print(task + '\t' + '\t'.join(vals))

            with open(os.path.join(outdir, 'summary_{}.csv'.format(data_name)), 'w') as f:
                f.write('task,method,mean,std,' + ','.join('seed{}'.format(s) for s in SEEDS) + '\n')
                for task in ['T0', 'T1', 'T2', 'T3']:
                    for method in METHOD_MODELS:
                        a = summary.get((data_name, task, method))
                        if a is None or np.all(np.isnan(a)):
                            f.write('{},{},FAIL,FAIL,{}\n'.format(task, method, ','.join([''] * len(SEEDS))))
                        else:
                            f.write('{},{},{:.4f},{:.4f},{}\n'.format(
                                task, method, np.nanmean(a), np.nanstd(a),
                                ','.join('{:.2f}'.format(v) for v in a)))
        print('\nDone. Total elapsed {:.0f}s'.format(time.time() - t0))
    else:
        from agmamba.paths import log_path as _lp
        logfile = _lp(args.dataset, args.task, args.method, args.seed)
        acc = run_one(args.method, args.dataset, args.task, args.seed, device, logfile)
        print('{} / {} / {} / seed{} -> Best tar-Acc: {:.2f}%'.format(
            args.dataset, args.task, args.method, args.seed, acc * 100.0))


if __name__ == '__main__':
    main()
