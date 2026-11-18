#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一协议的对比运行器（所有方法共用同一套数据/预算/评估网格）。

冻结的"公平性"四要素
--------------------
1. **数据**：同一 task 定义、同一划分（`run_cell.TASKS` / `load_data`）、同一重采样长度。
2. **预算**：60 epoch，`steps = 60 * iters_per_epoch`；batch 64；每 epoch 评估一次
   （`checkpoint_freq = iters_per_epoch`）-> 恰好 60 个评估点，供"后 10 轮均值"读数。
3. **种子**：{0,1,2,3,4}，同种子跨方法严格配对。
4. **读数**：末十个检查点均值（主口径），peak 仅作 secondary。

调度臂（`--arms`）
-----------------
* ``unif`` **统一协议**：所有方法共用基准框架的阶梯衰减语义
  （每到一个 milestone 把 lr 乘 0.5；milestones = 预算的 1/3、2/3）。
  对 AG-Mamba 通过 ``sched_mode='step'`` 生效；对基线本就是其框架默认行为。
* ``pub``  **发布配置**：不改调度，各方法用自己的默认
  （AG-Mamba = 余弦退火至零；基线 = 阶梯衰减，与 unif 重合）。

  -> 因此只有 ``ARMS_MAY_DIFFER`` 里的方法需要跑两个臂；其余方法两臂等价，
     只跑 ``unif``（避免翻倍成本）。

用法
----
    # 查看计划（不运行）
    python experiments/run_compare.py --dataset PHM2009 --plan
    # 运行 AG-Mamba 两个调度臂（4 任务 × 5 种子 × 2 臂 = 40 runs）
    python experiments/run_compare.py --dataset PHM2009 --methods AGMamba
    # 运行全部方法（基线只跑 unif）
    python experiments/run_compare.py --dataset PHM2009 --methods all
    # 强制重跑已有日志
    python experiments/run_compare.py --dataset PHM2009 --methods AGMamba --force
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
sys.path.insert(0, os.path.join(_ROOT, 'experiments'))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

import run_cell as R                                # noqa: E402
from agmamba import paths as _P                     # noqa: E402

# ---------------------------------------------------------------------------
# 协议常量（冻结，改动需记入 docs/COMPARE_PROTOCOL.md）
# ---------------------------------------------------------------------------
EPOCHS = R.EPOCHS                    # 60
BATCH_SIZE = R.BATCH_SIZE            # 64
SEEDS = [0, 1, 2, 3, 4]

# 只有"发布调度 != 统一阶梯"的方法需要跑 pub 臂
ARMS_MAY_DIFFER = {'AGMamba'}
ARMS = ('unif', 'pub')

# 每方法每臂的静态覆盖（与任务/种子无关的部分在运行时补齐）
#   注意：`cfg` 是模块级单例，臂之间会互相污染，因此两臂都**显式**写出调度，
#   不依赖任何"默认值"。pub 臂只有 ARMS_MAY_DIFFER 里的方法会跑（= AGMamba），
#   其发布配置是余弦退火至零（agmamba/configs.yaml: use_scheduler=True）。
ARM_STATIC = {
    'unif': {'cos': False, 'sched_mode': 'step'},
    'pub': {'sched_mode': 'cos'},
}


def build_overrides(arm, steps, method):
    """返回该臂要写入 cfg 的覆盖项。"""
    ov = dict(ARM_STATIC.get(arm, {}))
    if arm == 'unif':
        # 基准框架语义：milestones = 预算的 1/3、2/3，每档乘 0.5
        ov['schedule'] = [steps // 3, 2 * steps // 3]
        ov['sched_mul'] = 0.5
    if arm == 'pub' and method not in ARMS_MAY_DIFFER:
        return None                      # 与 unif 等价，不必重复跑
    return ov


def parse_accs(logfile):
    """从日志解析 [(step, acc%)]；返回 None 表示日志无效。

    ⚠ 必须兼容两种目标域精度标签（仓库历史遗留，见 docs/COMPARE_PROTOCOL.md §5）：
      · `tgt_acc:`       —— 本方法 / DANN / MRA_SNN / UDTL / WIDAN
      · `tgt_train_acc:` —— 移植方法（ADN/CCDG/CCN/CDDG/CNNC/DDGFD/DGNIS/
                            IEDGNet/WhiteningNet）；名字里有 train，但记录的
                            就是 target 精度（目标 loader 是 loaders[0]）
    只认 `tgt_acc` 会让 9 个基线解析为空、末-k 均值变 NaN。
    另：数值 >1.5 视为已是百分数（防 `tar-Acc` 类旧格式的尺度差异）。
    """
    import re
    pat_step = re.compile(r'=+\s*Step\s+(\d+)\s*=+')
    pat_acc = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
    cur, out = 0, []
    if not os.path.exists(logfile) or os.path.getsize(logfile) < 200:
        return None
    with open(logfile, errors='ignore') as f:
        for line in f:
            m = pat_step.search(line)
            if m:
                cur = int(m.group(1))
                continue
            m = pat_acc.search(line)
            if m:
                v = float(m.group(1))
                out.append((cur, v * 100.0 if v <= 1.5 else v))
    return out or None


def run_one(method, arm, data_name, task_name, seed, device, logfile):
    """跑一个 (method, arm, dataset, task, seed) 单元，返回完整 acc 轨迹。"""
    R.setup_seed(seed)
    task = R.TASKS[data_name][task_name]
    srcs, tar, num_cls = R.load_data(data_name, task)
    srcs = [R.ResampleDataset(s) for s in srcs]
    tar = R.ResampleDataset(tar)

    iters = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
    steps = EPOCHS * iters

    mod = importlib.import_module(R.METHOD_MODULES[method])
    cfg = R.make_config(mod, method, num_cls, steps, iters, device,
                        n_src=len(srcs))
    ov = build_overrides(arm, steps, method)
    for k, v in (ov or {}).items():
        setattr(cfg, k, v)
    cfg.n_domains = len(srcs)

    model_cls = getattr(mod, R.METHOD_MODELS[method])
    model = model_cls(cfg)

    logger = logging.getLogger('{}_{}_{}_{}'.format(
        method, arm, data_name, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.info('# method={} arm={} data={} task={} seed={} steps={} '
                'iters_per_epoch={} epochs={} batch={} lr={} schedule={} '
                'sched_mode={}'.format(
                    method, arm, data_name, task_name, seed, steps, iters,
                    EPOCHS, BATCH_SIZE, cfg.lr, getattr(cfg, 'schedule', None),
                    getattr(cfg, 'sched_mode', 'n/a')))

    train_loaders = [R.InfiniteDataLoader(s, BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)
    result = model.train_model(zip(*train_loaders), [test_loader], logger)
    for h in list(logger.handlers):
        h.close()
    logger.handlers = []
    return [a for _, a in parse_accs(logfile) or []]


def last5(accs):
    """主口径（§5.1）：末 5 个评估点均值。"""
    if accs is None or len(accs) < 5:
        return None
    return float(np.mean(accs[-5:]))


def last10(accs):
    if accs is None or len(accs) < 10:
        return None
    return float(np.mean(accs[-10:]))


def discover_units(dataset):
    """扫描 results/<dataset>/<method>/<arm>/<task>/*.log，返回 [(method, arm, task)]。

    用于让汇总**总是覆盖磁盘上已有的一切**：分批跑时（先本方法、后基线）
    若汇总只含本批的方法，后一批会覆盖掉前一批的表。
    """
    root = os.path.join(_P.RESULTS, dataset)
    out = []
    if not os.path.isdir(root):
        return out
    for method in sorted(os.listdir(root)):
        pm = os.path.join(root, method)
        if not os.path.isdir(pm):
            continue
        for arm in sorted(os.listdir(pm)):
            pa = os.path.join(pm, arm)
            if not os.path.isdir(pa):
                continue
            for task in sorted(os.listdir(pa)):
                pt = os.path.join(pa, task)
                if not os.path.isdir(pt):
                    continue
                if any(f.endswith('.log') for f in os.listdir(pt)):
                    out.append((method, arm, task))
    return out


def write_summary(dataset, methods=None, arms=None, tasks=None, seeds=None,
                  outdir=os.path.join(_ROOT, 'docs', 'manuscript')):
    """汇总为 CSV + Markdown：**last5 为主口径，last1/last10 并列，peak 留档**。

    口径依据 `docs/COMPARE_PROTOCOL.md` §5.1（主口径 last5；last1 为部署口径、
    last10 为稳健性口径，三者均进论文；peak 偏置随方法变化，**不进正文**）。
    本函数早期版本把 last10 写成"主口径"，与 §5.1 冲突，2026-10-01 已同步。

    `methods=None`（推荐）时**从磁盘发现**所有 (method, arm, task)，
    这样分批跑（先本方法、后基线）不会让后一批覆盖前一批的汇总。
    与 run_variants / 表 14 的历史口径（peak）显式区分开，避免混用。
    """
    import csv
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, '_cmp_%s.csv' % dataset)
    md_path = os.path.join(outdir, '_cmp_%s.md' % dataset)

    if seeds is None:
        seeds = list(SEEDS)
    if methods is None:
        units = discover_units(dataset)
    else:
        units = [(m, arm, t) for m in methods
                 for arm in (arms or ARMS) for t in (tasks or [])]
    # 与 unif 等价的臂不重复计入
    units = [(m, arm, t) for m, arm, t in units
             if not (arm == 'pub' and m not in ARMS_MAY_DIFFER)]

    def ms(vals):
        """跨种子的 mean / std(ddof=1)。"""
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    rows = []
    for m, arm, t in units:
        acc = {'last5': [], 'last1': [], 'last10': [], 'peak': []}
        for sd in seeds:
            tr = parse_accs(_P.log_path(dataset, t, m, sd, arm=arm))
            # 完整性闸门：半截日志（中断残留）不得计入读数
            if not tr or len(tr) < EPOCHS:
                continue
            v = [a for _, a in tr]
            acc['last5'].append(float(np.mean(v[-5:])))
            acc['last1'].append(float(v[-1]))
            acc['last10'].append(float(np.mean(v[-10:])))
            acc['peak'].append(float(np.max(v)))
        if not acc['last5']:
            continue
        row = {'method': m, 'arm': arm, 'task': t, 'n': len(acc['last5'])}
        for key in ('last5', 'last1', 'last10', 'peak'):
            mu, sd_ = ms(acc[key])
            row['%s_mean' % key] = mu
            row['%s_std' % key] = sd_
        rows.append(row)
    if not rows:
        print('[summary] 无可汇总的日志', flush=True)
        return

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L = ['# %s 统一协议对比（**last5 主口径** / last1 · last10 并列 / peak 留档）'
         % dataset, '',
         '由 `experiments/run_compare.py` 自动生成。',
         '',
         '- 数据/划分：`run_cell.TASKS` 同一 task 定义与划分',
         '- 预算：%d epoch，batch %d，`steps = %d * iters_per_epoch`'
         % (EPOCHS, BATCH_SIZE, EPOCHS),
         '- 评估网格：每 epoch 一次 -> 恒 %d 个评估点（满 %d 点才计入读数）'
         % (EPOCHS, EPOCHS),
         '- 种子：%s（同种子跨方法严格配对）' % seeds,
         '- **主读数 = 末 5 个评估点均值**（§5.1）；last1 = 部署口径，'
         'last10 = 稳健性口径；**peak 不进正文**，仅此留档', '']
    for arm in sorted({r['arm'] for r in rows}):      # 只列实际存在的臂
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        L += ['## arm = `%s`' % arm, '',
              '| 方法 | 任务 | n | **last5 (主)** | last1 (部署) '
              '| last10 (稳健) | peak (留档) |',
              '|---|---|---|---|---|---|---|']
        for r in sub:
            L.append('| %s | %s | %d | **%.2f ± %.2f** | %.2f | %.2f |'
                     ' %.2f |'
                     % (r['method'], r['task'], r['n'], r['last5_mean'],
                        r['last5_std'], r['last1_mean'], r['last10_mean'],
                        r['peak_mean']))
        # 跨任务平均（每方法一行）
        L += ['', '### 跨任务平均', '',
              '| 方法 | n | **last5 (主)** | last1 (部署) '
              '| last10 (稳健) | peak (留档) |', '|---|---|---|---|---|---|']
        mset = []
        for r in sub:
            if r['method'] not in mset:
                mset.append(r['method'])
        for m in mset:
            rs = [r for r in sub if r['method'] == m]
            L.append('| %s | %d | **%.2f** | %.2f | %.2f | %.2f |'
                     % (m, sum(r['n'] for r in rs),
                        np.mean([r['last5_mean'] for r in rs]),
                        np.mean([r['last1_mean'] for r in rs]),
                        np.mean([r['last10_mean'] for r in rs]),
                        np.mean([r['peak_mean'] for r in rs])))
        L.append('')
    with open(md_path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n[summary] %s\n[summary] %s' % (csv_path, md_path), flush=True)


def report_progress(dataset, outfile=None):
    """进度报告：逐方法跨任务 **last5 主口径**均值 + 排行榜，并落盘。

    每完成一个算法调用一次，便于随时比较与观察进度。
    数据来源同样是 `discover_units`（磁盘上已有的一切），因此
    报告始终是"到目前为止的完整榜单"，不会只含刚跑完的方法。
    """
    units = [(m, arm, t) for m, arm, t in discover_units(dataset)
             if not (arm == 'pub' and m not in ARMS_MAY_DIFFER)]
    per = {}
    for m, arm, t in units:
        vals = []
        for sd in SEEDS:
            tr = parse_accs(_P.log_path(dataset, t, m, sd, arm=arm))
            if tr and len(tr) >= EPOCHS:
                vals.append(float(np.mean([a for _, a in tr][-5:])))
        if vals:
            per.setdefault((m, arm), {})[t] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                len(vals))

    board = []
    for (m, arm), td in per.items():
        score = float(np.mean([v[0] for v in td.values()]))
        board.append((score, m, arm, td, sum(v[2] for v in td.values())))
    board.sort(reverse=True)

    L = ['===== 进度报告 @ %s  (dataset=%s, 已有结果的方法 %d 个) ====='
         % (time.strftime('%F %T'), dataset, len(board))]
    for score, m, arm, td, n in board:
        tasks_str = '  '.join('%s %.2f±%.2f' % (t, td[t][0], td[t][1])
                              for t in sorted(td))
        L.append('  %-14s %-5s cross=%.2f (n=%d)  %s'
                 % (m, arm, score, n, tasks_str))
    text = '\n'.join(L)
    print('\n' + text, flush=True)
    if outfile:
        with open(outfile, 'w') as f:
            f.write(text + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PHM2009')
    ap.add_argument('--methods', default='AGMamba',
                    help="'all' 或逗号分隔，如 AGMamba,WIDAN")
    ap.add_argument('--tasks', default=None, help='默认该数据集全部任务')
    ap.add_argument('--seeds', default=None, help='默认 0,1,2,3,4')
    ap.add_argument('--arms', default=None, help='默认 unif,pub（按需裁剪）')
    ap.add_argument('--force', action='store_true', help='重跑已有日志')
    ap.add_argument('--plan', action='store_true', help='只打印计划不运行')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = ([t.strip() for t in a.tasks.split(',')] if a.tasks
             else list(R.TASKS[a.dataset].keys()))
    seeds = ([int(s) for s in a.seeds.split(',')] if a.seeds else list(SEEDS))
    if a.methods == 'all':
        methods = list(R.METHOD_MODELS.keys())
    else:
        methods = [m.strip() for m in a.methods.split(',')]
    arms = ([x.strip() for x in a.arms.split(',')] if a.arms
            else list(ARMS))

    if a.plan:
        # 计算 steps（用 T0 的代表规模）
        srcs, tar, nc = R.load_data(a.dataset, R.TASKS[a.dataset][tasks[0]])
        iters = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
        steps = EPOCHS * iters
        print('dataset   : %s  tasks=%s seeds=%s' % (a.dataset, tasks, seeds))
        print('budget    : %d epoch x %d iters = %d steps, batch=%d, '
              '%d checkpoints' % (EPOCHS, iters, steps, BATCH_SIZE, EPOCHS))
        print('methods   : %s' % methods)
        print('arms      : %s  (仅 %s 两臂有差异)'
              % (arms, sorted(ARMS_MAY_DIFFER)))
        n_eff = 0
        for m in methods:
            for arm in arms:
                if build_overrides(arm, steps, m) is None:
                    continue
                n_eff += len(tasks) * len(seeds)
        print('本次单元数: %d' % n_eff)
        return

    print('### dataset=%s  methods=%s  arms=%s  tasks=%s  seeds=%s'
          % (a.dataset, methods, arms, tasks, seeds), flush=True)
    t_start = time.time()
    done = skipped = failed = na = 0

    steps_cache = {}

    def steps_for(task_name):
        """该任务的 steps（= EPOCHS * iters_per_epoch），带缓存。"""
        if task_name not in steps_cache:
            srcs, _, _ = R.load_data(a.dataset, R.TASKS[a.dataset][task_name])
            it = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
            steps_cache[task_name] = EPOCHS * it
        return steps_cache[task_name]

    progress_path = os.path.join(_ROOT, 'logs', 'progress_%s.txt' % a.dataset)
    report_progress(a.dataset, progress_path)      # 起始快照（含已完成的方法）

    for m in methods:
        t_m = time.time()
        for arm in arms:
            # 与 unif 等价的臂不重复跑：None 只取决于 (arm, method)，
            # 与 steps 无关（只有 ARMS_MAY_DIFFER 里的方法两臂有差异）。
            # 没有这道闸，13 个基线会各白跑一遍 pub 臂。
            if build_overrides(arm, steps_for(tasks[0]), m) is None:
                na += 1
                print('[n/a ] %s arm=%s 与 unif 等价，跳过' % (m, arm),
                      flush=True)
                continue
            for t in tasks:
                for sd in seeds:
                    logfile = _P.log_path(a.dataset, t, m, sd, arm=arm)
                    if not a.force:
                        tr = parse_accs(logfile)
                        # 完整性闸门：协议要求每个单元恒 EPOCHS 个评估点。
                        # 断电/被杀进程会留下"只有十几个点"的半截日志；若仅按
                        # `>= 10` 判定，它会冒充"已完成"被永久跳过，且末-k 读数
                        # 会取自错误的检查点区间（应为第 EPOCHS-k+1..EPOCHS 轮）。
                        if tr and len(tr) >= EPOCHS:
                            skipped += 1
                            print('[skip] %s -> last5 %.2f'
                                  % (os.path.basename(logfile),
                                     last5([v for _, v in tr])),
                                  flush=True)
                            continue
                        if tr:
                            print('[redo] %s 仅 %d/%d 个评估点（中断残留），重跑'
                                  % (os.path.basename(logfile), len(tr),
                                     EPOCHS), flush=True)
                    t0 = time.time()
                    try:
                        accs = run_one(m, arm, a.dataset, t, sd, device,
                                       logfile)
                        done += 1
                        print('[%s] %-46s last5=%.2f peak=%.2f (%.0fs)'
                              % (time.strftime('%H:%M:%S'),
                                 os.path.basename(logfile),
                                 last5(accs) or float('nan'),
                                 max(accs) if accs else float('nan'),
                                 time.time() - t0), flush=True)
                    except Exception as e:
                        failed += 1
                        print('[%s] FAIL %s : %s'
                              % (time.strftime('%H:%M:%S'),
                                 os.path.basename(logfile),
                                 str(e)[:160]), flush=True)
        # ---- 该方法全部单元跑完 -> 立刻出一次报告，便于比较与看进度 ----
        print('\n[method done] %s  用时 %.1f min'
              % (m, (time.time() - t_m) / 60.0), flush=True)
        report_progress(a.dataset, progress_path)
    print('\n### 完成 done=%d skipped=%d failed=%d 跳过等价臂=%d 耗时 %.1f min'
          % (done, skipped, failed, na, (time.time() - t_start) / 60.0))
    # 汇总从磁盘发现全部已有单元（分批跑时不会覆盖前一批的表）
    write_summary(a.dataset)


if __name__ == '__main__':
    main()
