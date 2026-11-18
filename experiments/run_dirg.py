#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG 附加试验：**域 = 不同转速，载荷固定** 的 LODO 跨域对比。

⚠ 这是**附加试验**，不参与论文（论文固定三基准：PHM2009 / PU / HUST）。
   一切产物都写在 `results_dirg/` 与 `logs/dirg_*`，**绝不进 `docs/manuscript/`**。

协议（刻意与论文 P-统一协议保持一致，便于横向参照）：
  - 源域训练样本 = 每域每类 50 条 1024 点窗（取每类记录**前** 50 窗）
  - 目标域测试样本 = 每类记录**末尾** 50 窗（与源域不重叠）
  - steps = 60 * ceil(N_src / 64)，checkpoint_freq = ceil(N_src / 64)
        => 每单元 60 个评估点，主口径取末 5 点（last5）
  - batch 64、Adam lr 1e-3 / wd 1e-4、5 种子（默认 0..2 以省时间）
  - AG-Mamba 强制 `AGM_FORCE_PY_SCAN=1`（纯 PyTorch 分块扫描 → 逐位可复现）

任务表：对某个载荷档，域 = 该档实际存在的转速（L1000 有 5 个域 → 5 个 LODO 任务；
        L1400 4 个域 → 4 个任务；L1800 3 个域 → 3 个任务）。载荷在任务内**恒定**。

用法：
  # 冒烟（1 任务 1 种子 12 步）
  python experiments/run_dirg.py --load 1000 --methods AGMamba --tasks T0 --seeds 0 --steps 12
  # 正式（5 任务 × 3 种子，全部 21 方法太慢，建议先用代表性子集）
  python experiments/run_dirg.py --load 1000 \
      --methods AGMamba,AGG,MMD,CORAL,DANN,WIDAN,MRA_SNN --seeds 0,1,2
  # 只看已有结果
  python experiments/run_dirg.py --load 1000 --report
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

# AG-Mamba 变体臂必须走纯 PyTorch 扫描（官方 CUDA 内核反传原子累加 -> 不可复现）
os.environ.setdefault('AGM_FORCE_PY_SCAN', '1')

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import experiments.run_cell as RC                               # noqa: E402
from datasets.DIRG_bearing import (                             # noqa: E402
    DIRG_DG, DOMAINS, SPEEDS, CLASSES, CLASS_DESC, num_cls,
    available_domains)
from utils.DatasetClass import InfiniteDataLoader                # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

OUTROOT = os.path.join(_ROOT, 'results_dirg')
SEED_DEFAULT = [0, 1, 2]


def build_tasks(load, ch=1, max_src=None, fs=51200):
    """按该载荷档实际存在的转速，生成 LODO 任务表（源域 = 其余转速）。

    max_src: 若给定，则只用**最靠近目标**的 max_src 个源域（用于"少源域"变体）。
    返回 {task_name: [src_ids, tgt_id]}, [域索引顺序]
    """
    ids, sps = available_domains(load, ch=ch, fs=fs)
    if len(ids) < 2:
        raise SystemExit('载荷档 L%d 的域太少（%s），无法做 LODO' % (load, sps))
    tasks = {}
    for k, tgt in enumerate(ids):
        srcs = [d for d in ids if d != tgt]
        if max_src:
            srcs = sorted(srcs, key=lambda d: abs(d - tgt))[:max_src]
            srcs = sorted(srcs)
        tasks['T%d' % k] = [srcs, [tgt]]
    return tasks, ids


def runroot(load, ch, fs=51200, sig=1024, tag_extra='', lp=0):
    """结果根目录：**通道写进路径**，避免不同通道的结果混进同一张汇总表。"""
    tag = 'L%d_ch%d' % (load, ch)
    if int(fs) != 51200:
        tag += '_fs%d' % int(fs)
    if int(sig) != 1024:
        tag += '_sig%d' % int(sig)
    if int(lp):
        tag += '_lp%d' % int(lp)
    if tag_extra:
        tag += '_%s' % tag_extra
    return os.path.join(OUTROOT, tag)


def logdir(load, ch, method, task, fs=51200, sig=1024, tag_extra='', lp=0):
    return os.path.join(runroot(load, ch, fs, sig, tag_extra, lp),
                        method, task)


def _confusion(model, ds, device, n_cls):
    # target-domain confusion matrix (rows=true, cols=pred); method-agnostic; only under --cm
    from torch.utils.data import DataLoader as _DL
    fwd = None
    for cand in (model, getattr(model, 'model', None)):
        if cand is None:
            continue
        try:
            cand.eval()
            xb, _ = next(iter(_DL(ds, batch_size=8)))
            with torch.no_grad():
                cand(xb.to(device))
            fwd = cand
            break
        except Exception:
            continue
    if fwd is None:
        return None
    C = np.zeros((n_cls, n_cls), dtype=int)
    with torch.no_grad():
        for xb, yb in _DL(ds, batch_size=RC.BATCH_SIZE):
            out = fwd(xb.to(device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            for t, p in zip(yb.numpy(), out.argmax(1).cpu().numpy()):
                C[int(t), int(p)] += 1
    return C


def run_one(method, task_name, task, load, seed, device, logfile, steps_override,
            ch, fs=51200, want_cm=False, sig=1024, mods=None, lp=0):
    """与 run_cell.run_one 同构；差异：① 用 DIRG 加载器 ② 返回完整精度序列。

    ⚠ 若 run_cell.run_one 将来改了训练循环，这里要同步（本文件刻意不改 run_cell，
       以免影响论文批次的可复现性）。
    """
    RC.setup_seed(seed)
    srcs_raw, tar_raw, n_cls = DIRG_DG(
        task, load=load, norm_type='mean-std',
        num_train_samples=50, num_test_samples=50, ch=ch,
        fs=fs, sig=sig, lp_hz=lp).train_test_data_split()
    srcs = [RC.ResampleDataset(s, length=int(sig)) for s in srcs_raw]
    tar = RC.ResampleDataset(tar_raw, length=int(sig))

    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = steps_override or (RC.EPOCHS * iters)
    checkpoint_freq = iters

    mod = importlib.import_module(RC.METHOD_MODULES.get(method, method))
    cfg = RC.make_config(mod, method, n_cls, steps, checkpoint_freq, device,
                         n_src=len(srcs))
    cfg.n_domains = len(srcs)
    # 窗长必须跟着数据走。AGMamba 前端把 1024/2048 下采样 8 倍得到
    # L=128/256，而 pos_embed 的 num_patches 默认 256 —— 2048 窗正好用满，
    # 不会越界（sig>2048 会越界，需同步调大 num_patches）。
    cfg.data_length = int(sig)
    # --mods k=v,k=v：逐单元覆盖 cfg 字段（如 alias=3.0）。
    # 注意 cfg 是 mod.configs 这个**模块级单例**，所以同一进程内的覆盖会
    # 一直生效 —— 同一次调用里所有单元拿到同一个值，正是我们要的。
    for kv in (mods or '').split(','):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition('=')
        k, v = k.strip(), v.strip()
        try:
            vv = int(v)
        except ValueError:
            try:
                vv = float(v)
            except ValueError:
                vv = {'true': True, 'false': False}.get(v.lower(), v)
        setattr(cfg, k, vv)
        print('[mod] cfg.%s = %r' % (k, vv), flush=True)
    print('[cfg] method=%s num_classes=%d steps=%d ckpt_freq=%d n_src=%d '
          'batch=%d lr=%g | fs=%d sig=%d lp=%d ch=%d | w_mmd=%g w_adv=%g (no alignment)'
          % (method, cfg.num_classes, cfg.steps, cfg.checkpoint_freq,
             len(srcs), cfg.batch_size, cfg.lr, int(fs), int(sig), int(lp), ch + 1,
             getattr(cfg, 'w_mmd', float('nan')),
             getattr(cfg, 'w_adv', float('nan'))), flush=True)
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
    train_iter = zip(*train_loaders)

    res = model.train_model(train_iter, [test_loader], logger)
    acc = np.asarray(res['acces'], dtype=float)
    acc = acc[:, 0] * 100.0 if acc.size else np.array([])
    if want_cm:
        C = _confusion(model, tar, device, n_cls)
        if C is not None:
            np.savetxt(logfile + '_cm.csv', C, fmt='%d', delimiter=',')
            print('[cm] ' + task_name + ' diag=' + str(np.diag(C).tolist()), flush=True)
    return acc


def summarize(acc):
    if acc.size == 0:
        return dict(n=0, last1=float('nan'), last5=float('nan'),
                    last10=float('nan'), peak=float('nan'))
    return dict(n=int(acc.size), last1=float(acc[-1]),
                last5=float(acc[-5:].mean()), last10=float(acc[-10:].mean()),
                peak=float(acc.max()))


def report(load, ch, methods, tasks, seeds, full_only=True, fs=51200,
           sig=1024, tag_extra='', lp=0):
    csvp = os.path.join(runroot(load, ch, fs, sig, tag_extra, lp),
                        '_summary.csv')
    if not os.path.exists(csvp):
        print('无汇总文件:', csvp)
        return
    import csv as _csv
    allrows = list(_csv.DictReader(open(csvp, encoding='utf-8')))
    rows = [r for r in allrows if r.get('load') == str(load)]
    dropped = 0
    if full_only:
        keep = [r for r in rows if int(r.get('n_eval') or 0) >= 60]
        dropped = len(rows) - len(keep)
        rows = keep
    print('\n%s\nDIRG L%d N / Ch%d -- 汇总（主口径 last5；池化 = 任务×种子均值）\n%s'
          % ('=' * 78, load, ch + 1, '=' * 78))
    if dropped:
        print('（已排除 %d 个非完整协议单元，如 n_eval<60 的冒烟行）' % dropped)
    print('%-14s %7s %7s %7s %7s %7s %6s' % ('method', 'last5', 'pool5m', 'last1',
                                             'last10', 'peak', 'n'))
    for m in methods:
        sel = [r for r in rows if r['method'] == m]
        if not sel:
            continue
        l5 = np.array([float(r['last5']) for r in sel])
        print('%-14s %7.2f %7.2f %7.2f %7.2f %7.2f %6d'
              % (m, l5.mean(), l5.std(), np.mean([float(r['last1']) for r in sel]),
                 np.mean([float(r['last10']) for r in sel]),
                 np.mean([float(r['peak']) for r in sel]), len(sel)))
    lodo = [t for t in tasks if not t.startswith('D')]
    ind = [t for t in tasks if t.startswith('D')]
    for grp, name in ((lodo, '跨转速 LODO'), (ind, '同域参考 D*')):
        if not grp:
            continue
        print('\n逐任务（%s，last5，跨种子均值）：' % name)
        print('%-14s %s' % ('method', ' '.join('%8s' % t for t in grp)))
        for m in methods:
            cells = []
            for t in grp:
                sel = [float(r['last5']) for r in rows
                       if r['method'] == m and r['task'] == t]
                cells.append('%8s' % ('%.2f' % np.mean(sel) if sel else '-'))
            print('%-14s %s' % (m, ' '.join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', type=int, default=1000, choices=[0, 1000, 1400, 1800])
    ap.add_argument('--methods', default='AGMamba')
    ap.add_argument('--tasks', default=None, help='默认全部 LODO 任务')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEED_DEFAULT))
    ap.add_argument('--lp', type=int, default=0,
                    help='low-pass cutoff in Hz applied at the SAME fs '
                         '(0 = off). 51200 + lp12800 = the missing 2x2 cell')
    ap.add_argument('--tag', default='',
                    help='extra suffix for the result dir (e.g. a30 for alpha sweep)')
    ap.add_argument('--sig', type=int, default=1024,
                    help='window points: 1024 native (20 ms @51200); '
                         '2048 = merge 2 adjacent windows -> 40 ms at NATIVE fs')
    ap.add_argument('--fs', type=int, default=51200,
                    help='target fs Hz (51200 native; 25600 = align with HUST -> 1024 pts = 40 ms)')
    ap.add_argument('--cm', action='store_true',
                    help='dump target-domain confusion matrix per cell (*_cm.csv)')
    ap.add_argument('--ch', type=int, default=1, help='通道 0..5，默认 1 = Radial y')
    ap.add_argument('--steps', type=int, default=None, help='覆盖训练步数（冒烟用）')
    ap.add_argument('--max-src', type=int, default=None, help='源域个数上限（少源域变体）')
    ap.add_argument('--indomain', action='store_true',
                    help='额外跑同域参考臂（D0..Dk：源域=目标域，仅作"该任务可学性"基准）')
    ap.add_argument('--mods', default=None, help='额外变体（逗号分隔 arm=kwargs，暂仅 AGMamba）')
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume', action='store_true',
                    help='跳过汇总表里已有（且 steps 一致）的单元')
    ap.add_argument('--report', action='store_true', help='只打印已有汇总')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    tasks_all, ids = build_tasks(args.load, ch=args.ch, max_src=args.max_src,
                                 fs=args.fs)
    if args.indomain:
        # 同域参考臂：源域 = 目标域（只有 1 个源域）
        for k, d in enumerate(ids):
            tasks_all['D%d' % k] = [[d], [d]]
    if args.tasks:
        want = [t.strip() for t in args.tasks.split(',') if t.strip()]
        for t in want:
            if t not in tasks_all:
                raise SystemExit('任务 %s 不存在（本载荷档可用：%s）'
                                 % (t, sorted(tasks_all)))
        tasks = {t: tasks_all[t] for t in want}
    else:
        tasks = tasks_all

    if args.report:
        report(args.load, args.ch, methods, list(tasks), seeds, fs=args.fs,
               sig=args.sig, tag_extra=args.tag, lp=args.lp)
        return

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('DIRG 附加试验（域=转速，载荷固定）')
    print('  载荷档 L%d N ｜ 通道 Ch%d ｜ 类别 %d 个 %s' % (args.load, args.ch + 1,
                                                          num_cls, CLASSES))
    print('  可用域: %s' % ', '.join(DOMAINS[i] for i in ids))
    for t, (s, g) in tasks.items():
        print('  %-4s 源域 %s -> 目标域 %s'
              % (t, [DOMAINS[i] for i in s], DOMAINS[g[0]]))
    print('  方法: %s ｜ 种子: %s ｜ 设备: %s ｜ steps: %s'
          % (methods, seeds, device,
             args.steps if args.steps else '60*ceil(N_src/64)'))
    print('  输出: %s\n' % OUTROOT)

    sump = os.path.join(runroot(args.load, args.ch, args.fs, args.sig,
                                args.tag, args.lp), '_summary.csv')
    os.makedirs(os.path.dirname(sump), exist_ok=True)
    new_file = not os.path.exists(sump)
    fsum = open(sump, 'a', encoding='utf-8', newline='')
    if new_file:
        fsum.write('load,ch,method,task,seed,kind,steps,n_eval,last1,last5,last10,'
                   'peak,seconds,log\n')
    import csv as _csv
    try:
        recorded = {(r['load'], r['method'], r['task'], r['seed']): r
                    for r in _csv.DictReader(open(sump, encoding='utf-8'))}
    except Exception:  # noqa: BLE001
        recorded = {}

    for method in methods:
        for tname, (src_ids, tgt_ids) in tasks.items():
            for seed in seeds:
                d = logdir(args.load, args.ch, method, tname, args.fs,
                           args.sig, args.tag, args.lp)
                os.makedirs(d, exist_ok=True)
                log = os.path.join(d, 'DIRG_%s_%s_L%d_seed%d.log'
                                   % (tname, method, args.load, seed))
                if args.resume:
                    old = recorded.get((str(args.load), method, tname, str(seed)))
                    done = old is not None and int(old.get('n_eval') or 0) >= RC.EPOCHS
                    if done and (args.steps is None or
                                            old.get('steps') == str(args.steps)):
                        print('[skip] %s %s seed%d (已有 n_eval=%s steps=%s)'
                              % (method, tname, seed, old.get('n_eval'),
                                 old.get('steps')), flush=True)
                        continue
                t0 = time.time()
                try:
                    acc = run_one(method, tname, [src_ids, tgt_ids], args.load,
                                  seed, device, log, args.steps, args.ch,
                                  fs=args.fs, want_cm=args.cm, sig=args.sig,
                                  mods=args.mods, lp=args.lp)
                    st = summarize(acc)
                    dt = time.time() - t0
                    np.savetxt(log + '_acc.csv', acc, fmt='%.4f')
                    print('[%s] %-14s %-4s seed%d  last5=%.2f  last1=%.2f  '
                          'n=%d (%.0fs)' % (time.strftime('%H:%M:%S'), method,
                                            tname, seed, st['last5'],
                                            st['last1'], st['n'], dt), flush=True)
                    fsum.write('%d,%d,%s,%s,%d,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%s\n'
                               % (args.load, args.ch, method, tname, seed,
                                  'indomain' if tname.startswith('D') else 'lodo',
                                  RC.EPOCHS * ((len(src_ids) * 50 + RC.BATCH_SIZE - 1)
                                               // RC.BATCH_SIZE)
                                  if args.steps is None else args.steps,
                                  st['n'], st['last1'], st['last5'], st['last10'],
                                  st['peak'], dt, os.path.relpath(log, _ROOT)))
                    fsum.flush()
                except Exception as e:  # noqa: BLE001
                    print('[FAIL] %-14s %-4s seed%d : %s'
                          % (method, tname, seed, str(e)[:200]), flush=True)
    fsum.close()
    print('\n### ALL DONE @ %s' % time.strftime('%F %T'))
    report(args.load, args.ch, methods, list(tasks), seeds, fs=args.fs,
           sig=args.sig, tag_extra=args.tag, lp=args.lp)


if __name__ == '__main__':
    main()
