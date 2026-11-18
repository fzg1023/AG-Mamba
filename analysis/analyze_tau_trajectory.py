#!/usr/bin/python
# -*- coding:utf-8 -*-
"""E1: 从现有训练日志提取 τ₀ / Δ 轨迹（纯 CPU，不占 GPU）。

扫描 results_dgfdb/ 下 AG-Mamba 的 base 日志（xxx_{DATA}_{TASK}_AGMamba_seed{S}.log），
解析每步的  loss_total | ce | mmd | adv | delta | tau0 ，回答三个问题：

  Q1  τ₀ 在训练过程中稳定吗（是否收敛，还是持续漂移）？
  Q2  τ₀ 稳态值是否随 LODO 任务（源域集合）变化？—— 若几乎不变，
      说明 τ₀ 通道对域差异不敏感（对可解释性是负面信号）。
  Q3  τ₀ 稳态值与最终精度是否相关？

产出:
  docs/manuscript/figs/tau0_trajectory.png       每条轨迹一条曲线
  docs/manuscript/figs/tau0_steady_vs_task.png   稳态 τ₀ / Δ 随任务变化 + 与精度对照
  docs/manuscript/figs/tau0_trajectory_stats.csv

用法: python analyze_tau_trajectory.py [DATA ...]    默认 PHM2009 PU HUST UO
"""
import os
import re
import sys
import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join('docs/manuscript', 'figs')
os.makedirs(OUTDIR, exist_ok=True)

RE_TAU = re.compile(r'tau0:\s*([0-9.eE+-]+)')
RE_DELTA = re.compile(r'delta:\s*([0-9.eE+-]+)')
RE_ACC = re.compile(r'tgt_acc:\s*([0-9.]+)')

TASK_ORDER = ['T{}'.format(i) for i in range(10)]


def parse_log(path):
    """-> dict(tau=list, delta=list, acc=list)"""
    tau, delta, acc = [], [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = RE_TAU.search(line)
            if m:
                tau.append(float(m.group(1)))
            m = RE_DELTA.search(line)
            if m:
                delta.append(float(m.group(1)))
            m = RE_ACC.search(line)
            if m:
                acc.append(float(m.group(1)))
    return {'tau': np.array(tau), 'delta': np.array(delta),
            'acc': np.array(acc)}


def find_logs(data):
    """base 配置日志，兼容多种命名:
         {DATA}_{TASK}_AGMamba_seed{S}.log                     (plain)
         {DATA}_{TASK}_AGMamba_fc_base_seed{S}.log              (fc_base)
         {DATA}_{TASK}_AGMamba_sweep_base_seed{S}.log           (sweep_base)
         {DATA}_{TASK}_AGMamba_rm_base_seed{S}.log              (rm_base)
    返回 {(task, seed, tag): path}
    """
    out = {}
    from agmamba.paths import iter_logs as _iter_logs
    for p in _iter_logs(dataset=data, method='AGMamba'):
        b = os.path.basename(p)
        m = re.match(
            r'{}_(T\d+)_AGMamba(?:_([a-zA-Z0-9_]+?))?_seed(\d+)\.log'
            .format(data), b)
        if not m:
            continue
        task, pre, seed = m.group(1), m.group(2) or 'plain', int(m.group(3))
        # 只保留 base 配置：无前缀，或前缀以 base 结尾
        if pre != 'plain' and not pre.endswith('base'):
            continue
        out[(task, seed, pre)] = p
    return out


def main():
    datas = sys.argv[1:] or ['PHM2009', 'PU', 'HUST', 'UO']
    rows = []
    print('=' * 96)
    print('E1: τ₀ / Δ 轨迹分析')
    print('=' * 96)

    for data in datas:
        logs = find_logs(data)
        if not logs:
            continue
        tasks = sorted({t for t, s, g in logs},
                       key=lambda x: TASK_ORDER.index(x)
                       if x in TASK_ORDER else 99)
        print('\n### {}  ({} 条 base 日志)'.format(data, len(logs)))
        print('{:6s} {:>5s} {:>12s} {:>8s} {:>8s} {:>8s} {:>9s} {:>8s} '
              '{:>7s}'.format('task', 'seed', 'tag', 'tau_head', 'tau_ss',
                              'tau_std', 'drift', 'delta_ss', 'bestacc'))
        for task in tasks:
            for seed in sorted({s for t, s, g in logs if t == task}):
                for (t, s, tag) in sorted(logs):
                    if t != task or s != seed:
                        continue
                    r = parse_log(logs[(t, s, tag)])
                    if len(r['tau']) < 10:
                        continue
                    n = len(r['tau'])
                    head = r['tau'][:max(1, n // 5)].mean()
                    ss = r['tau'][-(n // 3):].mean()   # 稳态 = 后 1/3
                    std = r['tau'][-(n // 3):].std()
                    drift = (ss - head) / max(abs(head), 1e-9)
                    d_ss = (r['delta'][-(n // 3):].mean()
                            if len(r['delta']) else np.nan)
                    acc = r['acc'].max() * 100 if len(r['acc']) else np.nan
                    rows.append(dict(data=data, task=task, seed=seed, tag=tag,
                                     tau_head=head, tau_ss=ss, tau_std=std,
                                     drift=drift, delta_ss=d_ss, acc=acc,
                                     respath=logs[(t, s, tag)]))
                    print('{:6s} {:5d} {:>12s} {:8.5f} {:8.5f} {:8.5f} '
                          '{:+8.1%} {:8.4f} {:7.2f}'.format(
                              task, seed, tag, head, ss, std, drift,
                              d_ss, acc))

        # ---- 任务级汇总（Q2）----
        print('  --- {} 任务级稳态 τ₀（跨种子均值±std） ---'.format(data))
        for task in tasks:
            v = np.array([r['tau_ss'] for r in rows
                          if r['data'] == data and r['task'] == task])
            dv = np.array([r['delta_ss'] for r in rows
                           if r['data'] == data and r['task'] == task])
            av = np.array([r['acc'] for r in rows
                           if r['data'] == data and r['task'] == task])
            if len(v):
                print('   {:5s}: tau_ss {:.5f} +- {:.5f} | delta_ss {:.4f} '
                      '+- {:.4f} | acc {:.2f}'.format(
                          task, v.mean(), v.std(), dv.mean(), dv.std(),
                          np.nanmean(av)))

    if not rows:
        print('\n没有找到任何 AG-Mamba base 日志。')
        return

    # =====================================================================
    # 图 1：τ₀ 轨迹
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6))
    for k, data in enumerate(datas):
        ax = axes[k // 2][k % 2]
        sub = [r for r in rows if r['data'] == data and r['seed'] == 0]
        if not sub:
            ax.axis('off')
            continue
        for r in sorted(sub, key=lambda x: TASK_ORDER.index(x['task'])
                        if x['task'] in TASK_ORDER else 99):
            rr = parse_log(r['respath'])
            ax.plot(rr['tau'], lw=1.0, alpha=0.8, label=r['task'])
        ax.set_title('{}  (seed 0, τ₀ per step)'.format(data), fontsize=10)
        ax.set_xlabel('training step')
        ax.set_ylabel(r'$\tau_0$ (batch mean)')
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=7.5)
    fig.suptitle(r'E1 (Q1): $\tau_0$ trajectory during training —— '
                 r'is it stable / does it drift?', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, 'tau0_trajectory.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print('\n[saved] {}'.format(p), flush=True)

    # =====================================================================
    # 图 2：稳态 τ₀ / Δ 随任务变化 + 与精度对照
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for data in datas:
        ts = []
        for task in TASK_ORDER:
            v = [r for r in rows if r['data'] == data and r['task'] == task]
            if v:
                ts.append((task, np.mean([r['tau_ss'] for r in v]),
                           np.mean([r['delta_ss'] for r in v]),
                           np.nanmean([r['acc'] for r in v])))
        if not ts:
            continue
        x = range(len(ts))
        axes[0].plot(x, [t[1] for t in ts], marker='o', lw=1.5, label=data)
        axes[1].plot(x, [t[2] for t in ts], marker='s', lw=1.5, label=data)
        axes[2].plot([t[1] for t in ts], [t[3] for t in ts], marker='o',
                     ls='none', ms=7, label=data)
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels([t[0] for t in ts], fontsize=8)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels([t[0] for t in ts], fontsize=8)
    axes[0].set_title(r'(a) Q2: steady-state $\tau_0$ vs LODO task')
    axes[0].set_xlabel('task'); axes[0].set_ylabel(r'$\tau_0$ (steady)')
    axes[1].set_title(r'(b) steady-state $\Delta_n$ vs LODO task')
    axes[1].set_xlabel('task'); axes[1].set_ylabel(r'$\|\Delta\|^2$ (steady)')
    axes[2].set_title(r'(c) Q3: steady $\tau_0$ vs best target acc')
    axes[2].set_xlabel(r'$\tau_0$ (steady)'); axes[2].set_ylabel('best acc (%)')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'tau0_steady_vs_task.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print('[saved] {}'.format(p), flush=True)

    # =====================================================================
    # CSV + 结论
    # =====================================================================
    csv = os.path.join(OUTDIR, 'tau0_trajectory_stats.csv')
    with open(csv, 'w') as f:
        f.write('data,task,seed,tag,tau_head,tau_ss,tau_std,drift,delta_ss,'
                'best_acc\n')
        for r in rows:
            f.write('{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.4f}\n'
                    .format(r['data'], r['task'], r['seed'], r['tag'],
                            r['tau_head'], r['tau_ss'], r['tau_std'],
                            r['drift'], r['delta_ss'], r['acc']))
    print('[saved] {}'.format(csv), flush=True)

    print('\n' + '=' * 96)
    print('结论')
    print('=' * 96)
    for data in datas:
        sub = [r for r in rows if r['data'] == data]
        if not sub:
            continue
        tss = np.array([r['tau_ss'] for r in sub])
        dr = np.array([r['drift'] for r in sub])
        rel_std = tss.std() / max(tss.mean(), 1e-9)
        print('{:8s}: tau_ss {:.5f} +- {:.5f} (相对离散 {:.1%}) | '
              'drift 中位 {:+.1%} | 稳态波动(std/mean) {:.1%}'.format(
                  data, tss.mean(), tss.std(), rel_std, np.median(dr),
                  np.mean([r['tau_std'] / max(r['tau_ss'], 1e-9) for r in sub])))
    print('\n判读要点:')
    print('  · 相对离散很小(<2%)  -> Q2 为否：τ₀ 对域差异不敏感（对可解释性不利）')
    print('  · 稳态波动很小(<2%)  -> Q1 为是：τ₀ 稳定，未发散')
    print('  · drift 接近 0       -> 训练早期即收敛，后续无漂移')
    print('=' * 96)


if __name__ == '__main__':
    main()
