"""生成 Fig. 14（跨数据集综合：三协议 + 容量）与 Fig. 15（运行间可复现性底噪）。

数据来源：results_dgfdb/ 下的运行日志（per-epoch tgt_acc，best-epoch 口径）。
输出：docs/manuscript/figures/fig14.{pdf,png}、fig15.{pdf,png}，
      并打印各面板数值以便与手稿核对。
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

D = 'results'
OUT = 'docs/manuscript/figures'
FIGD = 'DGFD論文/figures'
ACC = re.compile(r'tgt_acc:\s*([0-9.]+)')
STEP = re.compile(r'Step (\d+)')


def parse(path):
    """返回 (steps, losses, accs)：逐评估点的步号、loss_total、tgt_acc。"""
    steps, losses, accs = [], [], []
    cur_step, cur_loss = None, None
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            m = STEP.search(line)
            if m:
                cur_step = int(m.group(1))
                continue
            if line.startswith('loss_total:'):
                try:
                    cur_loss = float(line.split(':')[1].split('|')[0])
                except ValueError:
                    pass
                continue
            m = ACC.search(line)
            if m:
                steps.append(cur_step)
                losses.append(cur_loss)
                accs.append(float(m.group(1)) * 100.0)  # tgt_acc 以 0-1 分数记录 -> 百分点
    return steps, losses, accs


def best(path):
    _, _, a = parse(path)
    return max(a) if a else float('nan')


def find(data, task, arm, seed):
    """日志可能落在多个目录下（新协议在 results_dgfdb，原始八任务基线在
    results_widan / results_udtl / results_pu / results_phm2009；后两者中
    WIDAN/UDTL 的文件名不含方法名）。"""
    for name in ('{}_{}_{}'.format(data, task, arm), '{}_{}'.format(data, task)):
        p = _findlog('{}_seed{}.log'.format(name, seed))
        if os.path.exists(p):
            return p
    return None


def series(data, task, arm, seeds=(0, 1, 2, 3, 4)):
    out = []
    for s in seeds:
        p = find(data, task, arm, s)
        if p is not None:
            out.append(best(p))
    return np.array(out)


OURS = 'AGMamba_fc_xg_std'
BASE = 'AGMamba_fc_base'
CAP = 'AGMamba_fc_cap960'


def proto_means(data, tasks, arm):
    vals = [series(data, t, arm) for t in tasks]
    flat = np.concatenate([v for v in vals if len(v)])
    return flat.mean(), flat.std(ddof=1)


HIGH, LOW, MIX = ['T0', 'T1', 'T2', 'T3'], ['L0', 'L1', 'L2', 'L3'], ['M0', 'M1', 'M2', 'M3']
PU = ['T0', 'T1', 'T2', 'T3']

# ---------------------------------------------------------------- Fig. 14
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
ax = axes[0][0]

protos = [('High ladder\n(T0-T3)', HIGH), ('Low ladder\n(L0-L3)', LOW), ('Two-factor\n(M0-M3)', MIX)]
arms = [(OURS, 'AG-Mamba, 512 ch (1.37M)', '#1f4e79'),
        (CAP, 'AG-Mamba, 960 ch (3.85M)', '#5b9bd5'),
        (BASE, 'narrow-band control', '#a6a6a6'),
        ('WIDAN', 'WIDAN (3.98M)', '#c00000')]
x = np.arange(len(protos))
w = 0.2
for i, (arm, lab, col) in enumerate(arms):
    means, errs = [], []
    for _, tasks in protos:
        m, s = proto_means('PHM2009', tasks, arm)
        means.append(m)
        errs.append(s)
    ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=2.5, color=col, label=lab)
    for xi, m in zip(x + (i - 1.5) * w, means):
        ax.text(xi, m + 1.2, '{:.1f}'.format(m), ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in protos], fontsize=8)
ax.set_ylabel('Best-epoch accuracy (%)')
ax.set_ylim(60, 96)
ax.legend(fontsize=7.5, loc='lower left', ncol=2, framealpha=0.9)
ax.set_title('(a) PHM 2009: three protocols from the same 20 records', fontsize=9)
ax.grid(axis='y', alpha=0.25)

# (b) PU capacity ladder + HUST budget-matched points
ax = axes[0][1]
ch = [128, 512, 960]
w128 = np.mean([series('PU', t, 'AGMamba_fc_puac_a2_c128').mean() for t in PU])
w512 = np.mean([series('PU', t, 'AGMamba_fc_puac_a2_c512').mean() for t in PU])
w960 = np.mean([series('PU', t, CAP).mean() for t in PU])
a0_128 = np.mean([series('PU', t, 'AGMamba_fc_puac_a0_c128').mean() for t in PU])
a0_512 = np.mean([series('PU', t, 'AGMamba_fc_puac_a0_c512').mean() for t in PU])
xg_pu = np.mean([series('PU', t, OURS).mean() for t in PU])
wid_pu = np.mean([series('PU', t, 'WIDAN').mean() for t in PU])
base_pu = np.mean([series('PU', t, BASE).mean() for t in PU])
ax.plot(ch, [w128, w512, w960], 'o-', color='#1f4e79', label=r'ours, $\alpha$ = 2.0')
ax.plot([128, 512], [a0_128, a0_512], 's--', color='#7f7f7f', label=r'ours, $\alpha$ = 0 (analytic)')
ax.axhline(wid_pu, color='#c00000', ls=':', lw=1.2)
ax.text(710, wid_pu + 0.4, 'WIDAN, 3.98M ({:.2f})'.format(wid_pu), color='#c00000', fontsize=7.5)
ax.axhline(base_pu, color='#a6a6a6', ls=':', lw=1.2)
ax.text(710, base_pu - 1.1, 'narrow-band control ({:.2f})'.format(base_pu), color='#7f7f7f', fontsize=7.5)
ax.axhline(xg_pu, color='#1f4e79', ls=':', lw=1.0)
ax.text(150, xg_pu + 0.25, 'delivered run, same config ({:.2f})'.format(xg_pu), color='#1f4e79', fontsize=7.5)
ax.annotate('', xy=(512, w512), xytext=(512, w128),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1))
ax.text(530, (w512 + w128) / 2, '+1.99\np = 0.0005', fontsize=7.5, va='center')
ax.set_xscale('log')
ax.set_xticks(ch)
ax.set_xticklabels(['128\n(16 atoms)', '512\n(64 atoms)', '960\n(120 atoms)'], fontsize=8)
ax.set_xlabel('Front-end width (channels / bank atoms)')
ax.set_ylabel('Best-epoch accuracy, 4-task mean (%)')
ax.set_title('(b) PU: capacity ladder, aliasing held fixed', fontsize=9)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(alpha=0.25)

# (c) HUST probe capacity curve
ax = axes[1][0]
probe = [('AGMamba_fc_sa_lowf512', 512), ('AGMamba_fc_sa_cap768_lowf', 768),
         ('AGMamba_fc_cap960p', 960), ('AGMamba_fc_sa_cap1024', 1024)]
xs, ys, es = [], [], []
for arm, c in probe:
    v = np.concatenate([series('HUST', t, arm) for t in ['T0', 'T4', 'T9']])
    xs.append(c)
    ys.append(v.mean())
    es.append(v.std(ddof=1) / np.sqrt(len(v)))
ax.errorbar(xs, ys, yerr=es, fmt='o-', color='#1f4e79', capsize=3)
for c, m in zip(xs, ys):
    ax.text(c, m + 0.22, '{:.2f}'.format(m), ha='center', fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels([str(c) for c in xs])
ax.set_xlabel('Front-end width (channels)')
ax.set_ylabel('Probe accuracy, pooled T0/T4/T9 (%)')
ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
ax.set_title('(c) HUST: same ladder, 480-step probe (d,e)', fontsize=9)
ax.grid(alpha=0.25)
ax.text(0.03, 0.06, '960 - 512 = -0.03 (p = 0.94, 8/15 pairs)',
        transform=ax.transAxes, fontsize=7.5)

# (d) lever contrast
ax = axes[1][1]
labels = ['HUST T9\n(WIDAN forensics)', 'PU, 4 tasks\n(own 2x2 control)']
shape = [20.02, 1.29]
cap = [4.96, 1.99]
xs = np.arange(2)
ax.bar(xs - 0.18, shape, 0.36, color='#1f4e79', label='spectral-shape lever')
ax.bar(xs + 0.18, cap, 0.36, color='#f0a030', label='capacity lever')
ax.text(-0.18, 20.4, '20.02\n(random init)', ha='center', fontsize=7.5)
ax.text(0.18, 5.3, '4.96\n(halve)', ha='center', fontsize=7.5)
ax.text(0.82 - 0.0, 1.7, '1.29 (n.s.)\n' + r'$\alpha$ 2$\to$0 @512', ha='center', fontsize=7.5)
ax.text(1.18, 2.4, '1.99\n(p = 0.0005)', ha='center', fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Effect size (points)')
ax.set_ylim(0, 23)
ax.legend(fontsize=7.5, loc='upper right')
ax.set_title('(d) What moves accuracy is benchmark-specific', fontsize=9)
ax.grid(axis='y', alpha=0.25)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig14.pdf'))
fig.savefig(os.path.join(OUT, 'fig14.png'), dpi=300)
print('[OK] fig14')

# ---------------------------------------------------------------- Fig. 15
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

# (a) rep_a vs rep_b
ax = axes[0]
cells = [('PU', 'T0', 'PU T0'), ('PU', 'T2', 'PU T2'), ('HUST', 'T9', 'HUST T9')]
cols = ['#1f4e79', '#5b9bd5', '#c00000']
d_all = []
for (data, task, lab), col in zip(cells, cols):
    a = series(data, task, 'AGMamba_fc_rep_a')
    b = series(data, task, 'AGMamba_fc_rep_b')
    d_all.append(np.abs(a - b))
    ax.scatter(a, b, s=28, color=col, label=lab, zorder=3)
lim = [min(min(series(data, task, 'AGMamba_fc_rep_a').min(),
                    series(data, task, 'AGMamba_fc_rep_b').min())
                for data, task, _ in cells) - 4,
       max(max(series(data, task, 'AGMamba_fc_rep_a').max(),
               series(data, task, 'AGMamba_fc_rep_b').max())
           for data, task, _ in cells) + 3]
ax.plot(lim, lim, 'k--', lw=1, label='identity')
ax.fill_between(lim, [lim[0] - 2, lim[1] - 2], [lim[0] + 2, lim[1] + 2], color='k', alpha=0.07,
                label=r'$\pm$2-point band')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('accuracy, run A (%)')
ax.set_ylabel('accuracy, run B (%)')
ax.legend(fontsize=7, loc='upper left')
ax.set_title('(a) Same config, same seed, two processes', fontsize=9)
ax.grid(alpha=0.25)

# (b) |delta| per seed
ax = axes[1]
for i, ((data, task, lab), col) in enumerate(zip(cells, cols)):
    d = d_all[i]
    jitter = (np.arange(len(d)) - len(d) / 2) * 0.045
    ax.scatter(np.full(len(d), i) + jitter, d, s=26, color=col, zorder=3)
    ax.plot([i - 0.22, i + 0.22], [d.mean()] * 2, color='k', lw=1.6, zorder=4)
    ax.text(i + 0.02, d.mean() + 0.22, '{:.2f}'.format(d.mean()), fontsize=7.5)
pool = np.concatenate(d_all)
ax.axhline(pool.mean(), color='#c00000', ls=':', lw=1.2)
ax.text(-0.42, pool.mean() + 0.2, 'pooled mean {:.2f}'.format(pool.mean()),
        color='#c00000', fontsize=7.5)
ax.axhline(pool.max(), color='#7f7f7f', ls=':', lw=1)
ax.text(-0.42, pool.max() - 0.55, 'max {:.2f}'.format(pool.max()), color='#7f7f7f', fontsize=7.5)
ax.set_xticks(range(len(cells)))
ax.set_xticklabels([c[2] for c in cells], fontsize=8)
ax.set_xlim(-0.5, len(cells) - 0.5)
ax.set_ylabel('|accuracy difference| (points)')
ax.set_title('(b) Per-seed spread of the replicate pairs', fontsize=9)
ax.grid(axis='y', alpha=0.25)

# (c) training curves of the two processes
ax = axes[2]
st, _, ac = parse(find('PU', 'T2', 'AGMamba_fc_rep_a', 0))
st2, _, ac2 = parse(find('PU', 'T2', 'AGMamba_fc_rep_b', 0))
ax.plot(range(1, len(ac) + 1), ac, color='#1f4e79', lw=1.3, label='run A')
ax.plot(range(1, len(ac2) + 1), ac2, color='#c00000', lw=1.3, ls='--', label='run B')
ax.set_xlabel('epoch (evaluation point)')
ax.set_ylabel('target accuracy (%)')
ax.set_title('(c) PU T2, seed 0: identical until ~epoch 3', fontsize=9)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(alpha=0.25)
ax.text(0.03, 0.72, 'steps 1-3 bit-identical;\nsplit at the 4th decimal',
        transform=ax.transAxes, fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig15.pdf'))
fig.savefig(os.path.join(OUT, 'fig15.png'), dpi=300)
print('[OK] fig15')

# ---------------------------------------------------------------- 核对打印
print('\n--- Fig 14(a) 三协议（20 次运行的 mean ± s.d.）---')
for lab, tasks in protos:
    row = []
    for arm, alab, _ in arms:
        m, s = proto_means('PHM2009', tasks, arm)
        row.append('{} {:.2f}+-{:.2f}'.format(alab.split(',')[0], m, s))
    print(lab.replace('\n', ' '), ' | '.join(row))
print('\n--- Fig 14(b) PU 容量 ---')
print('alpha=2: 128ch {:.2f}  512ch {:.2f}  960ch {:.2f}'.format(w128, w512, w960))
print('alpha=0: 128ch {:.2f}  512ch {:.2f}'.format(a0_128, a0_512))
print('xg_std(run) {:.2f}  base {:.2f}  WIDAN {:.2f}'.format(xg_pu, base_pu, wid_pu))
print('\n--- Fig 14(c) HUST 探针容量 ---')
for (arm, c), m in zip(probe, ys):
    print('{:5d} ch  {:.2f}'.format(c, m))
print('\n--- Fig 15(a)(b) 逐种子 |delta|（rep_a vs rep_b）---')
for (data, task, lab), d in zip(cells, d_all):
    print('{:9s} mean {:.2f}  max {:.2f}  seeds {}'.format(lab, d.mean(), d.max(),
                                                          np.round(d, 2).tolist()))
print('pooled mean {:.2f}  max {:.2f}  n = {}'.format(pool.mean(), pool.max(), len(pool)))
