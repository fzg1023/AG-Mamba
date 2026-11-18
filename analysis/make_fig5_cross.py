#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 5 —— 三基准 cross 汇总：主对比的"一图总览"（21 方法 → 20 个配对 Δ）。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/_cmp_PHM2009_stats.md`
  · `docs/manuscript/_cmp_PU_stats.md`
  · `docs/manuscript/_cmp_HUST_stats.md`
  取每个文件里 **`unif` vs 各基线** 且标题含 **"主口径 `last5`"** 的那一段
  （`pub` 段与 `last1`/`last10`/`peak` 段一律不读）。

一张单面板图，回答一个问题："对二十个基线的领先在三个基准上是不是同一回事？"
  · 每个基线一行（纵轴），按三基准**平均 Δ** 排序；
  · 每个基准一个标记（圆 = PHM2009、方 = PU、三角 = HUST），同一方法的三点
    用细灰线连起来 —— 线的**长度**就是一个方法跨基准的离散度；
  · 灰带 = ±2.55 点可复现下限（§5.3），落在带内的差值按规矩不作解释；
  · **空心标记 = 该格不显著**（双侧 Wilcoxon p ≥ 0.05）—— 全文只有两格：
    PHM2009 无、PU 的 Mixup、HUST 的 WIDAN，即 58/60。

风格与 Fig. 6 / 8 / 9 / 10 / 11 / 12 一致：Agg 后端、**图题置于坐标轴下方**
（同 `make_fig6_frontend.py` 的 `cap()` 规则，不用 `set_title`）、pdf+png @400 dpi。

输出：docs/manuscript/figures/
  · fig5_cross.pdf / .png    （单面板，本身即"单独成图"版，含下端图题）
  · _fig5_values.txt

用法：python analysis/make_fig5_cross.py
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
MD = os.path.join(ROOT, 'docs', 'manuscript')
FIGD = os.path.join(MD, 'figures')
os.makedirs(FIGD, exist_ok=True)

FLOOR = 2.55
BENCH = [('PHM2009', '_cmp_PHM2009_stats.md', 'o', '#1f77b4'),
         ('PU', '_cmp_PU_stats.md', 's', '#d62728'),
         ('HUST', '_cmp_HUST_stats.md', '^', '#2ca02c')]

ROW = re.compile(
    r'^\|\s*([A-Za-z_0-9]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    r'\s*\*{0,2}([+-]?[\d.]+)\*{0,2}\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|'
    r'\s*([\d.]+)\s*\|')

LOG: list[str] = []
log = LOG.append


def parse(path):
    """只读 `unif` 段的 **主口径 last5** 表 → {方法: (Δ, wins, n, p)}。"""
    lines = open(path, encoding='utf-8').read().split('\n')
    sec, out = None, {}
    for ln in lines:
        if ln.startswith('## '):
            sec = ('unif' in ln) and ('主口径' in ln) and ('`last5`' in ln)
            continue
        if not sec:
            continue
        m = ROW.match(ln)
        if not m:
            continue
        name, n, _ours, _base, d, _w, _wn, p = m.groups()
        if name.lower().startswith('基线'):
            continue
        out[name] = (float(d), int(_w), int(_wn), float(p))
    return out


def main():
    data, meta = {}, {}
    for tag, fn, mk, col in BENCH:
        d = parse(os.path.join(MD, fn))
        if len(d) != 20:
            raise SystemExit('❌ %s：解析到 %d 个基线（应为 20）—— 请检查底稿标题'
                             % (fn, len(d)))
        data[tag] = d
        meta[tag] = (mk, col)
        nsig = [k for k, v in d.items() if v[3] >= 0.05]
        log('%s: %d 个基线，不显著 %d 个 %s'
            % (tag, len(d), len(nsig), nsig if nsig else ''))

    names = sorted(data['HUST'],
                   key=lambda k: -np.mean([data[t][k][0] for t, _, _, _ in BENCH]))
    for k in data['HUST']:
        if k not in names:
            raise SystemExit('❌ 三文件的基线名不一致：%s' % k)

    # ---------------------------------------------------------------- 作图
    fig = plt.figure(figsize=(7.4, 5.3))
    ax = fig.add_axes([0.20, 0.175, 0.735, 0.79])

    ys = np.arange(len(names))[::-1]
    ax.axvspan(-FLOOR, FLOOR, color='0.90', lw=0, zorder=0)
    ax.axvline(0.0, color='0.25', lw=0.8, zorder=1)

    n_cells = n_sig = 0
    for y, k in zip(ys, names):
        ds = [data[t][k][0] for t, _, _, _ in BENCH]
        ax.plot([min(ds), max(ds)], [y, y], color='0.72', lw=0.9, zorder=2)
        for t, _, _, _ in BENCH:
            mk, col = meta[t]
            d, _w, _wn, p = data[t][k]
            inband = abs(d) < FLOOR
            ax.plot([d], [y], marker=mk, ms=4.4, ls='none',
                    mfc=('none' if p >= 0.05 else col),
                    mec=('0.45' if inband else col), mew=0.9, zorder=4)
            n_cells += 1
            n_sig += int(p < 0.05)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlabel(r'pooled $\Delta$ against the proposed method   (points)',
                  fontsize=7.0)
    ax.tick_params(axis='x', labelsize=6.4)
    # x 上限必须由数据定：HUST 上的 WhiteningNet 达 +41.4 点，远超过两个
    # 小基准的量程（PHM2009 最大 +23.0）。写死上限会把标记推出轴外。
    dmax = max(data[t][k][0] for t, _, _, _ in BENCH for k in names)
    ax.set_xlim(-3.0, dmax + 2.5)
    ax.set_ylim(-1.0, len(names))
    ax.grid(axis='x', alpha=0.22, lw=0.4)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # 下限带的文字标注放在轴内左上角空白处（右上角被最大的 Δ 占据）
    ax.text(-2.4, len(names) - 0.35, r'$\pm$%.2f floor' % FLOOR,
            fontsize=5.4, color='0.35', va='top', ha='left')

    h = [Line2D([], [], marker=mk, color=col, ms=4.4, ls='none')
         for _, _, mk, col in BENCH]
    h.append(Line2D([], [], marker='o', color='0.45', ms=4.4, ls='none',
                    mfc='none', mew=0.9))
    lab = [t for t, _, _, _ in BENCH] + ['not significant (p >= 0.05)']
    ax.legend(h, lab, fontsize=5.8, frameon=False, loc='lower right',
              handlelength=0.9, labelspacing=0.35, borderpad=0.1,
              handletextpad=0.5)

    # 单面板图：图题不带 (a)（本节只有这一张图，无需面板字母）
    cap1 = ('cross-benchmark summary of the main comparison: pooled '
            r'$\Delta$ of the proposed model against each of the twenty baselines')
    cap2 = ('one marker per benchmark (circle PHM 2009, square PU, '
            'triangle HUST); the grey bar joins the three values of a method\n'
            'hollow markers: not significant (two-sided Wilcoxon '
            'p >= 0.05);  shaded band: the ' + r'$\pm$2.55' + '-point '
            'reproducibility floor of the code base')
    ax.text(0.0, -0.205, cap1, transform=ax.transAxes, fontsize=5.6,
            va='top', ha='left')
    ax.text(0.0, -0.275, cap2, transform=ax.transAxes, fontsize=5.0,
            va='top', ha='left', color='0.25')

    # 版式自检：所有标记必须落在轴内，且图题不得越出画布
    fig.canvas.draw()
    bb = ax.get_window_extent()
    for y, k in zip(ys, names):
        for t, _, _, _ in BENCH:
            d = data[t][k][0]
            x_px = ax.transData.transform((d, y))[0]
            assert bb.x0 - 1 <= x_px <= bb.x1 + 1, (k, t, d)
    log('layout check: OK')

    out = []
    out.append('# Fig. 5 数值（主口径 `last5`，`unif` 臂；Δ = 本方法 − 基线，单位点）')
    out.append('# 底稿：_cmp_PHM2009_stats.md / _cmp_PU_stats.md / _cmp_HUST_stats.md')
    out.append('# 排序：三基准平均 Δ 降序；p >= 0.05 标 [ns]；|Δ| < 2.55 标 [band]')
    out.append('')
    for t, _, _, _ in BENCH:
        ds = [data[t][k][0] for k in names]
        ns = sum(1 for k in names if data[t][k][3] >= 0.05)
        out.append('# %s: 池化 Δ 范围 %.2f … %.2f，不显著 %d/20'
                   % (t, min(ds), max(ds), ns))
    out.append('# 合计：%d/%d 显著' % (n_sig, n_cells))
    out.append('')
    out.append('%-14s %9s %9s %9s %9s' % ('baseline', 'PHM2009', 'PU', 'HUST', 'mean'))
    for k in names:
        ds = [data[t][k][0] for t, _, _, _ in BENCH]
        cells = []
        for t, _, _, _ in BENCH:
            d, _w, _wn, p = data[t][k]
            flag = 'ns' if p >= 0.05 else ''
            if abs(d) < FLOOR:
                flag += '*'
            cells.append('%+8.2f%s' % (d, flag))
        out.append('%-14s %s %+9.2f' % (k, ' '.join(cells), float(np.mean(ds))))
    log('wrote ' + '\n'.join(LOG[-1:]))

    for ext in ('pdf', 'png'):
        # 与其余图脚本一致：显式 rect 定位坐标轴 + `bbox_inches='tight'` 导出，
        # 否则画在轴外的**下端图题第二行会被裁掉**（本脚本首版即踩此坑）。
        fig.savefig(os.path.join(FIGD, 'fig5_cross.' + ext), dpi=400,
                    bbox_inches='tight', pad_inches=0.02)
    open(os.path.join(FIGD, '_fig5_values.txt'), 'w',
         encoding='utf-8').write('\n'.join(out) + '\n')
    plt.close(fig)

    print('\n'.join(LOG))
    print('wrote fig5_cross.[pdf|png] + _fig5_values.txt')


if __name__ == '__main__':
    main()
