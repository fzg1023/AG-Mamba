#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 6 —— §6.3 前端设计：(a) 混叠路径 α 的精度曲线；(b) 前端消融（初始化 / Laplace / 低频 / 容量）。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/_s63_alpha.md` 的 (a1) 表  → α 曲线（T0 / T9 / 池化 last5）
  · `docs/manuscript/_s63_front.md` 的 (a1) 表  → 各臂 T0 / T4 / T9 / 池化 last5

未完成的臂（缺列显示 `–`）在图中以**斜纹占位 + "running" 标注**呈现，绝不画成 0。
因此数据落地后**无需改脚本**，重跑本文件即可自动补全。

风格与 Fig. 2 / Fig. 4 一致（`analysis/make_fig4_filterbank.py`）：Agg 后端、
面板下方图注、字号 6.0–6.5、颜色 #1f77b4/#d62728/#2ca02c、pdf+png @400 dpi。

输出：docs/manuscript/figures/fig6_frontend.pdf / .png
用法：python analysis/make_fig6_frontend.py
"""
from __future__ import annotations

import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

NA = float('nan')
LOG: list[str] = []


# ----------------------------------------------------------------- 解析工具
def _num(tok):
    """'**84.75**' -> 84.75 ; '–' / '—' / '*' -> nan ; '91.49 ± 0.55' -> (mean, std)."""
    tok = tok.replace('*', '').strip()
    if tok in ('–', '—', '-', '', '...'):
        return NA, NA
    parts = [p.strip() for p in tok.split('±')]
    try:
        m = float(parts[0])
    except ValueError:
        return NA, NA
    s = NA
    if len(parts) > 1:
        try:
            s = float(parts[1])
        except ValueError:
            s = NA
    return m, s


def parse_alpha(path):
    """_s63_alpha.md (a1) -> dict(alpha -> {'T0':(m,s), 'T9':(m,s), 'pool':float})."""
    out = {}
    for line in open(path, encoding='utf-8'):
        if not line.startswith('|'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 7:
            continue
        try:
            a = float(c[0].replace('*', '').strip())
        except ValueError:
            continue
        if not c[1].startswith('`as_a'):
            continue
        out[a] = {'T0': _num(c[3]), 'T9': _num(c[4]), 'pool': _num(c[5])[0]}
    return out


# (a1) 表里各臂的顺序与显示名；权重只用于排序
ARM_ORDER = ['delivered', 'fr_rand', 'fr_mix', 'fr_lap', 'fr_narrow',
             'fr_cap768', 'fr_cap960']
ARM_LABEL = {
    'delivered': 'delivered\n1.37 M',
    'fr_rand': 'random\n1.37 M',
    'fr_mix': '50/50\n1.37 M',
    'fr_lap': 'Laplace\n1.37 M',
    'fr_narrow': 'narrow\n1.37 M',
    'fr_cap768': '768\n2.6 M',
    'fr_cap960': '960\n3.9 M',
}


def parse_front(path):
    """_s63_front.md **仅 (a1) 段** -> dict(arm -> {'T0','T4','T9':(m,s), 'pool':float, 'params':str}).

    ⚠ 必须限定在 (a1) 段：文件下面还有 (a2) 配对检验表，也是表格、列数偶合，
    误取会把 p 值当成精度（已踩过）。另用"第 3 列像参数量"作第二道闸门。
    """
    out, in_a1 = {}, False
    for line in open(path, encoding='utf-8'):
        if line.startswith('#'):
            in_a1 = line.lstrip('#').strip().startswith('(a1)')
            continue
        if not (in_a1 and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 9 or not re.match(r'^[\d.]+\s+[MK]$', c[2]):
            continue
        name = c[0].replace('*', '').strip()
        if name.startswith('本方法'):
            key = 'delivered'
        elif name.startswith('`fr_'):
            key = name.strip('`')
        else:
            continue
        out[key] = {'T0': _num(c[3]), 'T4': _num(c[4]), 'T9': _num(c[5]),
                    'pool': _num(c[6])[0], 'params': c[2]}
    return out


# ------------------------------------------------------------------ 通用工具
def tidy(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.0)


def cap(ax, text, y=-0.30, size=5.6):
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.25', linespacing=1.35)


# ------------------------------------------------------------------ 面板 (a)
def draw_a(ax, A):
    al = sorted(A)
    t0m = np.array([A[a]['T0'][0] for a in al])
    t0s = np.array([A[a]['T0'][1] for a in al])
    t9m = np.array([A[a]['T9'][0] for a in al])
    t9s = np.array([A[a]['T9'][1] for a in al])
    pm = np.array([A[a]['pool'] for a in al])

    ax.errorbar(al, t9m, yerr=t9s, fmt='--s', color='#d62728', ms=3.0, lw=1.0,
                capsize=2.0, elinewidth=0.7,
                label=r'T9 = 80 Hz (4$\times$ extrapolation)')
    ax.errorbar(al, t0m, yerr=t0s, fmt='-o', color='#1f77b4', ms=3.0, lw=1.0,
                capsize=2.0, elinewidth=0.7, label='T0 = 20 Hz (low end)')
    ax.plot(al, pm, '-D', color='0.15', ms=2.8, lw=1.2,
            label='pooled T0+T9 (10 pairs)')

    k = al.index(2.0)
    ax.plot([2.0], [pm[k]], 'o', mfc='none', mec='0.15', ms=7.5, mew=0.9)
    ax.annotate('delivered\n$\\alpha=2$', xy=(2.0, pm[k]),
                xytext=(2.35, 69.5), fontsize=5.6, color='0.15',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.5'))
    ax.annotate('+%.2f at T9\n(%.2f$\\to$%.2f)' % (A[2.0]['T9'][0] - A[0.0]['T9'][0],
                                                   A[0.0]['T9'][0], A[2.0]['T9'][0]),
                xy=(0.0, A[0.0]['T9'][0]), xytext=(0.12, 55.0), fontsize=5.6,
                color='#d62728', ha='left', va='center')
    ax.set_xlabel(r'aliasing strength $\alpha$', fontsize=6.5)
    ax.set_ylabel('target accuracy (%)', fontsize=6.5)
    ax.set_xticks(al)
    ax.set_xticklabels(['%.1f' % a for a in al])
    ax.set_ylim(50, 100)
    ax.legend(fontsize=5.4, frameon=False, loc='lower right', ncol=1)
    tidy(ax)

    # 上轴：同一组 α 处的 Π（谱峰均比），把 §3.3 的平面与精度对齐
    PI = {0.0: 34.2, 0.5: 30.7, 1.0: 21.5, 1.5: 13.2, 2.0: 13.1, 3.0: 10.3}
    sec = ax.secondary_xaxis('top')
    sec.set_xticks(al)
    sec.set_xticklabels(['%.1f' % PI[a] for a in al], fontsize=5.4, color='0.4')
    sec.set_xlabel(r'$\Pi$ (peak-to-mean of the bank)', fontsize=5.6, color='0.4',
                   labelpad=1.0)
    sec.tick_params(length=2.0, color='0.5')
    for a in al:
        LOG.append('(a) alpha=%.1f  T0=%.2f  T9=%.2f  pool=%.2f  Pi=%.1f'
                   % (a, A[a]['T0'][0], A[a]['T9'][0], A[a]['pool'], PI[a]))


# ------------------------------------------------------------------ 面板 (b)
def draw_b(ax, F):
    tasks = ['T0', 'T4', 'T9']
    cols = ['#1f77b4', '#7f7f7f', '#d62728']
    arms = [a for a in ARM_ORDER if a in F]
    grp_w, bar_w = 0.80, 0.24
    base = 55.0                      # 柱自 55 % 起画（否则差异不可见）；图注已声明

    for i, arm in enumerate(arms):
        d = F[arm]
        have = [t for t in tasks if d[t][0] == d[t][0]]      # 非 NaN
        if not have:
            ax.axvspan(i - grp_w / 2, i + grp_w / 2, facecolor='0.92', lw=0,
                       hatch='///', edgecolor='0.78', zorder=0)
            ax.text(i, 62, 'running', rotation=90, fontsize=5.0, color='0.35',
                    ha='center', va='center')
            continue
        for j, t in enumerate(tasks):
            v, s = d[t]
            if v != v:
                continue
            ax.bar(i + (j - 1) * bar_w, v - base, bottom=base, width=bar_w * 0.86,
                   color=cols[j], alpha=0.92, lw=0)
            if s == s:
                ax.errorbar(i + (j - 1) * bar_w, v, yerr=s, fmt='none',
                            ecolor='0.2', elinewidth=0.7, capsize=1.5)
            if arm == 'delivered':        # 任务名直标：不用图例（会压住柱子）
                ax.text(i + (j - 1) * bar_w, v + s + 1.0, t, fontsize=5.0,
                        color=cols[j], ha='center', va='bottom')
        # 池化标记只在**三个任务齐全**时才画：缺任务时摘要文件里的"池化"
        # 只是已完成任务的平均，直接画会误导
        if all(d[t][0] == d[t][0] for t in tasks) and d['pool'] == d['pool']:
            ax.plot([i], [d['pool']], 'k_', ms=9, mew=1.3, zorder=5)
            ax.text(i, d['pool'] + 1.8, '%.2f' % d['pool'], fontsize=5.3,
                    ha='center', va='bottom', color='0.1')

    ax.axhline(F['delivered']['pool'], color='0.45', lw=0.7, ls=':')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=4.9)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylim(base, 104)
    ax.set_ylabel('target accuracy (%)', fontsize=6.5)
    tidy(ax)
    for arm in arms:
        d = F[arm]
        LOG.append('(b) %-10s T0=%.2f T4=%.2f T9=%.2f pool=%s params=%s'
                   % (arm, d['T0'][0], d['T4'][0], d['T9'][0], d['pool'], d['params']))


def main():
    A = parse_alpha(os.path.join(ROOT, 'docs', 'manuscript', '_s63_alpha.md'))
    F = parse_front(os.path.join(ROOT, 'docs', 'manuscript', '_s63_front.md'))
    assert len(A) == 6, 'alpha 表解析异常：%d 行' % len(A)
    assert 'delivered' in F and 'fr_rand' in F, '前端表解析异常'

    CAP_A = ('(a) Aliasing path on HUST (480-step budget, 5 seeds; the two anchors\n'
             'are pooled over 10 pairs). The knob is a far-end lever: $\\alpha=0\\to2$\n'
             'buys %.1f points at 80 Hz but only %.1f at 20 Hz. Top axis: bank $\\Pi$.'
             % (A[2.0]['T9'][0] - A[0.0]['T9'][0], A[2.0]['T0'][0] - A[0.0]['T0'][0]))
    CAP_B = ('(b) Frontend ablations (three targets, 15 pairs; bars: T0/T4/T9 with\n'
             'seed spread; black dash: pooled `last5`; dotted line: delivered\n'
             'configuration; bars drawn from 55 %). The last two groups are widths\n'
             '512$\\to$768/960. Hatched groups are still running, not zeros.')

    PANELS = [('a', lambda ax: draw_a(ax, A), CAP_A),
              ('b', lambda ax: draw_b(ax, F), CAP_B)]

    # ---------------- 输出 1：合成图（两面板并排，图注置于各自下方）------------
    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(1, 2, wspace=0.26, width_ratios=[1.0, 1.20])
    for i, (key, fn, caption) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        fn(ax)
        cap(ax, caption, y=-0.34)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGD, 'fig6_frontend.' + ext), dpi=400,
                    bbox_inches='tight')
    plt.close(fig)

    # ---------------- 输出 2：单面板图（便于后续排版拼版）-----------------------
    # 与 Fig. 4 的单面板导出同规则：fig6a_panel / fig6b_panel，PDF + PNG @400 dpi。
    # (b) 面板含 7 组柱，故比 (a) 宽一些；(a) 需给图例与上轴留位置。
    SIZE = {'a': (3.7, 3.4), 'b': (4.8, 3.4)}
    RECT = {'a': [0.15, 0.40, 0.82, 0.55],   # [left, bottom, width, height]
            'b': [0.11, 0.40, 0.87, 0.55]}
    for key, fn, caption in PANELS:
        f1 = plt.figure(figsize=SIZE[key])
        ax = f1.add_axes(RECT[key])
        fn(ax)
        ax.text(0.0, -0.235, caption, transform=ax.transAxes, ha='left',
                va='top', fontsize=6.3, color='0.25', linespacing=1.35)
        for ext in ('pdf', 'png'):
            f1.savefig(os.path.join(FIGD, 'fig6' + key + '_panel.' + ext),
                       dpi=400, bbox_inches='tight')
        plt.close(f1)

    # ---------------- 数值留档（与 Fig. 4 的 `_fig4_values.txt` 同规则）--------
    with open(os.path.join(FIGD, '_fig6_values.txt'), 'w', encoding='utf-8') as fh:
        fh.write('# Fig. 6 数值（由 analysis/make_fig6_frontend.py 从 '
                 '_s63_alpha.md / _s63_front.md 解析）\n')
        for l in LOG:
            fh.write(l + '\n')

    print('# Fig. 6 written to figures/fig6_frontend.pdf/.png '
          '+ fig6a_panel / fig6b_panel（PDF+PNG）')
    for l in LOG:
        print(l)


if __name__ == '__main__':
    main()
