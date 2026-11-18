#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 10 —— §6.7 HUST：逐任务结构 + 难度梯度（21 方法）。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/_hust_tables.md` 的 **HUST-T1** 表 → 21 方法 × 10 任务（T0…T9）
    的 `last5` 均值 ± std 与十任务池化（n = 50）
  · `docs/manuscript/_cmp_HUST_stats.md` 的 **`unif` vs 各基线** 段 → 池化 Δ / wins / p

三个面板：
  (a) **难度梯度**：每个方法把 T9(80 Hz) 与 T0(20 Hz) 相对其**自身十任务均值**的偏差
      并列（按 T9 偏差排序）—— 本图直接支撑 §3.1/§3.7 的那两句：80 Hz 对**全部**
      方法都是最低精度任务（低 14.7–33.1 点），而 20 Hz 偏差从 +3.4 到 −10.9。
  (b) **逐任务曲线**：5 个代表方法（本方法、最强基线 WIDAN、第二 MRA-SNN、
      最弱 WhiteningNet、以及横向对比用的 CNNC）在十个任务上的精度，横轴按转速。
  (c) **配对分解**：本方法 − WIDAN 的逐任务 Δ（另叠加对 MRA-SNN 的 Δ），
      并画 ±2.55 点可复现下限带 —— 说明"池化 +2.06 不显著"的构成：
      低速端落后、60 Hz 以上领先，T9 领先最多。

风格与 Fig. 6 / Fig. 8 一致：Agg 后端、**面板编号与图题一律置于坐标轴下方**
（`cap()`，与 `make_fig6_frontend.py` 同规则；不再用 `set_title` 放在上端）、
字号 4.2–6.6、pdf+png @400 dpi。

输出：docs/manuscript/figures/
  · fig10_hust.pdf / .png      （2×2 合成图，(a) 纵跨两行）
  · fig10a_panel.pdf / .png    （难度梯度，独立成图，含下端图题）
  · fig10b_panel.pdf / .png    （逐任务曲线，独立成图，含下端图题）
  · fig10c_panel.pdf / .png    （配对分解，独立成图，含下端图题）
  · _fig10_values.txt

单面板图用显式 `RECT`（而非 `tight_layout`）定位坐标轴，给下端图题留出固定
高度，再用 `bbox_inches='tight'` 导出 —— 拼版时三个面板的图题基线一致。

用法：python analysis/make_fig10_hust.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
TAB = os.path.join(ROOT, 'docs', 'manuscript', '_hust_tables.md')
STATS = os.path.join(ROOT, 'docs', 'manuscript', '_cmp_HUST_stats.md')
os.makedirs(FIGD, exist_ok=True)

TASKS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
FLOOR = 2.55
C_OURS = '#d62728'
C_WID = '#1f77b4'
C_ALT = '#2ca02c'
C_WEAK = '#7f7f7f'
C_OTHER = '#9467bd'
LOG: list[str] = []
log = LOG.append

# (a) 在合成图里纵跨**两行**，轴高约为单行面板的 2.5 倍。若仍用 -0.34 这个
# 轴分数偏移，图题会落到画布外被裁掉；故 (a) 的图题偏移单独参数化，
# 由 main() 在“合成图 / 单面板图”两种场景下分别设定。
A_CAP_Y = -0.34

# (a) 的图题文案：合成图那一格列宽仅 ~2.7 in，长句会越到右边 (c) 的图题区，
# 所以用短语版；单面板图用多行完整版。T9 偏差区间原先是用 annotate 画在轴内的，
# 会**压住 Mixup 的 T9 红条**，而且把 §3.1 引用的两个数重复了一遍 —— 移入图题后
# 既无遮挡，又让图自证（读者不必回正文找区间）。
CAP_A_SHORT = '(a)  difficulty gradient over the speed tasks'
CAP_A_FULL = (CAP_A_SHORT + '\n'
              "      bar = deviation from that method's own 10-task mean\n"
              '      T9 (80 Hz): lowest task for all 21 (-33.1 to -14.7 pts)')
A_CAP_TEXT = CAP_A_SHORT


def cap(ax, text, y=-0.34, size=6.2):
    """面板编号 + 图题置于坐标轴**下方**（与 Fig. 6 / Fig. 8 的 cap() 同规则）。

    用 axes 分数坐标，因此同一段文字在合成图与单面板图里的相对位置一致；
    合成图靠 gridspec 的 hspace/bottom 留白，单面板图靠 bbox_inches='tight'
    把这段文字纳入导出边界。
    """
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.20', linespacing=1.35)


def _num(tok):
    tok = tok.replace('**', '').strip()
    if tok in ('–', '—', '-', '', '...'):
        return float('nan'), float('nan')
    parts = [p.strip() for p in tok.split('±')]
    try:
        m = float(parts[0])
    except ValueError:
        return float('nan'), float('nan')
    s = float('nan')
    if len(parts) > 1:
        try:
            s = float(parts[1])
        except ValueError:
            pass
    return m, s


def parse_table(path):
    """HUST-T1 表 -> OrderedDict(name -> dict(pool, sd_pool, tasks=[(m,s)]*10))."""
    rows, in_t1 = {}, False
    for line in open(path, encoding='utf-8'):
        if line.startswith('## '):
            in_t1 = 'HUST-T1' in line
            continue
        if not (in_t1 and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 14 or c[0].startswith('方法') or set(c[0]) <= set('-: '):
            continue
        name = c[0]
        tasks = [_num(c[2 + i]) for i in range(10)]
        m_pool, s_pool = _num(c[12])
        rows[name] = dict(family=c[1], tasks=tasks, pool=m_pool,
                          sd=s_pool, n=c[13])
    return rows


def parse_stats(path, arm='unif'):
    """`AG-Mamba unif vs 各基线 —— 主口径 last5` 段 -> dict(name -> (Δ, wins, p))。

    ⚠ **必须三重限定**（`_cmp_*_stats.md` 里有 4 个段都含 “`unif`” 与 “ vs ”：
    主口径 `last5`(L12)、`last10`(L118)、`last1`(L168)、`peak`(L218)）。
    只按 “含 unif 且含 vs” 匹配会让**最后**一段（peak）覆盖前面的 —— 实测
    WhiteningNet 的 Δ 会从 **+41.42** 变成 **+35.79**（peak 口径）。故加
    `主口径` 与 `` `last5` `` 两个条件；另：`peak` 口径全文禁用。
    """
    out, on = {}, False
    for line in open(path, encoding='utf-8'):
        if line.startswith('## '):
            on = (' vs ' in line and '`%s`' % arm in line
                  and '主口径' in line and '`last5`' in line)
            continue
        if not (on and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 8 or c[0].startswith('基线') or set(c[0]) <= set('-: '):
            continue
        try:
            d = float(c[4].replace('**', '').replace('+', ''))
        except ValueError:
            continue
        try:
            p = float(c[6])
        except ValueError:
            p = float('nan')
        out[c[0]] = (d, c[5], p)
    return out


def panel_a(ax, tab):
    # **含本方法**：§3.1 的断言是“对全部二十一个方法”。本方法取 `unif`（协议臂）。
    names = [n for n in tab if n != 'AGMamba(pub)']
    dev9 = {n: tab[n]['tasks'][9][0] - tab[n]['pool'] for n in names}
    dev0 = {n: tab[n]['tasks'][0][0] - tab[n]['pool'] for n in names}
    order = sorted(names, key=lambda n: dev9[n])
    y = np.arange(len(order))
    ax.barh(y + 0.19, [dev0[n] for n in order], height=0.36, color=C_WID,
            label='T0 (20 Hz) $-$ own mean')
    ax.barh(y - 0.19, [dev9[n] for n in order], height=0.36, color=C_OURS,
            label='T9 (80 Hz) $-$ own mean')
    ax.set_yticks(y)
    ax.set_yticklabels(['proposed  ' + n if n.startswith('AGMamba') else n
                        for n in order], fontsize=4.2)
    for tick, n in zip(ax.get_yticklabels(), order):
        if n.startswith('AGMamba'):
            tick.set_color(C_OURS)
    ax.axvline(0, color='0.3', lw=0.6)
    ax.set_xlabel('deviation from that method\'s own 10-task mean (points)',
                  fontsize=6.0)
    ax.tick_params(axis='x', labelsize=5.4)
    cap(ax, A_CAP_TEXT, y=A_CAP_Y)
    # 图例搬到中段空白带：第 15–17 行（DGNIS / WIDAN / MRA_SNN）的 T9 条都很短
    # （-8.4 / -0.7 / -1.0），因此 x < -9 整片是空白；原先的 'lower left' 会
    # 压住 DAEL 行的 T0 蓝条。用**数据坐标**锚定，合成图与单面板图位置一致。
    ax.legend(fontsize=4.6, loc='center left', frameon=False,
              bbox_to_anchor=(-32.4, 16.2), bbox_transform=ax.transData)
    lo, hi = min(dev9.values()), max(dev9.values())
    log('## (a) T9 / T0 deviation from own 10-task mean  (n=%d)' % len(order))
    for n in order:
        log('%-14s T9 %+.2f  T0 %+.2f  (pool %.2f)'
            % (n, dev9[n], dev0[n], tab[n]['pool']))
    log('T9 dev range: %.2f .. %.2f ; T0 dev range: %.2f .. %.2f'
        % (lo, hi, min(dev0.values()), max(dev0.values())))


def panel_b(ax, tab, sel):
    x = np.arange(10)
    for i, (name, col, ls) in enumerate(sel):
        t = tab[name]['tasks']
        m = np.array([v[0] for v in t])
        s = np.array([v[1] for v in t])
        ax.errorbar(x, m, yerr=s, fmt='o-', ms=2.0, lw=0.9, color=col, ls=ls,
                    elinewidth=0.5, capsize=1.0, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % s for s in SPEEDS], fontsize=4.6)
    ax.set_xlabel('target speed (Hz)', fontsize=6.0)
    ax.set_ylabel('accuracy (%, pooled over 5 seeds)', fontsize=6.0)
    ax.tick_params(axis='y', labelsize=5.4)
    ax.set_ylim(15, 103)
    cap(ax, '(b)  per-task accuracy')
    ax.legend(fontsize=4.4, loc='lower left', frameon=False)
    log('## (b) per-task curves')
    for name, _, _ in sel:
        log('%-14s %s' % (name, np.array2string(
            np.array([v[0] for v in tab[name]['tasks']]), precision=2)))


def panel_c(ax, tab, stats):
    x = np.arange(10)
    for name, col in (('WIDAN', C_WID), ('MRA_SNN', C_ALT)):
        tm = np.array([v[0] for v in tab['AGMamba(unif)']['tasks']])
        tb = np.array([v[0] for v in tab[name]['tasks']])
        d = tm - tb
        ax.bar(x + (-0.19 if name == 'WIDAN' else 0.19), d, width=0.36,
               color=col, label='proposed $-$ %s' % name)
        log('## (c) per-task delta vs %s' % name)
        log('  %s' % np.array2string(d, precision=2))
    ax.axhspan(-FLOOR, FLOOR, color='0.86', zorder=0)
    ax.axhline(0, color='0.3', lw=0.6)
    # 色带上沿左侧空白处（x=0.5 起）：该处 y∈[2.6, 4.7] 无任何柱（20 Hz 绿柱
    # 只到 5.3 且位于 x<0.37；30 Hz 蓝柱顶 2.2 < 2.6）。旧位置 x=9.4 贴住
    # 80 Hz 的绿柱（高 10.4），文字被柱体遮住。
    ax.text(0.5, 2.95, '±2.55 floor', fontsize=4.4, color='0.3',
            ha='left', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % s for s in SPEEDS], fontsize=4.6)
    ax.set_xlabel('target speed (Hz)', fontsize=6.0)
    ax.set_ylabel('Δ accuracy (points)', fontsize=6.0)
    ax.tick_params(axis='y', labelsize=5.4)
    ax.set_ylim(-4.5, 15.0)
    ax.legend(fontsize=4.6, loc='upper center', frameon=False, ncol=2)
    cap(ax, '(c)  where the pooled +2.06 comes from')


def main():
    tab = parse_table(TAB)
    stats = parse_stats(STATS, 'unif')
    log('# Fig. 10 数值（§6.7 HUST 逐任务与难度梯度）')
    log('')
    log('methods: %d ; paired stats rows: %d' % (len(tab), len(stats)))
    ours = tab['AGMamba(unif)']['pool']
    log('proposed unif pooled %.2f' % ours)
    for n in sorted(stats, key=lambda k: -stats[k][0]):
        log('%-14s Δ %+6.2f  wins %-6s p %s' % (n, stats[n][0], stats[n][1],
                                                stats[n][2]))

    # 图题改放下端后：hspace 要容得下上排 (b) 的单行图题，bottom 要容得下
    # 下排 (c) 的图题。合成图**不用** bbox_inches='tight'，所以必须靠
    # gridspec 边距把图题留在画布内。
    #   行高 H = (top-bottom)/(2+hspace) = 0.795/2.50 = 0.318
    #   图题偏移 0.34*H = 0.108 ；行间隙 hspace*H = 0.159 ✓
    #   下排轴底 0.140 -> 图题顶 0.032，字高 ~0.013 -> 底 0.019 > 0 ✓
    # left=0.13 是给 21 行 y 轴标（最长 "proposed  AGMamba(unif)"）留的位置，
    # 旧的 0.10 会把行首字母切掉。
    global A_CAP_Y, A_CAP_TEXT
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.28, left=0.13,
                          right=0.975, top=0.935, bottom=0.140)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    # (a) 轴高 = 0.795（跨两行），要让图题顶落在与 (c) 相同的 y=0.032：
    #   (0.032 - 0.140) / 0.795 = -0.136
    A_CAP_Y = -0.136
    A_CAP_TEXT = CAP_A_SHORT    # 合成图列宽仅 2.7 in，长图题会越到 (c) 图题区
    panel_a(ax_a, tab)
    panel_b(ax_b, tab, [('AGMamba(unif)', C_OURS, '-'), ('WIDAN', C_WID, '--'),
                        ('MRA_SNN', C_ALT, '-.'), ('CNNC', C_OTHER, ':'),
                        ('WhiteningNet', C_WEAK, (0, (3, 1, 1, 1)))])
    panel_c(ax_c, tab, stats)
    fig.text(0.5, 0.965, 'HUST: the ten speed targets, all 21 methods', ha='center',
             fontsize=6.6)
    fig.savefig(os.path.join(FIGD, 'fig10_hust.pdf'))
    fig.savefig(os.path.join(FIGD, 'fig10_hust.png'), dpi=400)
    plt.close(fig)

    # ---- 单面板导出（便于后续排版拼版）
    # 坐标轴用**显式 rect** 定位而不是 tight_layout：rect 的 bottom 固定为定值，
    # 保证三个面板的图题基线一致；再用 bbox_inches='tight' 把下端图题纳入
    # 导出边界（因此无需在画布内预留图题高度）。
    SEL_B = [('AGMamba(unif)', C_OURS, '-'), ('WIDAN', C_WID, '--'),
             ('MRA_SNN', C_ALT, '-.'), ('CNNC', C_OTHER, ':'),
             ('WhiteningNet', C_WEAK, (0, (3, 1, 1, 1)))]
    A_CAP_Y = -0.34          # 单面板图里 (a) 不再是双倍高，回到默认偏移
    A_CAP_TEXT = CAP_A_FULL  # 单面板图有 3.9 in 宽，用多行完整版（自含 T9 区间）
    for tag, fn, size, rect in (
            ('a', lambda a: panel_a(a, tab), (3.9, 4.6),
             [0.300, 0.26, 0.665, 0.70]),
            ('b', lambda a: panel_b(a, tab, SEL_B), (3.9, 2.7),
             [0.145, 0.30, 0.830, 0.64]),
            ('c', lambda a: panel_c(a, tab, stats), (3.9, 2.7),
             [0.135, 0.30, 0.845, 0.64])):
        f = plt.figure(figsize=size)
        a = f.add_axes(rect)
        fn(a)
        for ext in ('pdf', 'png'):
            f.savefig(os.path.join(FIGD, 'fig10%s_panel.%s' % (tag, ext)),
                      dpi=400, bbox_inches='tight')
        plt.close(f)

    with open(os.path.join(FIGD, '_fig10_values.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    print('wrote fig10_hust.[pdf|png] + fig10a/b/c_panel.*')
    print('  ' + os.path.join(FIGD, '_fig10_values.txt'))


if __name__ == '__main__':
    main()
