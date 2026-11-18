#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 8 —— §6.5 膨胀鲁棒性（受控合成）：前端是否真的"随转速塌陷"。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/figures/_s65_dilation.npz`   ← `analysis/exp_s65_dilation.py`
      冻结交付核组上的：响应漂移 D(s)（真实变速 / 合成膨胀）、同转速自基线
      D_self、逐通道 log2 响应比、以及"纯膨胀 vs 物理变速"的模型隔离行
  · `logs/_s65_bank.npz`                          ← `logs/_s65_accuracy_probe.py`
      同种子同数据序下的 **冻结分析器 vs 可训练分析器** 精度阶梯
      （真实 80 Hz 目标 + 已见转速的未膨胀/膨胀替身集）

四个面板：
  (a) 首层响应漂移 D(s)：真实变速 vs 合成膨胀 vs 同转速自基线
  (b) 逐通道响应比的通道–转速热图（物理 | 合成）—— 看"漂移落在哪些通道"
  (c) 端到端精度阶梯：trained vs frozen（末 5 个 checkpoint 的均值）
      把与真实远目标的差距分解为 **纯膨胀代价** 与 **残余几何差距**
  (d) 模型隔离（冲击序列）：**精确**膨胀 vs 物理变速的 D —— 证明该度量并非
      "看不见膨胀"，而是真实记录被宽带噪声主导

未完成/缺失的数据在图中以斜纹占位 + "running" 标注呈现，绝不画成 0。

风格与 Fig. 6 / Fig. 7 一致：Agg 后端、**面板编号与图题一律置于坐标轴下方**
（`cap()`，与 `make_fig6_frontend.py` 同规则；不再用 `set_title` 放在上端）、
字号 5.5–6.5、pdf+png @400 dpi。

输出：docs/manuscript/figures/
  · fig8_dilation.pdf / .png        （2×2 合成图；四个面板图题各自在下端）
  · fig8a_panel.pdf / .png          （漂移曲线，独立成图，含下端图题）
  · fig8b_panel.pdf / .png          （通道热图，独立成图，含下端图题）
  · fig8c_panel.pdf / .png          （精度阶梯，独立成图，含下端图题）
  · fig8d_panel.pdf / .png          （模型隔离，独立成图，含下端图题）
  · _fig8_values.txt                （数值留档）

单面板图用显式 `RECT`（而非 `tight_layout`）定位坐标轴，给下端图题留出固定
高度，再用 `bbox_inches='tight'` 导出 —— 这样拼版时四个面板的坐标轴框尺寸
与图题基线一致。

用法：python analysis/make_fig8_dilation.py
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
os.makedirs(FIGD, exist_ok=True)

DIL = os.path.join(FIGD, '_s65_dilation.npz')
PROBE = os.path.join(ROOT, 'logs', '_s65_bank.npz')      # 冻结臂（真实目标）
LAD2 = os.path.join(ROOT, 'logs', '_s65_ladder2.npz')    # 带限精确膨胀阶梯

C_PHYS = '#1f77b4'
C_SYN = '#d62728'
C_FROZ = '#7f7f7f'
C_K1 = '#2ca02c'
C_K2 = '#9467bd'

# (b) 面板图题：单面板图用全称版，合成图那格很窄、长句会越界，故用短语版
CAP_B_FULL = ('(b)  per-atom $\\log_2$ response ratio: '
              'physical (left) vs synthetic dilation (right)')
CAP_B_SHORT = '(b)  per-atom $\\log_2$ response ratio (physical vs synthetic)'

# 阶梯集的显示名（与 probe 的 names 顺序一致）
LAD_FMT = {
    'D9(real)': 'real\n80 Hz',
    'D8': '75 Hz\n×1.000',
    'D8x1.0667': '75→80 Hz\n×1.067',
    'D7': '70 Hz\n×1.000',
    'D7x1.1429': '70→80 Hz\n×1.143',
    'D6': '65 Hz\n×1.000',
    'D6x1.2308': '65→80 Hz\n×1.231',
}

LOG: list[str] = []
log = LOG.append


def log_num(name, arr):
    log('%-24s %s' % (name, np.array2string(np.atleast_1d(np.asarray(arr, float)),
                                            precision=4, max_line_width=200)))


def cap(ax, text, y=-0.34, size=6.2):
    """面板编号 + 图题置于坐标轴**下方**（与 Fig. 6 的 cap() 同规则）。

    用 axes 分数坐标，因此同一段文字在合成图与单面板图里的相对位置一致；
    单面板图靠 `bbox_inches='tight'` 把这段文字纳入导出边界。
    """
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.20', linespacing=1.35)


# ------------------------------------------------------------------ 数据
def load():
    d = np.load(DIL)
    out = {k: d[k] for k in d.files}
    if os.path.exists(PROBE):
        p = np.load(PROBE, allow_pickle=True)
        out['lad_probe_names'] = [str(x) for x in p['names']]
        out['ladder_acc'] = p['ladder_acc']
        out['ladder_names'] = [str(x) for x in p['ladder_names']]
    if os.path.exists(LAD2):
        q = np.load(LAD2, allow_pickle=True)
        out['lad2_names'] = [str(x) for x in q['names']]
        out['lad2_acc'] = q['acc']
        out['lad2_meta'] = q['meta']
        out['lad2_train_acc'] = q['train_acc']
    return out


# ------------------------------------------------------------------ 面板
def panel_a(ax, d):
    s = d['s']
    dself = float(d['d_self'][0])
    ax.axhspan(0.0, dself, color='0.88', zorder=0)
    ax.axhline(dself, color='0.45', lw=0.6, ls='-', zorder=1)
    ax.plot(s, d['d_phys'], 'o-', ms=2.6, lw=1.0, color=C_PHYS,
            label='physical speed change')
    ax.plot(s, d['d_synth'], 's--', ms=2.4, lw=1.0, color=C_SYN,
            label='synthetic dilation of the 20 Hz record')
    ax.axvline(3.75, color='0.55', lw=0.5, ls=':', zorder=1)
    ax.axvline(4.0, color='0.55', lw=0.5, ls=':', zorder=1)
    ax.text(3.77, ax.get_ylim()[1] * 0.94, 'far\n75–80 Hz', fontsize=4.6,
            color='0.35', ha='left', va='top')
    # 标注放进灰带（D ≤ 自基线）的**右端**：左端会压在红色虚线上（二者在
    # s≈1.6 处相交）；但也不能推到最右，否则贴着坐标框右边界。
    ax.text(2.80, dself * 0.42, 'within-speed self-baseline  %.4f' % dself,
            fontsize=4.6, color='0.35', va='center', ha='left')
    ax.set_xlabel('speed ratio $s$  (relative to the 20 Hz record)', fontsize=6.0)
    ax.set_ylabel('bank-response drift $D$', fontsize=6.0)
    ax.legend(fontsize=5.0, loc='upper left', frameon=False)
    ax.tick_params(labelsize=5.4)
    cap(ax, '(a)  first-layer response drift')
    log('## (a) D(s)')
    log_num('s', s)
    log_num('D_phys', d['d_phys'])
    log_num('D_synth', d['d_synth'])
    log_num('D_self', [dself])


def _heat(ax, M, speeds, vmax, show_y):
    """单个热图。**不设标题** —— 面板编号与图题由调用方用 `cap()` 放在下端。"""
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', origin='lower')
    ax.set_xticks([0, 31, 63])
    ax.tick_params(labelsize=4.8)
    ax.set_xlabel('bank atom index', fontsize=5.6)
    if show_y:
        ax.set_yticks(range(len(speeds)))
        ax.set_yticklabels(['%d' % v for v in speeds], fontsize=4.8)
        ax.set_ylabel('speed (Hz)', fontsize=5.6)
    else:
        ax.set_yticks([])
    return im


def panel_b(axes, d, caption=CAP_B_FULL):
    """(b) 通道热图。

    `caption` 可覆盖：合成图里该格很窄，长句会越过画布右边界，因此合成图传
    短语版 `CAP_B_SHORT`；单面板图（5.0 in 宽）用全称版。
    """
    speeds = d['speeds']
    mp, ms = d['chan_phys'], d['chan_synth']
    vmax = float(np.percentile(np.abs(np.concatenate([mp.ravel(), ms.ravel()])),
                               99.0))
    vmax = max(vmax, 0.1)
    im = _heat(axes[0], mp, speeds, vmax, True)
    _heat(axes[1], ms, speeds, vmax, False)
    cb = axes[1].figure.colorbar(im, ax=list(axes), fraction=0.032, pad=0.02)
    cb.ax.tick_params(labelsize=4.6)
    cb.set_label(r'$\log_2$ channel response ratio', fontsize=5.0)
    cap(axes[0], caption)
    log('## (b) chan heatmap  vmax(99pct)=%.3f' % vmax)
    for i, sp in enumerate(speeds):
        log('%-4d  phys %s' % (sp, np.array2string(mp[i], precision=3,
                                                   max_line_width=400)))
        log('%-4d  syn  %s' % (sp, np.array2string(ms[i], precision=3,
                                                   max_line_width=400)))


def panel_c(ax, d):
    nm = d['lad2_names']
    a = d['lad2_acc']

    def v(k):
        return a[nm.index(k)] if k in nm else float('nan')

    groups = [('75 Hz', 'raw_D8', 'D8_ideal_r1.0667', 'D8_lerp_1.0667'),
              ('70 Hz', 'raw_D7', 'D7_ideal_r1.1429', 'D7_lerp_1.1429'),
              ('65 Hz', 'raw_D6', 'D6_ideal_r1.2308', 'D6_lerp_1.2308'),
              ('real', 'D9_real_80Hz', None, None)]
    x = np.arange(len(groups))
    W = 0.26
    for i, (lab, kr, ki, kl) in enumerate(groups):
        ax.bar(i - W, v(kr), width=W, color='0.72',
               label='source record (no dilation)' if i == 0 else None)
        if ki is not None:
            ax.bar(i, v(ki), width=W, color=C_PHYS,
                   label='band-limited dilation to 80 Hz' if i == 0 else None)
            ax.bar(i + W, v(kl), width=W, color=C_SYN, hatch='///',
                   edgecolor='white', lw=0.0,
                   label='linear-interp. (1st version)' if i == 0 else None)
            ax.annotate('', xy=(i, v(ki)), xytext=(i - W, v(kr)),
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='0.3',
                                        connectionstyle='arc3,rad=-0.3'))
        else:
            ax.bar(i, v(kr), width=W, color='0.72', edgecolor=C_SYN,
                   lw=0.9, label='real 80 Hz target' if i == 0 else None)
    # 冻结臂在真实目标上的参考线（来自第一版探针的 T9 真实目标评估）
    rf = float('nan')
    if 'ladder_acc' in d:
        li = d['ladder_names'].index('frozen')
        pn = d['lad_probe_names']
        rf = d['ladder_acc'][li][pn.index('D9(real)')]
    if rf == rf:
        ax.axhline(rf, color=C_FROZ, lw=0.8, ls='--')
        ax.text(len(groups) - 1 + 0.42, rf + 1.5, 'frozen\n%.1f' % rf,
                fontsize=4.4, color=C_FROZ, ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.0)
    ax.set_ylabel('target accuracy (%)', fontsize=6.0)
    ax.tick_params(axis='y', labelsize=5.4)
    ax.set_ylim(0, 108)
    # 图例放**右上**：左上/右下都会压住 75 Hz 柱或 "real" 柱；
    # 右上在真实目标柱顶（≈78）上方，是唯一空区。
    ax.legend(fontsize=4.2, loc='upper right', frameon=False, ncol=1, borderpad=0.2)
    cap(ax, '(c)  end-to-end accuracy ladder, final weights')
    log('## (c) accuracy ladder (final weights, seed 0)')
    for k in nm:
        log('%-20s %.2f' % (k, v(k)))
    log('frozen real-target = %.2f' % rf)


def panel_d(ax, d):
    s, ks = d['mdl_s'], d['mdl_kappa']
    dil, ph = d['mdl_dil'], d['mdl_phys']
    dself = float(d['d_self'][0])
    ax.axhline(dself, color='0.5', lw=0.6, ls=':', zorder=0)
    ax.text(1.27, dself * 1.15, 'real within-speed baseline', fontsize=4.4,
            color='0.4', va='bottom')
    for kap, col in zip([ks[0], ks[-1]], [C_K1, C_K2]):
        m = ks == kap
        ax.plot(s[m], dil[m], 'o-', ms=2.4, lw=1.0, color=col,
                label=r'exact, $\kappa$=%.2f' % kap)
        ax.plot(s[m], ph[m], '^--', ms=2.2, lw=0.9, color=col, alpha=0.65,
                label=r'physical, $\kappa$=%.2f' % kap)
    ax.set_yscale('log')
    ax.set_xlabel('speed ratio $s$', fontsize=6.0)
    ax.set_ylabel('bank-response drift $D$', fontsize=6.0)
    ax.tick_params(labelsize=5.4)
    # 图例避开数据：这个面板四角都有曲线，唯一空带是"精确膨胀平台(≳0.19)"与
    # "物理变速虚线(≲0.06)"之间。故用**两列短标签**把图例压成两行，锚在
    # axes 上沿略下方（bbox y=0.88）——四行单列版本会占满整个空带并压住虚线。
    ax.legend(fontsize=4.2, loc='upper left', bbox_to_anchor=(0.0, 0.88),
              frameon=False, ncol=2, borderpad=0.15, columnspacing=0.8,
              handletextpad=0.4)
    cap(ax, '(d)  impulse-train control')
    log('## (d) impulse-train model (kappa, s, D_dil, D_phys)')
    for a, b, c, e in zip(ks, s, dil, ph):
        log('kappa=%.2f  s=%.2f  D_dil=%.4f  D_phys=%.5f' % (a, b, c, e))


# ------------------------------------------------------------------ 主
def main():
    d = load()
    log('# Fig. 8 数值（§6.5 膨胀鲁棒性）')
    log('来源：_s65_dilation.npz（冻结核组度量）+ _s65_bank.npz（训练侧探针）')
    log('')

    # ---- 合成图。（hspace / bottom 都比"图题在上端"时留得更大：
    #      第一行两面板的图题要落在行间空隙里，第二行的图题要落在图外下方。）
    fig = plt.figure(figsize=(7.4, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.34,
                          left=0.075, right=0.965, top=0.93, bottom=0.17)
    ax_a = fig.add_subplot(gs[0, 0])
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.10)
    ax_b1 = fig.add_subplot(sub[0, 0])
    ax_b2 = fig.add_subplot(sub[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, d)
    panel_b([ax_b1, ax_b2], d, CAP_B_SHORT)
    panel_c(ax_c, d)
    panel_d(ax_d, d)

    fig.text(0.5, 0.965, 'A pure dilation of a seen-speed record reproduces the '
             'far-target loss (and exceeds it); freezing the analyser makes it '
             'worse, not better', ha='center', fontsize=6.6)
    fig.savefig(os.path.join(FIGD, 'fig8_dilation.pdf'))
    fig.savefig(os.path.join(FIGD, 'fig8_dilation.png'), dpi=400)
    plt.close(fig)

    # ---- 单面板导出（便于后续排版拼版）
    # 坐标轴用**显式 rect** 定位而不是 tight_layout：rect 的 bottom 固定为 0.32，
    # 保证四个面板的坐标轴框等高、且下端图题的基线一致；再用
    # bbox_inches='tight' 把图题纳入导出边界。
    RECT = [0.145, 0.32, 0.82, 0.60]          # [left, bottom, width, height]
    for tag, fn, size, rect in (('a', panel_a, (3.7, 3.0), RECT),
                                ('c', panel_c, (3.9, 3.0),
                                 [0.135, 0.32, 0.845, 0.60]),
                                ('d', panel_d, (3.7, 3.0),
                                 [0.155, 0.32, 0.81, 0.60])):
        f = plt.figure(figsize=size)
        a = f.add_axes(rect)
        fn(a, d)
        for ext in ('pdf', 'png'):
            f.savefig(os.path.join(FIGD, 'fig8%s_panel.%s' % (tag, ext)),
                      dpi=400, bbox_inches='tight')
        plt.close(f)

    f = plt.figure(figsize=(5.0, 3.0))
    sb = f.add_gridspec(1, 2, wspace=0.10, left=0.10, right=0.88, bottom=0.32,
                        top=0.92)
    a1, a2 = f.add_subplot(sb[0, 0]), f.add_subplot(sb[0, 1])
    panel_b([a1, a2], d)
    for ext in ('pdf', 'png'):
        f.savefig(os.path.join(FIGD, 'fig8b_panel.' + ext), dpi=400,
                  bbox_inches='tight')
    plt.close(f)

    with open(os.path.join(FIGD, '_fig8_values.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    print('wrote fig8_dilation.[pdf|png] + fig8a/b/c/d_panel.*')
    print('  ' + os.path.join(FIGD, '_fig8_values.txt'))


if __name__ == '__main__':
    main()
