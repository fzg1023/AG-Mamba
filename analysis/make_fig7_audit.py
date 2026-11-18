#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 7 —— §6.4 机制审计汇总。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/_s64_align.md`    的 (a1)/(a2) 表 → 对齐四臂
  · `docs/manuscript/_s65_tau_ssm.md`  的 (a1)/(a2) 表 → τ 三臂 + SSM on/off

三个面板（与 Fig. 6 同一视觉语法，便于并列阅读）：
  (a) 对齐四臂：T0/T4/T9 分组柱 + 池化 `last5` 黑横线 + 参考臂点线
  (b) τ / SSM 四臂：同上
  (c) 判决面板：六个消融的**池化 Δ** 横条 + ±2.55 点可复现下限带
      —— 直接呈现"什么存活、什么不能判定"这条判定规则

⚠ 解析要点：两个文件的 **(a2) 配对检验表列数也是 9**，与 (a1) 相同 →
**必须限定在 `## (a1)` 段**，否则会把 Δ 当精度（Fig. 6 踩过同样的坑）。

输出：docs/manuscript/figures/
  · fig7_audit.pdf / .png         （合成图，三面板）
  · fig7a_panel.pdf / .png        （对齐四臂）
  · fig7b_panel.pdf / .png        （τ / SSM 四臂）
  · fig7c_panel.pdf / .png        （Δ 与下限）
  · _fig7_values.txt              （数值留档）

用法：python analysis/make_fig7_audit.py
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
FIGD = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(FIGD, exist_ok=True)

FLOOR = 2.55          # §5.3 同配置重跑 mean|Δ|（点）
NA = float('nan')
LOG: list[str] = []


# ------------------------------------------------------------------ 解析
def _num(tok):
    """'**89.00**' -> (89.0, nan) ; '94.15 ± 0.25' -> (94.15, 0.25) ; '–' -> (nan, nan)."""
    tok = tok.replace('*', '').strip()
    if tok in ('–', '—', '-', '', '...', '✓'):
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


def parse_md(path):
    """返回 (arms, stats)：

    arms[arm]  = {'T0','T4','T9': (mean, std), 'pool': float}
    stats[arm] = {'d': float, 'p': float, 'wins': str}
    仅取 `## (a1)` 段做 arms、`## (a2)` 段做 stats。
    """
    arms, stats, sec = {}, {}, None
    for line in open(path, encoding='utf-8'):
        if line.startswith('#'):
            sec = line.lstrip('#').strip()[:4]
            continue
        if not line.startswith('|') or sec not in ('(a1)', '(a2)'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        # (a1) 表列数**不固定**：对齐表 9 列（含 w_mmd/w_adv），τ/SSM 表 8 列（含"说明"）
        # → 一律取**最后 6 格**，即 T0 / T4 / T9 / pool(last5) / pool(last10) / pool(last1)
        if sec == '(a1)':
            if len(c) not in (8, 9):
                continue
            nums = c[-6:]
        elif sec == '(a2)':
            if len(c) != 9:
                continue
        else:
            continue
        # 臂名清洗：先去掉 markdown 强调/反引号与中文括号补充说明
        name = re.sub(r'[`*]', '', c[0])
        name = re.sub(r'[（(].*?[）)]', '', name).strip()
        if not name or name.startswith('臂') or name.startswith('Arm'):
            continue
        if sec == '(a1)':
            vals = [_num(x) for x in nums]
            if any(v[0] != v[0] for v in vals):
                continue
            arms[name] = {'T0': vals[0], 'T4': vals[1], 'T9': vals[2],
                          'pool': vals[3][0]}
        else:
            d = _num(c[2])[0]
            if d != d:
                continue
            stats[name] = {'d': d, 'p': _num(c[-3])[0], 'wins': c[-2]}
    return arms, stats


ALIGN = os.path.join(ROOT, 'docs', 'manuscript', '_s64_align.md')
TAU = os.path.join(ROOT, 'docs', 'manuscript', '_s65_tau_ssm.md')

# 显示名（图内用短名，图注给全名）
LAB = {
    'aln_noall': 'proposed\n(no alignment)',
    'aln_nommd': 'adversarial\nonly',
    'aln_noadv': 'MMD\nonly',
    'aln_full': 'MMD +\nadversarial',
    'tau_zero': 'proposed\n($\\tau$ off)',
    'tau_normal': '$\\tau$ on',
    'tau_shuffle': '$\\tau$ shuffled',
    'ssm_off': 'no selective\nSSM',
}


# ------------------------------------------------------------------ 通用
def tidy(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.0)


def cap(ax, text, y=-0.30, size=5.6):
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.25', linespacing=1.35)


TASK_COLS = ['#1f77b4', '#7f7f7f', '#d62728']


def draw_arms(ax, arms, order, ref, base=65.0):
    """分组柱：每组三根（T0/T4/T9）+ 池化黑横线；ref 为参考臂（画点线）。"""
    grp_w, bar_w = 0.80, 0.24
    for i, arm in enumerate(order):
        d = arms.get(arm)
        if d is None:
            continue
        for j, t in enumerate(('T0', 'T4', 'T9')):
            v, s = d[t]
            if v != v:
                continue
            ax.bar(i + (j - 1) * bar_w, v - base, bottom=base,
                   width=bar_w * 0.86, color=TASK_COLS[j], alpha=0.92, lw=0)
            if s == s:
                ax.errorbar(i + (j - 1) * bar_w, v, yerr=s, fmt='none',
                            ecolor='0.2', elinewidth=0.7, capsize=1.5)
        if d['pool'] == d['pool']:
            ax.plot([i], [d['pool']], 'k_', ms=9, mew=1.3, zorder=5)
            tag = '%.2f' % d['pool']
            if arm != ref and arm in STATS_ALL:
                tag += '\n(%+.2f)' % STATS_ALL[arm]['d']
            # 池化标签放在黑横线**下方**：上方会被同组的 T0/T4 柱与误差棒压住
            ax.text(i, d['pool'] - 1.5, tag, fontsize=5.2, ha='center',
                    va='top', color='0.1', linespacing=1.15)
    refp = arms[ref]['pool']
    ax.axhline(refp, color='0.45', lw=0.7, ls=':')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LAB.get(a, a) for a in order], fontsize=5.0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(base, 106)
    ax.set_ylabel('target accuracy (%)', fontsize=6.5)
    tidy(ax)


# ------------------------------------------------------------------ 面板
def draw_a(ax, A):
    draw_arms(ax, A, ['aln_noall', 'aln_nommd', 'aln_noadv', 'aln_full'],
              'aln_noall')
def draw_b(ax, T):
    draw_arms(ax, T, ['tau_zero', 'tau_normal', 'tau_shuffle', 'ssm_off'],
              'tau_zero')
def draw_c(ax, rows):
    """Δ 横条 + ±FLOOR 带。rows = [(label, Δ, p, wins)]，按 |Δ| 排序。"""
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(-FLOOR, FLOOR, color='0.87', lw=0, zorder=0)
    for y, (lab, d, p, w) in zip(ys, rows):
        col = '#d62728' if d < -FLOOR else '0.55'
        ax.barh(y, d, height=0.55, color=col, alpha=0.92, lw=0)
        ax.text(d + (0.18 if d < 0 else -0.18), y,
                '%.2f' % d + ('  n.s.' if abs(d) < FLOOR else ''),
                fontsize=5.2, va='center',
                ha='left' if d < 0 else 'right', color='0.15')
    ax.axvline(0, color='0.3', lw=0.8)
    # 下限文字放在最上一行之上（仍在灰带内），避免与任何条形/标签相撞
    ax.set_ylim(-0.65, len(rows) + 0.55)
    ax.text(-FLOOR + 0.12, len(rows) + 0.12, '$\\pm$2.55 floor', fontsize=5.0,
            color='0.40', ha='left', va='center')
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    ax.set_xlim(-5.2, 0.9)
    ax.set_xlabel('pooled $\\Delta$ vs proposed (points, last-5)', fontsize=6.5)
    tidy(ax)


def main():
    AL, AS = parse_md(ALIGN)
    TL, TS = parse_md(TAU)
    global STATS_ALL
    STATS_ALL = dict(AS, **TS)
    assert 'aln_noall' in AL and 'aln_full' in AL, '对齐表解析异常'
    assert 'tau_zero' in TL and 'ssm_off' in TL, 'τ/SSM 表解析异常'
    # 两阶段的参考臂必须一致（同源性自检的下游体现）
    assert abs(AL['aln_noall']['pool'] - TL['tau_zero']['pool']) < 1e-9, \
        '两阶段参考臂池化值不一致：%.2f vs %.2f' % (AL['aln_noall']['pool'],
                                                TL['tau_zero']['pool'])

    rows = []
    for key, name in (('aln_nommd', 'adversarial only'),
                      ('aln_noadv', 'MMD only'),
                      ('aln_full', 'MMD + adversarial'),
                      ('tau_normal', r'$\tau$ modulation on'),
                      ('tau_shuffle', r'$\tau$ shuffled'),
                      ('ssm_off', 'no selective SSM')):
        s = STATS_ALL[key]
        rows.append((name, s['d'], s['p'], s['wins']))
    rows.sort(key=lambda r: r[1])

    CAP_A = ('(a) Alignment arms (four loss-term ablations; strictly paired).\n'
             'Bars: T0/T4/T9 with seed spread; black dash: pooled last-5\n'
             'with $\\Delta$ vs the proposed model; dotted: proposed arm.\n'
             'Bars drawn from 65 %.')
    CAP_B = ('(b) $\\tau$ modulation and state-space arms (architectural,\n'
             'shared (task, seed) split). Enabling or shuffling $\\tau_0$ is\n'
             'inert; removing the selective SSM branch is not.')
    CAP_C = ('(c) The decision rule of Section 5.3 applied to every arm:\n'
             'pooled $\\Delta$ against the proposed model, with the $\\pm$2.55-point\n'
             'reproducibility floor shaded. Effects inside the band are not read.')

    PANELS = [('a', lambda ax: draw_a(ax, AL), CAP_A, (3.9, 3.4)),
              ('b', lambda ax: draw_b(ax, TL), CAP_B, (3.9, 3.4)),
              ('c', lambda ax: draw_c(ax, rows), CAP_C, (3.5, 3.4))]

    # ---- 输出 1：合成图 ----
    fig = plt.figure(figsize=(10.6, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.32, width_ratios=[1.0, 1.0, 0.92])
    for i, (_, fn, caption, _) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        fn(ax)
        cap(ax, caption, y=-0.34)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGD, 'fig7_audit.' + ext), dpi=400,
                    bbox_inches='tight')
    plt.close(fig)

    # ---- 输出 2：单面板 ----
    RECT = [0.14, 0.40, 0.83, 0.55]
    for key, fn, caption, size in PANELS:
        f1 = plt.figure(figsize=size)
        ax = f1.add_axes(RECT)
        fn(ax)
        ax.text(0.0, -0.235, caption, transform=ax.transAxes, ha='left',
                va='top', fontsize=6.3, color='0.25', linespacing=1.35)
        for ext in ('pdf', 'png'):
            f1.savefig(os.path.join(FIGD, 'fig7' + key + '_panel.' + ext),
                       dpi=400, bbox_inches='tight')
        plt.close(f1)

    # ---- 数值留档 ----
    with open(os.path.join(FIGD, '_fig7_values.txt'), 'w', encoding='utf-8') as fh:
        fh.write('# Fig. 7 数值（由 analysis/make_fig7_audit.py 从 '
                 '_s64_align.md / _s65_tau_ssm.md 解析）\n')
        fh.write('# 参考臂池化 last5 = %.2f（两阶段一致）| floor = %.2f\n'
                 % (AL['aln_noall']['pool'], FLOOR))
        for src, arms, st in (('align', AL, AS), ('tau/ssm', TL, TS)):
            for k, v in arms.items():
                fh.write('%-6s %-12s T0=%.2f T4=%.2f T9=%.2f pool=%.2f'
                         % (src, k, v['T0'][0], v['T4'][0], v['T9'][0], v['pool']))
                if k in st:
                    fh.write('  d=%+.2f p=%.4g wins=%s' % (st[k]['d'], st[k]['p'],
                                                           st[k]['wins']))
                fh.write('\n')
    LOG.extend('%s %s' % (r[0], '%.2f' % r[1]) for r in rows)
    print('# Fig. 7 written: figures/fig7_audit.* + fig7a/b/c_panel.*')
    for l in LOG:
        print(l)


STATS_ALL: dict = {}

if __name__ == '__main__':
    main()
