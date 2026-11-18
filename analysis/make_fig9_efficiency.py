#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 9 —— §6.6 效率（21 方法：参数量 / 单窗 FLOPs / 推理时延）。

数据源（**单一事实来源**，本脚本只读不写）：
  · `docs/manuscript/_s66_efficiency.md` 的 **(b)** 表 → 21 方法的参数 / FLOPs / 时延
  · 同文件的 **(c)** 表 → SSM 分支开关的时延与参数占比
  · 同文件的 **(d)** 表 → 两种 SSM 后端（官方 CUDA 内核 vs 纯 PyTorch 扫描）的时延

⚠ **`analysis/make_fig8.py` 是 v0.2 的旧脚本**（原子三对照 = 旧 Fig. 9，内容已并入
Fig. 4 的面板 (a)–(c)），与本脚本**无关**，勿混用；其产物 `figures/fig8.pdf/.png`
为遗留文件。

四个面板（**面板图题一律置于坐标轴下方**，与 Fig. 6/8/10/11/12 同规则）：
  (a) 单窗 FLOPs vs 中位时延（双对数散点，21 方法；本方法高亮）
  (b) 中位时延排序横条（对数轴；标注参数量），本方法高亮
  (c) 左：两种 SSM 后端的中位时延（官方内核 / 纯 PyTorch 扫描，后者是主对比的
      确定性折衷）；右：SSM 分支开关的时延对比（描述性）
  (d) SNR 噪声网格（训练干净 / 测试加噪，§6.6 的第二半）—— 数据落地后自动画；
      缺 `_s66_snr.md` 时退化为声明性占位文本

输出：docs/manuscript/figures/
  · fig9_efficiency.pdf / .png        （四面板合成图）
  · fig9a_panel.* / fig9b_panel.* / fig9c_panel.* / fig9d_panel.*（单面板，便于拼版）
  · _fig9_values.txt
用法：python analysis/make_fig9_efficiency.py
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
SRC = os.path.join(ROOT, 'docs', 'manuscript', '_s66_efficiency.md')
os.makedirs(FIGD, exist_ok=True)

OURS = 'AGMamba'
C_OURS = '#d62728'
C_BASE = '#1f77b4'
C_ALT = '#555555'
LOG: list[str] = []
log = LOG.append


def cap(ax, text, y=-0.34, size=6.2):
    """面板编号 + 图题置于坐标轴**下方**（与 Fig. 6/8/10/11/12 的 cap() 同规则）。

    用 axes 分数坐标，故同一段文字在合成图与单面板图里的相对位置一致；
    合成图靠 gridspec 的 hspace/bottom 留白，单面板图靠 bbox_inches='tight'
    把这段文字纳入导出边界。
    """
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.20', linespacing=1.35)


def _f(tok):
    tok = tok.replace('**', '').replace(',', '').strip()
    try:
        return float(tok)
    except ValueError:
        return float('nan')


def parse_b(path):
    """(b) 表 -> list of dicts。列：方法|入口|参数量|FLOPs|中位|最小|p90|吞吐。"""
    out, in_b = [], False
    for line in open(path, encoding='utf-8'):
        if line.startswith('##'):
            in_b = line.lstrip('#').strip().startswith('(b)')
            continue
        if not (in_b and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 8 or c[0].startswith('方法') or set(c[0]) <= set('-: '):
            continue
        out.append(dict(name=c[0].replace('`', ''), entry=c[1],
                        params=_f(c[2]), flops=_f(c[3]), med=_f(c[4]),
                        p90=_f(c[6]), thr=_f(c[7])))
    return out


def parse_two_col(path, sec):
    """(c)/(d) 表 -> dict(name -> [values...])，名字取与方法匹配的关键词。"""
    rows, in_s = [], False
    for line in open(path, encoding='utf-8'):
        if line.startswith('##'):
            in_s = line.lstrip('#').strip().startswith(sec)
            continue
        if not (in_s and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) < 3 or c[0].startswith('配置') or c[0].startswith('后端') \
                or set(c[0]) <= set('-: '):
            continue
        rows.append(c)
    return rows


def panel_a(ax, d):
    for r in d:
        ours = r['name'] == OURS
        ax.scatter(r['flops'], r['med'], s=26 if ours else 14,
                   marker='*' if ours else 'o',
                   color=C_OURS if ours else C_BASE,
                   edgecolor='k' if ours else 'none', linewidth=0.4,
                   zorder=5 if ours else 3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('single-window FLOPs (M)', fontsize=6.0)
    ax.set_ylabel('median latency (ms/window)', fontsize=6.0)
    ax.tick_params(labelsize=5.4)
    cap(ax, '(a)  single-window FLOPs against median latency')
    r = [x for x in d if x['name'] == OURS][0]
    ax.annotate('proposed\n%.2f M params\n%.0f MFLOPs, %.2f ms'
                % (r['params'] / 1e6, r['flops'], r['med']),
                xy=(r['flops'], r['med']), xytext=(-70, 26),
                textcoords='offset points', fontsize=4.6, color=C_OURS,
                arrowprops=dict(arrowstyle='->', lw=0.5, color=C_OURS))
    ax.grid(alpha=0.18, lw=0.4)


def panel_b(ax, d):
    ds = sorted(d, key=lambda r: r['med'])
    y = np.arange(len(ds))
    ax.barh(y, [r['med'] for r in ds], height=0.72,
            color=[C_OURS if r['name'] == OURS else
                   (C_ALT if r['med'] > 10 else C_BASE) for r in ds])
    ax.set_yticks(y)
    ax.set_yticklabels(['%s  (%.2f M)' % (r['name'], r['params'] / 1e6)
                        for r in ds], fontsize=4.4)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('median latency (ms/window, batch 64, eval/no-grad)', fontsize=6.0)
    ax.tick_params(axis='x', labelsize=5.4)
    ax.set_xlim(0.4, 40)
    cap(ax, '(b)  median latency, sorted (parameter count in brackets)')
    for i, r in enumerate(ds):
        if r['name'] == OURS:
            ax.text(r['med'] * 1.12, i, 'proposed  %.2f ms  (%.0f windows/s)'
                    % (r['med'], r['thr']), fontsize=4.4, va='center',
                    color=C_OURS)


def panel_c(ax, rows_c, rows_d):
    """左：两种 SSM 后端（中位数取第 6 列，因第 4 列是多次测量的整串）；
    右：SSM 分支开关（中位数第 3 列）。"""
    names, vals, cols = [], [], []
    for c in rows_d:                      # official / py_scan
        v = _f(c[5]) if len(c) > 5 else float('nan')
        if v != v:                        # 退化：取串中最后一个可解析数
            nums = re.findall(r'[0-9]+\.[0-9]+', ' '.join(c))
            v = float(nums[-1]) if nums else float('nan')
        names.append(c[0] + '\n(backend)')
        vals.append(v)
        cols.append(C_OURS if c[0].startswith('official') else C_ALT)
    for c in rows_c:                      # use_ssm True / False
        names.append(c[0].replace('`', '').replace('（交付）', '\n(delivered)')
                     .replace('use_ssm=', 'SSM ').replace('True', 'on')
                     .replace('False', 'off'))
        vals.append(_f(c[2]))
        cols.append(C_OURS if ('on' in c[0] and 'delivered' in c[0]) else C_BASE)
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=4.4)
    ax.set_ylabel('median latency (ms/window)', fontsize=6.0)
    ax.tick_params(axis='y', labelsize=5.4)
    cap(ax, '(c)  scan backend, and the state-space branch on/off')
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.4, '%.2f' % v, fontsize=4.4, ha='center')
    ax.set_ylim(0, max(vals) * 1.22)
    log('## (c) backends / SSM share')
    for n, v in zip(names, vals):
        log('%-28s %.3f ms' % (n.replace('\n', ' '), v))


SNR_FILE = os.path.join(ROOT, 'docs', 'manuscript', '_s66_snr.md')
SNR_XLAB = ['clean', '10', '5', '0', '-5']
SNR_COL = {'AGMamba': '#d62728', 'WIDAN': '#1f77b4', 'MRA_SNN': '#2ca02c'}
SNR_LAB = {'AGMamba': 'proposed', 'WIDAN': 'WIDAN', 'MRA_SNN': 'MRA-SNN'}


def parse_snr(path):
    """`_s66_snr.md` 的 (b) 表 -> {method: np.array(5)}（已跨三个目标池化）。

    (b) 表的第 3–7 列每格形如 `94.13 ± 0.75 / 93.16 ± 0.87 / …`（5 个数值用 ' / '
    分隔，见 summary_snr.py 的注释：裸 `|` 会冲开表列）→ 取每段的均值部分。
    """
    if not os.path.exists(path):
        return None
    per, in_b = {}, False
    for line in open(path, encoding='utf-8'):
        if line.startswith('##'):
            in_b = line.lstrip('#').strip().startswith('(b)')
            continue
        if not (in_b and line.startswith('|')):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) != 8 or c[0].startswith('方法') or set(c[0]) <= set('-: '):
            continue
        name = c[0].strip('`')
        vals = []
        for cell in c[2:7]:
            tok = cell.split('/')[0].strip()
            try:
                vals.append(float(tok.split('±')[0].strip()))
            except ValueError:
                vals.append(float('nan'))
        if all(v != v for v in vals):
            continue
        per.setdefault(name, {})[c[1]] = np.array(vals)
    out = {}
    for k, d in per.items():
        out[k] = np.nanmean(np.stack(list(d.values())), axis=0)
    return out or None


def panel_d(ax, snr):
    if not snr:
        ax.axis('off')
        cap(ax, '(d)  noise robustness - pending', y=-0.02)
        ax.text(0.02, 0.70, 'SNR noise grid: not yet run.\n\n'
                 'Design: additive white Gaussian noise\n'
                 'at SNR of clean / 10 / 5 / 0 / -5 dB\n'
                 '(per window, relative to its RMS),\n'
                 'targets T0 / T4 / T9, proposed model\n'
                 'plus the two strongest baselines per\n'
                 'benchmark, seeds 0-2.\n\n'
                 'Panel (d) and Table 8 will be extended\nwhen it lands.',
                 fontsize=5.0, va='top', ha='left', color='0.25')
        return
    x = np.arange(5)
    for m, col in SNR_COL.items():
        if m in snr:
            ax.plot(x, snr[m], 'o-', ms=2.8, lw=1.0, color=col,
                    label=SNR_LAB[m])
            log('%-9s %s' % (m, np.round(snr[m], 2)))
    # 9 类随机水平（HUST）= 11.1%：用来读出「基线在 0 dB 已到随机线」这件事
    ax.axhline(100.0 / 9.0, color='0.55', lw=0.7, ls=':')
    ax.text(0.05, 100.0 / 9.0 + 1.6, 'chance (9 classes)', fontsize=4.2,
            color='0.45', ha='left', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(SNR_XLAB, fontsize=4.8)
    ax.set_xlabel('test-time SNR (dB, per-window AWGN)', fontsize=5.6)
    ax.set_ylabel('accuracy (%), pooled over T0/T4/T9', fontsize=5.6)
    ax.tick_params(axis='y', labelsize=5.4)
    # 图例放右上（该区在两条基线曲线之上、全空）；放左下会压住“chance”标注
    ax.legend(fontsize=4.4, frameon=False, loc='upper right')
    cap(ax, '(d)  noise robustness: accuracy against test-time SNR\n'
            '      (per-window AWGN, training clean; dotted line = 9-class chance)',
        y=-0.30)


def main():
    d = parse_b(SRC)
    rows_c = parse_two_col(SRC, '(c)')
    rows_d = parse_two_col(SRC, '(d)')
    snr = parse_snr(SNR_FILE)
    log('# Fig. 9 数值（§6.6 效率鲁棒性）')
    log('')
    for r in sorted(d, key=lambda r: r['med']):
        log('%-12s %-8s %9d %8.1f %6.2f %6.2f %9.0f'
            % (r['name'], r['entry'], int(r['params']), r['flops'], r['med'],
               r['p90'], r['thr']))

    # 图题改放下端后必须留白（合成图**不用** bbox_inches='tight'）：
    #   H = (top-bottom)/(2+hspace) = (0.925-0.155)/2.62 = 0.294
    #   图题偏移 0.34*H = 0.100 ；行间隙 hspace*H = 0.182 ✓
    #   下排轴底 0.155 -> 图题顶 0.055；(d) 两行字高 ~0.042 -> 底 0.013 > 0 ✓
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30, left=0.085,
                          right=0.975, top=0.925, bottom=0.155)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c(ax_c, rows_c, rows_d)
    panel_d(ax_d, snr)
    fig.text(0.5, 0.965, 'Efficiency of the proposed model against the twenty '
             'baselines (HUST window, batch 64, eval/no-grad)', ha='center',
             fontsize=6.6)
    fig.savefig(os.path.join(FIGD, 'fig9_efficiency.pdf'))
    fig.savefig(os.path.join(FIGD, 'fig9_efficiency.png'), dpi=400)
    plt.close(fig)

    # ---- 单面板导出（便于后续排版拼版）
    # 坐标轴用**显式 rect** 定位（不用 tight_layout），再用 bbox_inches='tight'
    # 把下端图题纳入导出边界 —— 拼版时四张图的图题基线一致。
    for tag, fn, size, rect in (
            ('a', lambda ax: panel_a(ax, d), (3.8, 3.0),
             [0.165, 0.31, 0.805, 0.60]),
            ('b', lambda ax: panel_b(ax, d), (4.4, 3.8),
             [0.225, 0.27, 0.755, 0.66]),
            ('c', lambda ax: panel_c(ax, rows_c, rows_d), (4.0, 3.0),
             [0.125, 0.31, 0.855, 0.60]),
            ('d', lambda ax: panel_d(ax, snr), (4.0, 3.2),
             [0.125, 0.31, 0.855, 0.60])):
        f = plt.figure(figsize=size)
        a = f.add_axes(rect)
        fn(a)
        for ext in ('pdf', 'png'):
            f.savefig(os.path.join(FIGD, 'fig9%s_panel.%s' % (tag, ext)),
                      dpi=400, bbox_inches='tight')
        plt.close(f)

    with open(os.path.join(FIGD, '_fig9_values.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    print('wrote fig9_efficiency.[pdf|png] + fig9a/b/c/d_panel.*  (%d methods)'
          % len(d))
    print('  ' + os.path.join(FIGD, '_fig9_values.txt'))


if __name__ == '__main__':
    main()
