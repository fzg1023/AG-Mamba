#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一协议（P-统一）对比的**配对显著性检验**（主口径 last5）。

口径为什么是 5、以及 last1 / last10 / peak 各自的定位，见
`docs/COMPARE_PROTOCOL.md` §5.1 与 `analysis/readout_window_study.py`。

为什么必须配对
--------------
协议里同种子跨方法严格配对（`seed ∈ {0..4}`）。同一 (task, seed) 在不同方法间
共享**完全相同的训练/测试划分与初始化流**，因此"同种子配对差"消掉了任务难度与
种子噪声，比两个独立样本的均值比较灵敏得多（n=20 时尤其明显）。

检验
----
对每个基线方法与 AG-Mamba：
  对 (task, seed) 配对，检验配对差的中位数是否为 0（双侧 Wilcoxon 符号秩）。
  同时报 wins/n（AG-Mamba 胜出的配对数）——它不依赖分布假设，可直接引用。

口径
----
· 主臂：AG-Mamba 的 `unif` 臂（与全部基线同处"统一阶梯"调度，才是公平对比）。
  `pub` 臂（余弦）作为次要参照一并给出。
· 指标：每个单元的末 5 个评估点均值（`last5`，**主口径**），% 单位。
  同时给出 `last1`（部署口径）与 `last10`（稳健性）；`peak` 仅留档不进正文。
· 数据来源：`results/<DS>/<method>/<arm>/<task>/*.log`，与 run_compare 同一解析规则。

用法
----
    python analysis/summary_compare.py PHM2009
    python analysis/summary_compare.py PU
输出
----
    docs/manuscript/_cmp_<DS>_stats.md
"""
import os
import re
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

from agmamba import paths as _P  # noqa: E402

OURS = 'AGMamba'
# 与 run_compare.parse_accs 同规则：同时兼容历史标签
PAT_ACC = re.compile(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)')
PAT_STEP = re.compile(r'=+\s*Step\s+(\d+)\s*=+')
SEEDS = [0, 1, 2, 3, 4]


def read_series(path):
    """读一个单元日志的全部评估点序列（%）。无效/过短日志返回 None。"""
    if not os.path.exists(path) or os.path.getsize(path) < 200:
        return None
    vals = []
    with open(path, errors='ignore') as f:
        for line in f:
            if PAT_STEP.search(line):
                continue
            m = PAT_ACC.search(line)
            if m:
                v = float(m.group(1))
                vals.append(v * 100.0 if v <= 1.5 else v)
    return vals or None


def window_mean(path, k):
    """某单元**末 k 个评估点**的均值（%）。

    k=10 为协议主口径；k=5、k=1（末轮）作为稳健性附加读数。
    日志无效或评估点不足 k 个返回 None。
    """
    vals = read_series(path)
    if not vals or len(vals) < k:
        return None
    return float(np.mean(vals[-k:]))


def last10(path):
    """末十轮均值（主口径的便捷入口，保留旧调用方兼容）。"""
    return window_mean(path, 10)


def reduce_series(vals, k):
    """按读数口径归约一个单元的评估点序列。

    · k 为整数  → **末 k 个评估点均值**（k=10 协议主口径、k=5、k=1 末轮）
    · k == 'peak' → 该单元的最大值（峰值口径）

    数据不足（评估点不足 k 个）返回 None。
    """
    if not vals:
        return None
    if k == 'peak':
        return float(max(vals))
    if len(vals) < k:
        return None
    return float(np.mean(vals[-k:]))


def discover(dataset):
    """返回 (method, arm, task) 三元组列表。"""
    root = os.path.join(_P.RESULTS, dataset)
    out = []
    if not os.path.isdir(root):
        return out
    for m in sorted(os.listdir(root)):
        pm = os.path.join(root, m)
        if not os.path.isdir(pm):
            continue
        for arm in sorted(os.listdir(pm)):
            pa = os.path.join(pm, arm)
            if not os.path.isdir(pa):
                continue
            for t in sorted(os.listdir(pa)):
                if os.path.isdir(os.path.join(pa, t)):
                    out.append((m, arm, t))
    return out


def unit_matrices(dataset, ks=(1, 5, 10, 'peak')):
    """一次读盘 → {k: (method, arm) -> {(task, seed): 该口径读数}}

    每个日志只解析一遍再算各口径。各口径各读一遍会把 IO 变多倍，
    在 UNC / 网络盘上会明显变慢。
    """
    out = {k: {} for k in ks}
    for m, arm, t in discover(dataset):
        for sd in SEEDS:
            vals = read_series(_P.log_path(dataset, t, m, sd, arm=arm))
            if not vals:
                continue
            for k in ks:
                v = reduce_series(vals, k)
                if v is not None:
                    out[k].setdefault((m, arm), {})[(t, sd)] = v
    return out


def paired(ours, base):
    """按 (task, seed) 配对，返回 (n, mean_delta, p_wilcoxon, wins, deltas)"""
    keys = sorted(set(ours) & set(base))
    if not keys:
        return 0, float('nan'), float('nan'), 0, {}
    d = {k: ours[k] - base[k] for k in keys}
    arr = np.array(list(d.values()), dtype=float)
    wins = int((arr > 0).sum())
    if np.allclose(arr, 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(arr, alternative='two-sided').pvalue)
        except Exception:  # noqa: BLE001  例如全为 0 时 scipy 会报错
            p = float('nan')
    return len(keys), float(arr.mean()), p, wins, d


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'PHM2009'
    mats = unit_matrices(dataset)
    mat = mats[5]
    if not mat:
        print('没有找到 %s 的结果日志' % dataset)
        return

    tasks = sorted({t for k in mat.values() for t, _ in k})
    methods = sorted({m for m, _ in mat if m != OURS})

    L = ['# %s 统一协议对比：配对显著性检验（**主口径 last5**）'
         % dataset, '',
         '由 `analysis/summary_compare.py` 生成。',
         '',
         '- 口径：每个单元的**末 k 个评估点均值**，单位 %。'
         '**主口径 k=5**（选择依据见 `docs/COMPARE_PROTOCOL.md` §5.1 与 '
         '`_readout_window_study.md`）；正文并列 `k=1`（部署口径）与 `k=10`（稳健性）；'
         '`peak` 仅在本文件留档',
         '- 配对：同 `(task, seed)` 严格配对（同划分、同初始化流）',
         '- 检验：双侧 **Wilcoxon 符号秩**；`wins` = AG-Mamba 胜出的配对数',
         '- 任务：%s；种子：%s；每方法最大配对数 n = %d'
         % (tasks, SEEDS, len(tasks) * len(SEEDS)),
         '',
         '> 主臂是 AG-Mamba 的 `unif`（与全部基线同处统一阶梯调度）。'
         '`pub`（余弦）列于末尾作次要参照。', '']

    for arm in ('unif', 'pub'):
        ours = mat.get((OURS, arm))
        if not ours:
            continue
        L += ['## AG-Mamba `%s` vs 各基线 —— **主口径 `last5`**' % arm, '',
              '| 基线 | n | AGMamba | 基线 | **Δ (配对均值)** | wins | p (Wilcoxon) | 判定 |',
              '|---|---|---|---|---|---|---|---|']
        rows = []
        for m in methods:
            b = mat.get((m, 'unif')) or mat.get((m, 'pub'))
            if not b:
                continue
            n, md, p, w, _ = paired(ours, b)
            if n == 0:
                continue
            ov = float(np.mean([ours[k] for k in sorted(set(ours) & set(b))]))
            bv = float(np.mean([b[k] for k in sorted(set(ours) & set(b))]))
            verdict = ('胜 (p<0.05)' if (p < 0.05 and md > 0) else
                       '负 (p<0.05)' if (p < 0.05 and md < 0) else '不显著')
            rows.append((md, m, n, ov, bv, w, p, verdict))
        rows.sort(reverse=True)
        for md, m, n, ov, bv, w, p, verdict in rows:
            L.append('| %s | %d | %.2f | %.2f | **%+.2f** | %d/%d | %.4f | %s |'
                     % (m, n, ov, bv, md, w, n, p, verdict))
        L.append('')

    # 逐任务明细（仅 unif 臂）
    ours = mat.get((OURS, 'unif'))
    if ours:
        L += ['## 逐任务 Δ（AG-Mamba unif − 基线，`last5` 主口径）', '',
              '| 基线 | ' + ' | '.join(tasks) + ' | 全部 |', '|---|'
              + '---|' * (len(tasks) + 1)]
        for m in methods:
            b = mat.get((m, 'unif'))
            if not b:
                continue
            cells = []
            for t in tasks:
                d = [ours[(t, sd)] - b[(t, sd)] for sd in SEEDS
                     if (t, sd) in ours and (t, sd) in b]
                cells.append('%+.2f' % np.mean(d) if d else '—')
            n, md, _, _, _ = paired(ours, b)
            L.append('| %s | %s | **%+.2f** |' % (m, ' | '.join(cells), md))
        L.append('')

    # ---------------- 附加读数：last1（末轮）/ last5 / peak ----------------
    # 为什么要看多口径：last10 是协议主口径。但"末十轮均值"会平滑掉末段波动，
    # 若某方法在最后几轮出现回落（PU 的 T0 就有这种迹象），不同口径可能给出
    # 不同结论。这里给出其它读数用于检验结论对口径是否稳健：
    #   · k=1    = 最后一个评估点（即第 60 轮，末轮终值）
    #   · k=5    = 末五个评估点均值
    #   · 'peak' = 该单元 60 个评估点的最大值（即 run_compare 报的 peak）
    # 注意：peak 是**向上有偏**的统计量——逐单元取 max 会把单轮噪声也当成性能，
    # 且偏置量级取决于该方法的波动幅度，故跨方法比较 peak 并不公平。协议
    # 因此用 last10 作主口径，peak 只作参照（偏置的实测见文末专节）。
    KEYS = (1, 5, 10, 'peak')
    L += ['---', '',
          '## 口径对照：四种读数的 Δ（AG-Mamba `unif` 臂）', '',
          'Δ = 同 `(task, seed)` 配对的均值差（每格 n=%d）。'
          '「同号」= last1/last5/last10/peak 的 Δ 全部同向。'
          % (len(tasks) * len(SEEDS)), '',
          '> `last5` 为主口径，`last1`（部署口径）与 `last10`（稳健性）**进论文**；'
          '`peak` **不进论文**，在此仅作留档（偏置随方法变化，见文末专节）。', '',
          '| 基线 | **Δ last5（主）** | p | Δ last10 | Δ last1 | Δ peak | 同号 |',
          '|---|---|---|---|---|---|---|']
    o5 = mats[5].get((OURS, 'unif'))
    rows_r = []
    for m in methods:
        if not o5:
            continue
        if not (mats[5].get((m, 'unif')) or mats[5].get((m, 'pub'))):
            continue
        cells = {}
        for k in KEYS:
            ok, bk = mats[k].get((OURS, 'unif')), (mats[k].get((m, 'unif'))
                                                  or mats[k].get((m, 'pub')))
            if ok and bk:
                _, md, p, _, _ = paired(ok, bk)
                cells[k] = (md, p)
            else:
                cells[k] = (float('nan'), float('nan'))
        d5 = cells[5][0]
        same = all((cells[k][0] > 0) == (d5 > 0) for k in KEYS
                   if cells[k][0] == cells[k][0])   # 跳过 NaN
        rows_r.append((d5, m, cells, same))
    rows_r.sort(reverse=True)
    for _d, m, c, same in rows_r:
        L.append('| %s | **%+.2f** | %.4f | %+.2f | %+.2f | %+.2f | %s |'
                 % (m, c[5][0], c[5][1], c[10][0], c[1][0], c['peak'][0],
                    '是' if same else '**否**'))
    L.append('')

    for k, name in ((10, 'last10（稳健性口径 —— 进论文）'),
                    (1, 'last1（部署口径 —— 进论文）'),
                    ('peak', 'peak（峰值，逐单元 max）—— 仅留档，不进论文')):
        ok = mats[k].get((OURS, 'unif'))
        if not ok:
            continue
        L += ['## AG-Mamba `unif` vs 各基线 —— %s' % name, '']
        if k == 'peak':
            L += ['> **口径提示（peak 为何不进论文）**：peak = 逐单元 60 个评估点的最大值'
                  '再跨种子平均。它**向上有偏**（把单轮噪声抬起来的点也算作性能），'
                  '且偏置量级取决于该方法的波动幅度（实测 0.85–6.89 分，跨度 8 倍），'
                  '**不适合跨方法比较**。协议主口径是 `last5`；本表仅在本文件留档，'
                  '偏置实测见文末专节。', '']
        L += ['| 基线 | n | AGMamba | 基线 | **Δ** | wins | p (Wilcoxon) | 判定 |',
              '|---|---|---|---|---|---|---|---|']
        rr = []
        for m in methods:
            bk = mats[k].get((m, 'unif')) or mats[k].get((m, 'pub'))
            if not bk:
                continue
            n, md, p, w, _ = paired(ok, bk)
            if n == 0:
                continue
            keys = sorted(set(ok) & set(bk))
            ov = float(np.mean([ok[x] for x in keys]))
            bv = float(np.mean([bk[x] for x in keys]))
            verdict = ('胜 (p<0.05)' if (p < 0.05 and md > 0) else
                       '负 (p<0.05)' if (p < 0.05 and md < 0) else '不显著')
            rr.append((md, m, n, ov, bv, w, p, verdict))
        rr.sort(reverse=True)
        for md, m, n, ov, bv, w, p, verdict in rr:
            L.append('| %s | %d | %.2f | %.2f | **%+.2f** | %d/%d | %.4f | %s |'
                     % (m, n, ov, bv, md, w, n, p, verdict))
        L.append('')

        L += ['### 逐任务 Δ（%s，AG-Mamba unif − 基线）' % name, '',
              '| 基线 | ' + ' | '.join(tasks) + ' | 全部 |', '|---|'
              + '---|' * (len(tasks) + 1)]
        for m in methods:
            bk = mats[k].get((m, 'unif'))
            if not bk:
                continue
            cells = []
            for t in tasks:
                d = [ok[(t, sd)] - bk[(t, sd)] for sd in SEEDS
                     if (t, sd) in ok and (t, sd) in bk]
                cells.append('%+.2f' % np.mean(d) if d else '—')
            n, md, _, _, _ = paired(ok, bk)
            L.append('| %s | %s | **%+.2f** |' % (m, ' | '.join(cells), md))
        L.append('')

    # ---------------- peak 的口径偏置实测 ----------------
    # 两种 peak 定义：
    #   A) 逐单元取 max → 再跨种子平均（run_compare 报的 peak，受单轮噪声上偏）
    #   B) 先跨种子平均轨迹 → 再取 max（单轮噪声被平均掉，是"真实可达峰值"）
    # 恒有 B <= A；A-B 就是噪声把峰值抬高的量级。这个量级若与方法无关，
    # peak 还能勉强比；若随方法变化，则跨方法比 peak 就是比"谁抖得更厉害"。
    pb = {}
    for m, arm, t in discover(dataset):
        series = []
        for sd in SEEDS:
            vals = read_series(_P.log_path(dataset, t, m, sd, arm=arm))
            if vals:
                series.append(vals)
        if not series:
            continue
        n_min = min(len(s) for s in series)
        a = float(np.mean([max(s) for s in series]))
        traj = [float(np.mean([s[i] for s in series])) for i in range(n_min)]
        pb.setdefault((m, arm), {})[t] = (a, float(max(traj)))

    rows_pb = []
    for (m, arm), td in pb.items():
        if not td:
            continue
        A = float(np.mean([v[0] for v in td.values()]))
        B = float(np.mean([v[1] for v in td.values()]))
        rows_pb.append((A - B, m, arm, A, B))
    rows_pb.sort(reverse=True)
    L += ['---', '',
          '## peak 的口径偏置实测（为什么协议不用 peak 作主口径）', '',
          '两种 peak 定义：**A** = 逐单元取 60 个评估点的 max 再跨种子平均'
          '（即 `run_compare` 报的 peak）；**B** = 先跨种子平均轨迹、再取 max'
          '（单轮噪声被平均掉，是"真实可达峰值"）。恒有 B ≤ A，'
          '**A − B 就是噪声把峰值抬高的量级**。', '',
          '| 方法 | 臂 | peak A（逐单元 max 均值） | peak B（平均轨迹 max） | A − B |',
          '|---|---|---|---|---|']
    for g, m, arm, A, B in rows_pb:
        L.append('| %s | %s | %.2f | %.2f | **+%.2f** |' % (m, arm, A, B, g))
    L.append('')

    out = os.path.join(ROOT, 'docs', 'manuscript', '_cmp_%s_stats.md' % dataset)
    # 必须显式 utf-8：报告含 U+2212（−）等非 GBK 字符，在 Windows 上
    # 用默认编码会 UnicodeEncodeError（Linux 默认 UTF-8 故不暴露）。
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\n[written] %s' % out)


if __name__ == '__main__':
    main()
