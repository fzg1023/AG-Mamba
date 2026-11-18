# -*- coding: utf-8 -*-
"""从现有 HUST 日志统计两种精度口径：

  best  = 轨迹上最优目标精度（当前主表口径，预言式检查点选择）
  final = 最后一个检查点的目标精度（非预言，无需访问目标域做选择）

输出：控制台报告 + docs/manuscript/final_vs_best_epoch.md
"""
import os
import re
import numpy as np

# --- 结果路径：统一走 agmamba.paths（results/<数据集>/<算法>/<臂>/<任务>/）---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
from agmamba.paths import find as _findlog  # noqa: E402

try:
    from scipy import stats as st
except Exception:                                    # pragma: no cover
    st = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
RD = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'docs/manuscript', 'final_vs_best_epoch.md')

TASKS = ['T{}'.format(i) for i in range(10)]
SEEDS = [0, 1, 2, 3, 4]

METHODS = [
    ('AG-Mamba', 'HUST_{t}_AGMamba_fc_sc_steps6x_seed{s}.log'),
    ('WIDAN (std)', 'HUST_{t}_WIDAN_seed{s}.log'),
    ('WIDAN (matched)', 'HUST_{t}_WIDAN_long_seed{s}.log'),
    ('DANN', 'HUST_{t}_DANN_seed{s}.log'),
    ('UDTL', 'HUST_{t}_UDTL_seed{s}.log'),
]


def series(path):
    if not os.path.exists(path):
        return []
    with open(path, errors='ignore') as f:
        return [float(x) for x in re.findall(r'tgt_acc:\s*([0-9.]+)', f.read())]


def collect(pat):
    best, final, nck = {}, {}, {}
    for t in TASKS:
        for s in SEEDS:
            v = series(_findlog(pat.format(t=t, s=s)))
            if not v:
                continue
            k = (t, s)
            best[k] = max(v) * 100.0
            final[k] = v[-1] * 100.0
            nck[k] = len(v)
    return best, final, nck


def paired(a, b):
    """a, b: dict[(task,seed)] -> acc；返回 (mean Δ, t, p, #Δ>0, n)。"""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], float)
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return float(d.mean()), float('nan'), float('nan'), int((d > 0).sum()), n
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = float(st.ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue)
    return float(d.mean()), float(t), p, int((d > 0).sum()), n


data = {}
for name, pat in METHODS:
    data[name] = collect(pat)

L = []
L.append('# HUST：best-epoch（现口径）与 final-epoch（非预言）双口径对照')
L.append('')
L.append('数据源：`results_dgfdb/HUST_T*_*.log`，每运行 10 任务 × 5 种子 = 50 运行；'
         'AG-Mamba 为交付配置 `sc_steps6x`（2880 步，余弦）。'
         'best = 轨迹最优目标精度；final = 末个检查点目标精度。')
L.append('')
L.append('## 1. 总览')
L.append('')
L.append('| 方法 | 运行数 | 检查点/e 中位数 | best 均值±std | final 均值±std | best−final |')
L.append('|---|---|---|---|---|---|')

for name, _ in METHODS:
    b, f, c = data[name]
    if not b:
        L.append('| {} | 0 | – | – | – | – |'.format(name))
        continue
    ks = sorted(b)
    bv = np.array([b[k] for k in ks])
    fv = np.array([f[k] for k in ks])
    nck = int(np.median([c[k] for k in ks]))
    L.append('| {} | {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:+.2f} |'.format(
        name, len(ks), nck, bv.mean(), bv.std(), fv.mean(), fv.std(),
        (bv - fv).mean()))
L.append('')

L.append('## 2. AG-Mamba 逐任务（5 种子均值）')
L.append('')
L.append('| 任务 | 目标转速 (Hz) | best | final | best−final |')
L.append('|---|---|---|---|---|')
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
b, f, _ = data['AG-Mamba']
for i, t in enumerate(TASKS):
    ks = sorted([k for k in b if k[0] == t])
    if not ks:
        continue
    bv = np.array([b[k] for k in ks])
    fv = np.array([f[k] for k in ks])
    L.append('| {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:+.2f} |'.format(
        t, SPEEDS[i], bv.mean(), bv.std(), fv.mean(), fv.std(),
        (bv - fv).mean()))
L.append('')

L.append('## 3. 配对检验：AG-Mamba vs 基线（同任务同种子）')
L.append('')
L.append('| 对比 | 口径 | Δ (点) | t | p | 更优对数 | n |')
L.append('|---|---|---|---|---|---|---|')
ref_b, ref_f, _ = data['AG-Mamba']
for name, _ in METHODS[1:]:
    ob, of, _ = data[name]
    if not ob:
        L.append('| AG vs {} | – | 数据缺失 | | | | |'.format(name))
        continue
    for tag, ref, oth in (('best', ref_b, ob), ('final', ref_f, of)):
        d, t, p, w, n = paired(ref, oth)
        L.append('| AG vs {} | {} | {:+.2f} | {:+.2f} | {:.4f} | {}/{} | {} |'.format(
            name, tag, d, t, p, w, n, n))
L.append('')
L.append('注：Δ > 0 表示 AG-Mamba 更高；p 为双侧配对 t 检验；'
         'WIDAN (std) 为 480 步标准预算，WIDAN (matched) 为与 AG-Mamba 同预算。')

txt = '\n'.join(L) + '\n'
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write(txt)
print(txt)

# 附：检查点密度的一致性核查
print('检查点密度（应分别对应各自步数/iters_per_epoch）：')
for name, _ in METHODS:
    _, _, c = data[name]
    if c:
        vals = sorted(set(c.values()))
        print('  {:16s} unique checkpoints: {}'.format(name, vals[:8]))
