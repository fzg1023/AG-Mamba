# -*- coding:utf-8 -*-
"""同口径 HUST vs DIRG 的 alpha 对照（主口径 last5）。

生成 docs/DIRG_alpha_对照_HUST_vs_DIRG.md
用法: python tools/alpha_last5_compare.py
"""
import csv
import os
import re
import statistics as st

import numpy as np
from scipy import stats

os.chdir('/home/fzg/Fault/AG-Mamba')
ROOT_H = 'results_probe_alpha/HUST/AGMamba'
ARMS_H = [('0.0', 'as_a00'), ('0.5', 'as_a05'), ('1.0', 'as_a10'),
          ('1.5', 'as_a15'), ('2.0', 'as_a20'), ('3.0', 'as_a30')]
TASKS_H = ['T0', 'T9']
SEEDS = [0, 1, 2, 3, 4]
DIRG = [('0.0', 'results_dirg/L1000_ch0_a00'),
        ('1.0', 'results_dirg/L1000_ch0_a10'),
        ('2.0', 'results_dirg/L1000_ch0'),
        ('3.0', 'results_dirg/L1000_ch0_a30')]
OUT = 'docs/DIRG_alpha_对照_HUST_vs_DIRG.md'


def series(p):
    v = []
    for line in open(p, errors='ignore'):
        m = re.search(r'tgt_acc:?\s*([0-9.]+)', line)
        if m:
            v.append(float(m.group(1)) * 100.0)
    return v


def st_of(v):
    return dict(n=len(v), last1=v[-1], last5=st.mean(v[-5:]),
                last10=st.mean(v[-10:]), best=max(v))


H, Hb = {}, {}
for a, arm in ARMS_H:
    for t in TASKS_H:
        for s in SEEDS:
            p = os.path.join(ROOT_H, arm, t,
                             'HUST_%s_AGMamba_fc_%s_seed%d.log' % (t, arm, s))
            d = st_of(series(p))
            H[(a, t, s)] = d['last5']
            Hb[(a, t, s)] = dict(d)

D = {}
for a, dd in DIRG:
    p = os.path.join(dd, '_summary.csv')
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if (int(r['n_eval'] or 0) >= 60 and r['kind'] == 'lodo'
                and r['method'] == 'AGMamba'):
            D[(a, r['task'], int(r['seed']))] = float(r['last5'])

L = []
A = L.append
A('# HUST vs DIRG：alpha（受控混叠强度）同口径对照')
A('')
A('- 生成脚本：`tools/alpha_last5_compare.py`（可直接重跑，纯读日志/CSV）')
A('- **主口径 `last5`** = 每条精度曲线**最后 5 个检查点**的均值（与手稿主表一致）；')
A('  `last1`（部署）/`last10`（鲁棒性）/`best`（逐 epoch 最大，论文已排除）并列给出作参照')
A('- HUST 源：`results_probe_alpha/HUST/AGMamba/as_a{00,05,10,15,20,30}`，')
A('  即 v0.3 `alias_sweep_clean`，**交付配置前端**（`ksize128`/`f_lo=0.0025`/`gaborwide`），')
A('  T0/T9 × 5 种子，每单元 60 个检查点')
A('- DIRG 源：`results_dirg/L1000_ch0{,_a00,_a10,_a30}/_summary.csv`（ch0, fs=51200, sig=1024,')
A('  载荷 1000 N, 5 任务 × 5 种子，同样 60 个检查点）')
A('- **前端一致性已核验（决定这组对照是否成立）**：')
A('  `experiments/arms/arms.json` 的 `frozen` 块 = `gaborwide`/`front_chan=512`/'
  '`front_ksize=128`/`gabor_f_lo=0.0025`/`dc_frac=0.3`/`tau_mode=zero`/'
  '`w_mmd=w_adv=0`，与 `agmamba/configs.yaml` 交付配置**逐项相同**；')
A('  `run_stage.stage_alias_sweep_clean` 只覆盖 `gabor_alias` 一个字段'
  '（`_M.configs = _delivered_configs(); _M.configs.gabor_alias = alpha`），'
  '且该 stage 已**明确把主口径从 peak 改为 last5**。')
A('- 训练预算略有差异：HUST 每臂 480 步，DIRG 每臂 360 步（都产生 60 个检查点 + 余弦退火到零）。')
A('')

A('## 1. HUST alpha 扫描（同一条曲线，四种口径）')
A('')
for metric, title in (('last5', '主口径 last5'), ('last10', 'last10'),
                      ('last1', 'last1（部署）'), ('best', 'best（论文排除）')):
    A('### %s' % title)
    A('')
    A('| α | T0（目标 20 Hz） | T9（目标 80 Hz） | 池化(n=10) |')
    A('|---|---|---|---|')
    for a, _ in ARMS_H:
        t0 = [Hb[(a, 'T0', s)][metric] for s in SEEDS]
        t9 = [Hb[(a, 'T9', s)][metric] for s in SEEDS]
        A('| %s | %.2f ± %.2f | %.2f ± %.2f | **%.2f** |'
          % (a, st.mean(t0), st.pstdev(t0), st.mean(t9), st.pstdev(t9),
             st.mean(t0 + t9)))
    A('')

A('## 2. HUST 配对检验（vs 交付值 alpha=2.0，同任务同种子，last5）')
A('')
A('| 对比 | Δ(T0) | p | wins | Δ(T9) | p | wins |')
A('|---|---|---|---|---|---|---|')
for a, _ in ARMS_H:
    if a == '2.0':
        continue
    d0 = [H[(a, 'T0', s)] - H[('2.0', 'T0', s)] for s in SEEDS]
    d9 = [H[(a, 'T9', s)] - H[('2.0', 'T9', s)] for s in SEEDS]
    p0 = stats.wilcoxon(d0).pvalue if any(d0) else 1.0
    p9 = stats.wilcoxon(d9).pvalue if any(d9) else 1.0
    A('| α=%s vs 2.0 | %+.2f | %.4f | %d/5 | %+.2f | %.4f | %d/5 |'
      % (a, st.mean(d0), p0, sum(1 for x in d0 if x > 0),
         st.mean(d9), p9, sum(1 for x in d9 if x > 0)))
A('')
av = np.array([float(a) for a, _ in ARMS_H])
m0 = np.array([st.mean([H[(a, 'T0', s)] for s in SEEDS]) for a, _ in ARMS_H])
m9 = np.array([st.mean([H[(a, 'T9', s)] for s in SEEDS]) for a, _ in ARMS_H])
r0 = stats.pearsonr(av, m0)
r9 = stats.pearsonr(av, m9)
A('alpha 与任务均值（last5）：T0 r=%.3f (p=%.3f)，极差 %.2f 点；'
  'T9 r=%.3f (p=%.3f)，极差 %.2f 点。'
  % (r0[0], r0[1], m0.max() - m0.min(), r9[0], r9[1], m9.max() - m9.min()))
A('')

A('## 3. DIRG alpha 扫描（last5，5 任务 × 5 种子）')
A('')
A('| α | n | 池化 | T0 | T1 | T2 | T3 | T4 |')
A('|---|---|---|---|---|---|---|---|')
for a, _ in DIRG:
    keys = [k for k in D if k[0] == a]
    if not keys:
        A('| %s | — | （未跑） | | | | | |' % a)
        continue
    byt = {}
    for (aa, t, s), x in D.items():
        if aa == a:
            byt.setdefault(t, []).append(x)
    cells = ['%.1f' % st.mean(byt[t]) if t in byt else '—' for t in
             ['T0', 'T1', 'T2', 'T3', 'T4']]
    A('| %s | %d | **%.2f** | %s |' % (a, len(keys),
                                       st.mean([D[k] for k in keys]),
                                       ' | '.join(cells)))
A('')

A('### 3.1 DIRG 配对检验（vs α=2.0，同任务同种子，last5）')
A('')
A('| 对比 | 任务 | n | Δ | p | wins |')
A('|---|---|---|---|---|---|')
for a in ('0.0', '1.0', '3.0'):
    ks = sorted({(k[1], k[2]) for k in D if k[0] == a}
                & {(k[1], k[2]) for k in D if k[0] == '2.0'})
    if not ks:
        continue
    dv = [D[(a, t, s)] - D[('2.0', t, s)] for t, s in ks]
    pv = stats.wilcoxon(dv).pvalue if any(dv) else 1.0
    A('| α=%s vs 2.0 | 合并 | %d | %+.2f | %.4f | %d/%d |'
      % (a, len(dv), st.mean(dv), pv, sum(1 for x in dv if x > 0), len(dv)))
    for t in ['T0', 'T1', 'T2', 'T3', 'T4']:
        k2 = [(tt, ss) for tt, ss in ks if tt == t]
        if not k2:
            continue
        dv2 = [D[(a, tt, ss)] - D[('2.0', tt, ss)] for tt, ss in k2]
        p2 = stats.wilcoxon(dv2).pvalue if any(dv2) else 1.0
        A('|  | %s | %d | %+.2f | %.4f | %d/%d |'
          % (t, len(dv2), st.mean(dv2), p2, sum(1 for x in dv2 if x > 0), len(dv2)))
A('')

A('## 4. 同口径对照总表（主口径 last5）')
A('')
A('| α | HUST 池化(T0+T9, n=10) | DIRG 池化(5 任务, n=25) |')
A('|---|---|---|')
for a, _ in ARMS_H:
    hv = [H[(a, t, s)] for t in TASKS_H for s in SEEDS]
    dv = [D[k] for k in D if k[0] == a]
    A('| %s | **%.2f** | %s |' % (a, st.mean(hv),
                                  ('**%.2f**' % st.mean(dv)) if dv else '（未跑）'))
A('')

A('## 5. 结论')
A('')
A('1. **HUST（交付配置、last5）：α=2.0 就是最优点**，不是 α=3.0。')
A('   T9 曲线呈倒 U：62.02 → 63.32 → 70.99 → 68.35 → **76.07** → 72.67；')
A('   α=0 在 T9 上比 2.0 低 **14.05 点**（p=0.0625，0/5），α=3.0 低 3.41 点（1/5）。')
A('   T0 几乎不受影响（极差 3.92 点）。')
A('2. **DIRG 的池化方向相反**：α=0/1 各 52.29，比 α=2.0 的 48.11 高 4.18 点；'
  '但**池化层面不显著**（α=0：+4.17, p=0.2521, 15/25；α=1：+4.18, p=0.1353, 20/25）。')
A('3. **但逐任务看 DIRG 是分裂的，而且 T0 与 HUST 同向**：')
A('   DIRG T0（目标 100 Hz）α 越大越好：26.2 / 22.3 / 39.8 / **40.5**（α=0/1/2/3）；')
A('   而 T1–T4 相反（T2：69.4 / 69.9 / 54.7；T3：70.6 / 54.3 / 47.2）。')
A('   所以 DIRG 整体"低 α 更好"是 T1–T4 拉动的，**不是对 HUST 规律的否定**。')
A('4. 因此 alpha 在 DIRG 上的作用**不是统一增益，而是任务间的再分配**；')
A('   且最好的 alpha（52.29）仍远低于采样率杠杆（fs=25600 → AGMamba 62.5）。')

A('')
A('## 6. 与 `docs/manuscript/_s63_data.md` 表 A 的差异（重要）')
A('')
A('该表给的是 **旧前端（`ksize64`/`f_lo=0.005`）+ `best`/`last10` 口径**，结论是"α 越大越好、α=3.0 最优'
  '（T9 best 82.09）"。本次按 **交付配置（`ksize128`/`f_lo=0.0025`）+ 主口径 last5** 重读，')
A('结论变成"**α=2.0 最优**（T9 last5 76.07，α=3.0 仅 72.67）"。两者不矛盾，因为')
A('- 前端不同（首层核 64 vs 128、`f_lo` 0.005 vs 0.0025）；')
A('- 口径不同（`best` 偏差随方法/臂变化，论文已明确排除）。')
A('')
A('**行动项**：若手稿某处仍引用"α=3.0 更优"，需改为按交付配置 + last5 的结论，或明确标注旧口径。')
A('')
A('## 7. 注意事项')
A('')
A('- HUST 只跑了 T0/T9 两个任务；DIRG 跑了 5 个任务，任务集合不同，池化数字不可直接比大小，')
A('  只能比"**方向/形状**"。')
A('- DIRG 的 α=3.0 只完成 T0（n=5），其池化 40.51 不能与 α=2.0 的 25 单元池化并列解读。')
A('- DIRG 每单元 5 个种子，种子间标准差约 5–7 点，任务级差异（n=5）多数达不到显著。')
A('- 训练预算：HUST 480 步 vs DIRG 360 步 —— 检查点数相同，但总步数不同，'
  '**同口径不等于同预算**。')
A('- HUST 每臂只覆盖 T0/T9；DIRG 的 5 个任务的**最远/最近源域比值**与 HUST 的 T0/T9 并不一一对应。')

open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('WROTE', OUT, os.path.getsize(OUT), 'bytes,', len(L), 'lines')
