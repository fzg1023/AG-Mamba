# -*- coding: utf-8 -*-
"""临时脚本（用完即删）：DGNIS 三源域回归复检。

背景：DGNIS 原实现硬编码 8 个故障分类头，HUST（9 源域）会 IndexError。
修成"按源域数建头"后 PU 旧结果**不再可复现** —— 根因是全局 torch RNG 的
消耗量会泄漏给之后构造的 InfiniteDataLoader（RandomSampler 用全局 RNG）。
改为 `max(8, num_domains)` 后，≤8 源域应重新与历史实现逐位同构。
本脚本用 PU/T0/seed0 做金丝雀比对。
"""
import os
import sys

ROOT = '/home/fzg/Fault/AG-Mamba'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
import run_compare as RC                        # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
lf = os.path.join(ROOT, 'logs', '_verify', 'verify_DGNIS_PU2.log')
RC.run_one('DGNIS', 'unif', 'PU', 'T0', 0, dev, lf)

old = RC.parse_accs(os.path.join(
    ROOT, 'results', 'PU', 'DGNIS', 'unif', 'T0',
    'PU_T0_DGNIS_fc_unif_seed0.log'))
new = RC.parse_accs(lf)
if not old or not new:
    print('[????] DGNIS@PU old=%s new=%s' % (bool(old), bool(new)), flush=True)
else:
    ov = [round(v, 4) for _, v in old]
    nv = [round(v, 4) for _, v in new]
    dmax = max(abs(a - b) for a, b in zip(ov, nv)) if len(ov) == len(nv) else -1
    print('[%s] DGNIS@PU len %d vs %d max|diff|=%.4f'
          % ('SAME' if ov == nv else 'DIFF', len(ov), len(nv), dmax), flush=True)
print('DONE', flush=True)
