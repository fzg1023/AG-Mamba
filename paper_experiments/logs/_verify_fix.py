# -*- coding: utf-8 -*-
"""临时脚本（用完即删）：验证 datasets_src 修复在 HUST 上真正生效。

两件事
------
A. 4 个崩溃方法各跑 1 个单元（T0/seed0），确认不再抛
   `Expected input batch_size (576) to match target batch_size (192)`、
   且产出满 60 个评估点。
B. 3 个"判定为安全"的方法（CNNC/IEDGNet/WhiteningNet）重跑 T0/seed0，
   与磁盘上已有日志逐点比对。本仓库 DETERMINISTIC=True，若逐位相同，
   即证明旧的（datasets_src 硬编码为 3）结果在 9 源域下确实未被污染。

结果写到 logs/_verify_fix.out（由调用方重定向），不触碰 results/ 正式日志。
"""
import os
import sys
import time
import shutil

ROOT = '/home/fzg/Fault/AG-Mamba'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
import run_compare as RC                        # noqa: E402

TMP = os.path.join(ROOT, 'logs', '_verify')
os.makedirs(TMP, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device, flush=True)

CRASHED = ['ADN', 'CCN', 'CDDG', 'DGNIS']
SAFE = ['CNNC', 'IEDGNet', 'WhiteningNet']

print('\n===== A. 崩溃方法重跑验证 =====', flush=True)
for m in CRASHED:
    lf = os.path.join(TMP, 'verify_%s.log' % m)
    t0 = time.time()
    try:
        accs = RC.run_one(m, 'unif', 'HUST', 'T0', 0, device, lf)
    except Exception as e:
        print('[FAIL] %-12s %s' % (m, str(e)[:160]), flush=True)
        continue
    n = len(accs)
    l10 = sum(accs[-10:]) / 10.0 if n >= 10 else float('nan')
    print('[ ok ] %-12s pts=%-3d last10=%.2f peak=%.2f (%.0fs)'
          % (m, n, l10, max(accs) if accs else float('nan'),
             time.time() - t0), flush=True)

print('\n===== B. 安全方法逐位等价性复检 =====', flush=True)
for m in SAFE:
    lf = os.path.join(TMP, 'verify_%s.log' % m)
    t0 = time.time()
    try:
        RC.run_one(m, 'unif', 'HUST', 'T0', 0, device, lf)
    except Exception as e:
        print('[FAIL] %-12s %s' % (m, str(e)[:160]), flush=True)
        continue
    old = RC.parse_accs(
        os.path.join(ROOT, 'results', 'HUST', m, 'unif', 'T0',
                     'HUST_T0_%s_fc_unif_seed0.log' % m))
    new = RC.parse_accs(lf)
    if not old or not new:
        print('[????] %-12s 旧=%s 新=%s' % (m, bool(old), bool(new)), flush=True)
        continue
    ov = [round(v, 4) for _, v in old]
    nv = [round(v, 4) for _, v in new]
    same = ov == nv
    dmax = max(abs(a - b) for a, b in zip(ov, nv)) if len(ov) == len(nv) else -1
    print('[%s] %-12s len %d vs %d  max|diff|=%.4f (%.0fs)'
          % ('SAME' if same else 'DIFF', m, len(ov), len(nv), dmax,
             time.time() - t0), flush=True)

print('\n===== C. DGNIS 三源域回归复检（PU，防"为 9 域改动"伤及旧结果）=====', flush=True)
try:
    lf = os.path.join(TMP, 'verify_DGNIS_PU.log')
    RC.run_one('DGNIS', 'unif', 'PU', 'T0', 0, device, lf)
    old = RC.parse_accs(os.path.join(ROOT, 'results', 'PU', 'DGNIS', 'unif',
                                     'T0', 'PU_T0_DGNIS_fc_unif_seed0.log'))
    new = RC.parse_accs(lf)
    if old and new:
        ov = [round(v, 4) for _, v in old]
        nv = [round(v, 4) for _, v in new]
        dmax = (max(abs(a - b) for a, b in zip(ov, nv))
                if len(ov) == len(nv) else -1)
        print('[%s] DGNIS@PU    len %d vs %d  max|diff|=%.4f'
              % ('SAME' if ov == nv else 'DIFF', len(ov), len(nv), dmax),
              flush=True)
    else:
        print('[????] DGNIS@PU    旧=%s 新=%s' % (bool(old), bool(new)),
              flush=True)
except Exception as e:
    print('[FAIL] DGNIS@PU    %s' % str(e)[:160], flush=True)

print('\n===== 验证结束 =====', flush=True)
