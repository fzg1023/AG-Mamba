# -*- coding: utf-8 -*-
"""临时冒烟（用完即删）：验证 alias_sweep_clean 的配置与主对比 pub 臂同源。

α=2.0 + 交付配置 + 480 步 + 余弦退火 + seed0，应当与
results/HUST/AGMamba/pub/... 的日志**逐点一致**。不一致就别跑 60 单元的正式批次。
"""
import os
import sys

ROOT = '/home/fzg/Fault/AG-Mamba'
os.chdir(ROOT)
# 必须在 import agmamba.paths 之前设好（其 RESULTS 是导入时求值的）
os.environ['AGM_RESULTS'] = os.path.join(ROOT, 'results_probe_alpha')
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, ROOT)

import numpy as np                       # noqa: E402
import run_stage as S                    # noqa: E402
import agmamba.method as M               # noqa: E402
path = S.lm_logpath('HUST', 'T0', 'as_a20', 0)
print('smoke target:', path, flush=True)
M.configs = S._delivered_configs()
M.configs.gabor_alias = 2.0
print('  交付配置检查: front_mode=%s alias=%s use_ssm=%s tau_mode=%s '
      'w_mmd=%s w_adv=%s use_scheduler=%s f_lo=%s ksize=%s steps=%s'
      % (M.configs.front_mode, M.configs.gabor_alias, M.configs.use_ssm,
         M.configs.tau_mode, M.configs.w_mmd, M.configs.w_adv,
         M.configs.use_scheduler, M.configs.gabor_f_lo,
         M.configs.front_ksize, M.configs.steps), flush=True)

S.run_one('AGMamba', 'HUST', 'T0', 0, 'cuda', path)
a = S._points(path)
b = S._points(os.path.join(ROOT, 'results', 'HUST', 'AGMamba', 'pub', 'T0',
                           'HUST_T0_AGMamba_fc_pub_seed0.log'))
print('points: as_a20=%d  pub=%d' % (len(a), len(b)), flush=True)
if len(a) >= 5 and len(b) >= 5:
    print('last5: as_a20=%.4f  pub=%.4f  |Δ|=%.6f'
          % (np.mean(a[-5:]), np.mean(b[-5:]), abs(np.mean(a[-5:]) - np.mean(b[-5:]))),
          flush=True)
    if len(a) == len(b):
        d = [abs(x - y) for x, y in zip(a, b)]
        print('逐点 max|Δ|=%.6f  非零点数=%d/%d'
              % (max(d), sum(1 for x in d if x > 1e-6), len(d)), flush=True)
        print('VERDICT:', 'SAME ✓' if max(d) < 1e-6 else 'DIFF ✗', flush=True)
    else:
        print('VERDICT: 长度不一致 ✗', flush=True)
