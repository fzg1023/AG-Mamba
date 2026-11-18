#!/usr/bin/python
# -*- coding:utf-8 -*-
"""检查当前 run_stage 注册表可正常导入、全部 stage 有效。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.run_stage as R  # noqa: E402

print('module import OK')
print('STAGES 共 {} 个:'.format(len(R.STAGES)))
for k in sorted(R.STAGES):
    print('  {}'.format(k))
print()
print('GROUPS 共 {} 个:'.format(len(R.GROUPS)))
for k in sorted(R.GROUPS):
    print('  {:16s} -> {}'.format(k, R.GROUPS[k]))
print()
missing = [k for k, v in R.STAGES.items() if not callable(v)]
print('不可调用的 stage: {}'.format(missing if missing else '无'))
print('OK')
