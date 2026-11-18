"""列出各协议前缀下可用的臂名（去掉 _seedN.log 后的唯一集合）。"""
import os
import re
import collections

D = 'results'
pat = re.compile(r'^(.+)_seed(\d+)\.log$')
groups = collections.defaultdict(set)
for fn in sorted(os.listdir(D)):
    m = pat.match(fn)
    if not m:
        continue
    groups[m.group(1)].add(int(m.group(2)))

for pref in ['PHM2009_T0', 'PHM2009_L0', 'PHM2009_M0', 'PU_T0', 'HUST_T0', 'HUST_T9']:
    arms = [(k, len(v)) for k, v in sorted(groups.items()) if k.startswith(pref + '_')]
    print('==', pref)
    for k, n in arms:
        print('   {:52s} n_seeds={}'.format(k, n))
