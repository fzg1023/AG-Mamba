"""统计 HUST 各方法在 results_dgfdb 中的日志覆盖（用于判断能否补 last-ten）。"""
import os
import re
import collections

RD = 'results'
pat = re.compile(r'^HUST_(T\d+)_(.+)_seed(\d+)\.log$')
cov = collections.defaultdict(lambda: collections.defaultdict(set))
for fn in os.listdir(RD):
    m = pat.match(fn)
    if m:
        t, arm, s = m.group(1), m.group(2), int(m.group(3))
        cov[arm][t].add(s)

print('{:<34s} {:>6s} {:>8s}'.format('arm', 'tasks', 'max seeds'))
for arm in sorted(cov, key=lambda a: -len(cov[a])):
    tasks = cov[arm]
    fs = max(len(v) for v in tasks.values())
    if len(tasks) >= 5:
        print('{:<34s} {:>6d} {:>8d}'.format(arm, len(tasks), fs))

print('\n--- 其他目录（widan/udtl/pu/phm2009） ---')
for d in ('results',):
    if not os.path.isdir(d):
        continue
    names = sorted(set(re.sub(r'_seed\d+\.log$', '', f) for f in os.listdir(d)))
    print(d, len(names), names[:6])
