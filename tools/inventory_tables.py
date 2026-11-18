#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重编号前的清单：统计 "Table N" / "表 N"（N = 12..19）的所有出现位置。

用途：为一次性的表号平移（12-19 -> 14-21）提供完整清单，避免漏改。
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
p = os.path.join(ROOT, 'docs/manuscript', 'AG-Mamba_MSSP_manuscript.md')
lines = open(p, encoding='utf-8').read().split('\n')

for n in range(12, 20):
    pat_en = re.compile(r'Tables? {}\b'.format(n))
    pat_cn = re.compile(r'表 {}[^0-9]'.format(n))
    hits = []
    for i, ln in enumerate(lines, 1):
        for pat, tag in ((pat_en, 'EN'), (pat_cn, 'CN')):
            for m in pat.finditer(ln):
                s = max(0, m.start() - 25)
                e = min(len(ln), m.end() + 30)
                hits.append((i, tag, ln[s:e].replace('\n', ' ')))
    print('== {} == 共 {} 处'.format(n, len(hits)))
    for i, tag, txt in hits:
        print('  L{:<5d} [{}] {}'.format(i, tag, txt))
    print()
