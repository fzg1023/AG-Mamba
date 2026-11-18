#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手稿结构自检：行数、公式编号、表/图题、中文对照块计数，以及遗留的候补/占位标记。

用法：python check_manuscript.py [手稿文件名]
默认检查 AG-Mamba_MSSP_manuscript.md。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本位于 analysis/ 或 tools/ 之下）
name = sys.argv[1] if len(sys.argv) > 1 else 'AG-Mamba_MSSP_manuscript.md'
p = os.path.join(ROOT, 'docs/manuscript', name)
s = open(p, encoding='utf-8').read()
lines = s.split('\n')
print('文件: {}'.format(p))
print('行数: {}'.format(len(lines)))
print('tag{{n}} 数: {}'.format(len(re.findall(r'tag\{', s))))
print('EN 表题 (**Table n.**): {}'.format(len(re.findall(r'(?m)^\*\*Table \d', s))))
print('CN 表题 (**表 n.**): {}'.format(len(re.findall(r'(?m)^\*\*表 \d', s))))
print('EN 图题 (**Fig. n.**): {}'.format(len(re.findall(r'(?m)^\*\*Fig\. \d', s))))
print('CN 图题 (**图 n.**): {}'.format(len(re.findall(r'(?m)^\*\*图 \d', s))))
print('插入图片: {}'.format(len(re.findall(r'(?m)^!\[', s))))
print('中文对照块: {}'.format(len(re.findall(r'供审阅', s))))
print('候补/占位: {}'.format(len(re.findall(r'候补|TODO|待补占位', s))))
# 表格行数不匹配检查：每个表题后应紧跟一个表头 + 分隔 + 数据行
print('\n各节起始行：')
for m in re.finditer(r'(?m)^#{2,3} .*$', s):
    ln = s[:m.start()].count('\n') + 1
    if ln > 900:
        print('  {:5d}  {}'.format(ln, m.group(0)[:70]))
