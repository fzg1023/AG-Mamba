#!/usr/bin/python
# -*- coding:utf-8 -*-
"""DIRG（Politecnico di Torino 滚动轴承台架）-> 1024 点窗 npz。

只服务于"附加试验"：**不参与论文的任何数字**。

数据事实（取自台架说明书 PDF，`data/DIRG_bearing/*.pdf`）：
  - 文件名格式 `CnA_fff_vvv_m.mat`：
      CnA : 轴承编号 0A..6A（0A = 无缺陷；1A/2A/3A = 内圈压痕 450/250/150 um；
                            4A/5A/6A = 滚动体压痕 450/250/150 um）-> **7 类**
      fff : 名义转速 (Hz)，100..500（100 Hz = 6000 rpm）
      vvv : 载荷传感器输出 (mV)，灵敏度 0.499 mV/N -> 载荷 N = vvv / 0.499
      m   : 该工况的第 m 次记录
  - 采样率 fs = 51,200 Hz，时长 T = 10 s -> 每文件 512,000 点
  - 6 通道 = 2 个三轴加速度计：Ch1-3 = Axial x / Radial y / Radial z（故障轴承 B1 侧）
                              Ch4-6 = 同上（另一侧轴承）
  - 台架实测的载荷档位：0 / ~1000 N / ~1400 N / ~1800 N
      ~1000 N 在 5 个转速上齐全；~1400 N 只到 400 Hz；~1800 N 只到 300 Hz
      （说明书原文：受逆变器功率限制，高载荷达不到高转速）

用法：
  python tools/prepare_dirg.py                    # 默认 --loads 1000,1400,1800 --ch 1
  python tools/prepare_dirg.py --loads 1000 --ch 2
输出：
  datasets/dirg_data/dirg_x1024_ch<CH>.npz       # 键 `{speed}_{cls}_{tag}` -> (500,1024) f32
  datasets/dirg_data/dirg_x1024_ch<CH>_meta.csv  # 逐键溯源（文件名 / 实测载荷 / 窗口数）
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'DIRG_bearing', 'VariableSpeedAndLoad')
OUTDIR = os.path.join(ROOT, 'datasets', 'dirg_data')

FS = 51200.0            # Hz
DURATION = 10.0         # s
MV_PER_N = 0.499        # 载荷传感器灵敏度 mV/N
SIG = 1024              # 窗长
N_WIN = int(FS * DURATION) // SIG        # 500 个不重叠窗

CLASSES = ['0A', '1A', '2A', '3A', '4A', '5A', '6A']
SPEEDS = [100, 200, 300, 400, 500]
# 实测 vvv 落在 [lo, hi] 即归入该载荷档（已用全库网格核对，三档互不重叠）
LOADS = {'L0': (0, 20), 'L1000': (400, 520), 'L1400': (650, 760), 'L1800': (850, 930)}

NAME_RE = re.compile(r'^(C\dA)_(\d{3})_(\d{3})_(\d)\.mat$')


def scan():
    """返回 {(cls, speed, tag): (vvv, run, path)}。"""
    idx = {}
    for p in glob.glob(os.path.join(SRC, '*.mat')):
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        cls, speed, vvv, run = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        cls = cls[1:]          # 'C0A' -> '0A'（与 CLASSES 的写法一致）
        for tag, (lo, hi) in LOADS.items():
            if lo <= vvv <= hi:
                idx[(cls, speed, tag)] = (vvv, run, p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loads', default='1000,1400,1800',
                    help='要准备的载荷档（N），逗号分隔，可选 1000/1400/1800')
    ap.add_argument('--ch', type=int, default=1,
                    help='通道索引 0..5（默认 1 = Channel 2, Radial y）')
    ap.add_argument('--windows', type=int, default=0,
                    help='windows kept per file (0 = all available at --fs)')
    ap.add_argument('--fs', type=int, default=int(FS),
                    help='target fs in Hz (51200 native; 25600 = align with HUST -> 1024 pts = 40 ms)')
    args = ap.parse_args()

    tags = ['L' + t.strip() for t in args.loads.split(',') if t.strip()]
    for t in tags:
        if t not in LOADS:
            raise SystemExit('未知载荷档 %s（可选 %s）' % (t, sorted(LOADS)))

    idx = scan()
    print('扫描到 %d 个可用 (类, 转速, 载荷档) 组合' % len(idx))
    if not idx:
        raise SystemExit('扫描结果为空 —— 检查 %s 是否存在 .mat' % SRC)
    os.makedirs(OUTDIR, exist_ok=True)
    data, meta = {}, []
    speeds_by_tag = {}
    for tag in tags:
        for speed in SPEEDS:
            n_here = 0
            for cls in CLASSES:
                key3 = (cls, speed, tag)
                if key3 not in idx:
                    continue
                vvv, run, path = idx[key3]
                mat = loadmat(path)
                k = [x for x in mat if not x.startswith('__')][0]
                X = np.asarray(mat[k], dtype=np.float64)
                if X.ndim != 2 or X.shape[1] != 6:
                    raise SystemExit('意外的矩阵形状 %s: %s' % (path, X.shape))
                x = X[:, args.ch]
                if int(args.fs) != int(FS):
                    from math import gcd
                    from scipy.signal import resample_poly
                    g = gcd(int(FS), int(args.fs))
                    x = resample_poly(x, int(args.fs) // g, int(FS) // g)
                nwin = args.windows or (x.size // SIG)
                n = nwin * SIG
                if x.size < n:
                    raise SystemExit('%s too short: %d < %d' % (path, x.size, n))
                win = x[:n].reshape(nwin, SIG).astype(np.float32)
                if not np.isfinite(win).all():
                    raise SystemExit('%s 含非有限值' % path)
                name = '%d_%s_%s' % (speed, cls, tag)
                data[name] = win
                meta.append([name, os.path.basename(path), cls, speed,
                             int(tag[1:]), '%.1f' % (vvv / MV_PER_N), int(vvv), run,
                             args.ch, nwin, int(args.fs), int(x.size),
                             '%.6f' % float(np.sqrt(np.mean(x * x)))])
                n_here += 1
            if n_here:
                speeds_by_tag.setdefault(tag, []).append((speed, n_here))

    stem = 'dirg_x1024_ch%d' % args.ch
    if int(args.fs) != int(FS):
        stem += '_fs%d' % int(args.fs)
    npz = os.path.join(OUTDIR, stem + '.npz')
    np.savez_compressed(npz, **data)
    csvp = os.path.join(OUTDIR, stem + '_meta.csv')
    with open(csvp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'file', 'class', 'speed_hz', 'load_nominal_n',
                    'load_measured_n', 'load_vvv_mv', 'run', 'channel',
                    'n_windows', 'fs_hz', 'n_samples_used', 'rms_raw'])
        w.writerows(meta)

    print('npz  = %s  (%.1f MB)' % (npz, os.path.getsize(npz) / 1e6))
    print('meta = %s  (%d 键)' % (csvp, len(meta)))
    for tag, rows in sorted(speeds_by_tag.items()):
        print('  %-6s 转速 %s' % (tag, ', '.join('%dHz(%d类)' % r for r in rows)))
    miss = [(t, s, c) for t in tags for s in SPEEDS for c in CLASSES
            if (c, s, t) not in idx]
    if miss:
        print('  缺失 (tag, speed, cls) 共 %d 个：%s%s'
              % (len(miss), miss[:8], ' ...' if len(miss) > 8 else ''))


if __name__ == '__main__':
    main()
