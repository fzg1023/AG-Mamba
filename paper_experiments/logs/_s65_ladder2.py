#!/usr/bin/python
# -*- coding:utf-8 -*-
"""§6.5 L3 重做：**带限精确膨胀**替身集 + 训练溯源自检。

第一版（`logs/_s65_accuracy_probe.py`）的做法有两个可量化的毛病：

 1. **线性插值是 sinc² 低通**：`y[i] = lerp(x, i·s)` 在核组通带（f ≤ 0.45）内
    也有明显衰减（到新 Nyquist 处约 −7.9 dB），会把"膨胀"与"低通"混在一起；
 2. **窗长不足**：压缩 s 倍需要读到窗尾之外（i·s ≤ 1091 > 1023），只能镜像补尾。

本脚本改用物理上正确的离散膨胀 —— 对**连续**记录做带限抗混叠重采样：

    y[i] = x(i·r),  r = f_target / f_source   （时间压缩 = 梳齿加密）

  · HUST 的 npz 每类每域是 **256×1024 无重叠**样本（`prepare_hust.py` 预切分），
    故把 tail 的 50 条拼成 **51200 点连续段**，整段用
    `scipy.signal.resample_poly(seg, up, down)`（有理比 r = up/down =
    16/15、8/7、16/13）重采样，再按 1024 非重叠切窗；
  · 抗混叠滤波 = 物理上真实的"采样率不变、转速变快"代价：超过 Nyquist/r 的
    带外成分被滤除（r = 16/15 时，核组通带 f ≤ 0.45 完全无损：0.45·16/15 =
    0.48 < 0.5）；
  · 逐窗 `(w − mean)/std`（与 `Normalize('mean-std')` 逐行一致），标签按
    `CLASSES` 顺序 → 与 pipeline 路径**同口径**。

同时修正第一版的**溯源瑕疵**：把评估集构造放在 `setup_seed` **之前**，使训练
轨迹与交付 `run_one` 逐位同源（第一版多出的数据加载挪动了 loader 的随机数流，
60 个评估点上最大差 0.44 点）。本脚本会逐点比对参照日志并给出结论。

输出：
  logs/_s65_ladder2.md        结果表（正文引用来源）
  logs/_s65_ladder2.npz       全部精度 + 替身集构造参数
  logs/_s65_model_trained.pt  末权重完整 state_dict（便于日后复算，免重训）
  logs/_s65_arm_trained2.log  逐步训练日志（溯源比对用）

用法：AGM_FORCE_PY_SCAN=1 python logs/_s65_ladder2.py
"""
import os
import sys
import time
import logging

os.environ['AGM_FORCE_PY_SCAN'] = '1'

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader, Dataset              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments'))

import experiments.run_cell as RC                            # noqa: E402
import experiments.run_stage as RS                           # noqa: E402
import agmamba.method as _M                                  # noqa: E402
from utils.DatasetClass import InfiniteDataLoader            # noqa: E402
from datasets.HUST_bearing import CLASSES, data_load         # noqa: E402

OUT_MD = os.path.join(_ROOT, 'logs', '_s65_ladder2.md')
OUT_NPZ = os.path.join(_ROOT, 'logs', '_s65_ladder2.npz')
OUT_PT = os.path.join(_ROOT, 'logs', '_s65_model_trained.pt')
REF_LOG = os.path.join(_ROOT, 'results_probe_align', 'HUST', 'AGMamba',
                       'aln_noall', 'T9',
                       'HUST_T9_AGMamba_fc_aln_noall_seed0.log')
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_F = 80.0
SPEED = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]

# 源域替身：域号 -> 有理比 (up, down)
#   resample_poly(x, up, down) 的输出采样率 = up/down 倍，且 y[i] = x(i·down/up)
#   时间压缩 r = f_tgt/f_src 要求 down/up = r  ⇒  (up, down) = (f_src, f_tgt) 约分
#   例：80/75 = 16/15 ⇒ (15, 16)；80/70 = 8/7 ⇒ (7, 8)；80/65 = 16/13 ⇒ (13, 16)
LAD2 = [(8, 15, 16), (7, 7, 8), (6, 13, 16)]
LAD1 = [(8, 80.0 / 75.0), (7, 80.0 / 70.0), (6, 80.0 / 65.0)]   # 第一版（对照用）


# --------------------------------------------------------------- 替身集构造
def raw_tail_windows(dom):
    """源域 tail 的**原始**（未归一）50×1024 段，按 `CLASSES` 顺序返回 (N,1024)。"""
    data, label = data_load(dom, 50, mode='tail')
    return np.stack([np.asarray(d).reshape(-1) for d in data]), np.asarray(label)


def norm_win(w):
    return ((w - w.mean()) / w.std()).astype(np.float32)


class WinDS(Dataset):
    """(N,1024) 数组 + 标签 -> 与 ResampleDataset 同形的 (x(1024,), y)。"""

    def __init__(self, W, y):
        self.W, self.y = W.astype(np.float32), np.asarray(y)

    def __len__(self):
        return len(self.W)

    def __getitem__(self, i):
        return self.W[i], int(self.y[i])


def ideal_sets(dom, up, down):
    """带限精确膨胀替身集：返回 (未膨胀集, 膨胀集, 元信息)。"""
    from scipy.signal import resample_poly
    W, y = raw_tail_windows(dom)
    seg = W.reshape(-1)                       # 51200 点连续段（无重叠拼接）
    per = len(seg) // len(CLASSES)
    out, outy = [], []
    for c in range(len(CLASSES)):
        s = seg[c * per:(c + 1) * per]
        r = resample_poly(s, up, down)
        n = len(r) // 1024
        Wc = r[:n * 1024].reshape(n, 1024)
        Wc = np.stack([norm_win(w) for w in Wc])
        out.append(Wc)
        outy.append(np.full(n, c))
    Wc = np.concatenate(out)
    yc = np.concatenate(outy)
    r = down / up                      # 时间压缩因子（>1）
    return (WinDS(np.stack([norm_win(w) for w in W]), y), WinDS(Wc, yc),
            dict(dom=dom, up=up, down=down, r=r, n=len(Wc),
                 band_lost_max=max(0.0, r * 0.45 - 0.5)))


class CrudeView(Dataset):
    """第一版口径：窗内线性插值 + 镜像补尾（仅作对照，不用于结论）。"""

    def __init__(self, ds, s):
        self.ds, self.s = ds, float(s)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.s == 1.0:
            return x, y
        a = np.asarray(x, dtype=np.float64)
        L = a.shape[-1]
        need = int(np.ceil((L - 1) * (self.s - 1.0))) + 2
        a = np.concatenate([a, a[-need:][::-1]])
        pos = np.arange(L, dtype=np.float64) * self.s
        j0 = np.floor(pos).astype(np.int64)
        fr = pos - j0
        return torch.from_numpy(((a[j0] * (1 - fr) + a[j0 + 1] * fr)
                                 .astype(np.float32))), y


def eval_sets(model, sets):
    model.eval()
    out = []
    for _, ds in sets:
        yp, yt = [], []
        for x, y in DataLoader(ds, batch_size=RC.BATCH_SIZE, shuffle=False,
                               num_workers=0):
            p = model.predict(x.to(DEVICE))
            yp.extend(p.detach().cpu().numpy().tolist())
            yt.extend(y.numpy().tolist())
        out.append(100.0 * float(np.mean(np.asarray(yp) == np.asarray(yt))))
    model.train()
    return out


def ref_points(path):
    pts = []
    if not os.path.exists(path):
        return pts
    for ln in open(path, encoding='utf-8', errors='ignore'):
        t = ln.strip()
        if t.startswith('tgt_acc:'):
            pts.append(float(t.split(':')[1]) * 100.0)
    return pts


def main():
    t0 = time.time()
    L = []
    L.append('# §6.5 L3 重做：带限精确膨胀的精度阶梯（含线性插值对照）')
    L.append('')
    L.append('生成脚本 `logs/_s65_ladder2.py` ｜ HUST T9（目标 80 Hz）｜ seed %d ｜ '
             '交付配置 480 步 ｜ `AGM_FORCE_PY_SCAN=1`' % SEED)
    L.append('')

    # ---- 评估集（**在 setup_seed 之前**构造，避免挪动随机数流）
    print('[data] building ideal stand-in sets ...', flush=True)
    ideal, meta = [], []
    for dom, up, down in LAD2:
        raw_ds, dil_ds, m = ideal_sets(dom, up, down)
        ideal.append(('raw_D%d' % dom, raw_ds))
        ideal.append(('D%d_ideal_r%.4f' % (dom, m['r']), dil_ds))
        meta.append(m)
        print('  D%d: raw %d 窗 / 膨胀 r=%.4f (up=%d down=%d) -> %d 窗' % (
            dom, len(raw_ds), m['r'], up, down, m['n']), flush=True)

    # 第一版口径（窗内线性插值）的对照集：用 pipeline 的 tail 划分
    def pipeline_target(dom):
        task = [[d for d in range(10) if d != dom], [dom]]
        _, tar, _ = RC.load_data('HUST', task)
        return RC.ResampleDataset(tar)

    crude = []
    for dom, s in LAD1:
        ds = pipeline_target(dom)
        crude.append(('D%d_lerp_%.4f' % (dom, s), CrudeView(ds, s)))
        if dom == LAD2[0][0]:
            crude.append(('D%d_pipe_raw' % dom, ds))

    # ---- 训练（与 run_one 同序：seed -> data -> config -> model -> loaders）
    RC.setup_seed(SEED)
    srcs, tar, num_cls = RC.load_data('HUST', RC.TASKS['HUST']['T9'])
    srcs = [RC.ResampleDataset(s) for s in srcs]
    tar = RC.ResampleDataset(tar)
    iters = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    _M.configs = RS._delivered_configs()
    cfg = RC.make_config(_M, 'AGMamba', num_cls, RC.EPOCHS * iters, iters,
                         DEVICE, n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = getattr(_M, RC.METHOD_MODELS['AGMamba'])(cfg)
    model.to(DEVICE)
    print('[model] init_mode=%s conv1.requires_grad=%s' % (
        model.model.front.init_mode,
        model.model.front.conv1.weight.requires_grad), flush=True)

    logger = logging.getLogger('s65_ladder2')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(_ROOT, 'logs',
                                          '_s65_arm_trained2.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    train_loaders = [InfiniteDataLoader(s, RC.BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=RC.BATCH_SIZE, shuffle=False,
                             num_workers=0)
    train_iter = zip(*train_loaders)
    print('[train] %d steps ...' % (RC.EPOCHS * iters), flush=True)
    t1 = time.time()
    res = model.train_model(train_iter, [test_loader], logger)
    acc = np.asarray(res['acces'], dtype=float) * 100.0
    print('[train] done %.0fs  best=%.2f last5=%.2f' % (
        time.time() - t1, acc[:, 0].max(), acc[-5:, 0].mean()), flush=True)

    torch.save(model.state_dict(), OUT_PT)

    # ---- 溯源自检
    rp = ref_points(REF_LOG)
    dd = (float(np.max(np.abs(np.array(rp) - acc[:, 0])))
          if len(rp) == len(acc) else float('nan'))
    L.append('## 0. 溯源自检（训练轨迹 vs 交付参照 `aln_noall` ≡ pub, seed0）')
    L.append('')
    L.append('- 评估点数 %d vs %d；逐点 max|Δ| = %.6f -> %s'
             % (len(rp), len(acc), dd,
                '**逐位同源**' if dd < 1e-9 else '不同源（差 %.4f 点）' % dd))
    L.append('- 本臂 best = %.2f、last5 = %.2f（参照 last5 见 §6.4 / Table 7）'
             % (float(acc[:, 0].max()), float(acc[-5:, 0].mean())))
    L.append('')

    # ---- 评测
    print('[eval] ...', flush=True)
    sets = [('D9_real_80Hz', tar)] + ideal + crude
    vals = eval_sets(model, [s for s in sets])
    L.append('## 1. 精度阶梯（末权重、单种子；同一权重评估全部集合）')
    L.append('')
    L.append('| 评估集 | 构造 | 梳齿位置 | 窗数 | 精度 % |')
    L.append('| --- | --- | --- | --- | --- |')
    meta_by_dom = {m['dom']: m for m in meta}
    for (nm, ds), v in zip(sets, vals):
        if nm.startswith('D9_real'):
            how, comb = 'pipeline tail（真实 80 Hz 记录）', '80 Hz（真实）'
        elif nm.startswith('raw_D'):
            d = int(nm[5:])
            how = 'pipeline 同源的原始 tail 窗'
            comb = '%.0f Hz（未膨胀）' % SPEED[d]
        elif '_ideal_r' in nm:
            d = int(nm[1:nm.index('_ideal_r')])
            m = meta_by_dom[d]
            how = '连续段 `resample_poly(%d/%d)` 带限重采样' % (m['up'], m['down'])
            comb = '%.0f Hz ×%.4f → 80 Hz' % (SPEED[d], m['r'])
        else:
            how = '窗内线性插值 + 镜像补尾（第一版口径）'
            comb = '—'
        L.append('| `%s` | %s | %s | %d | **%.2f** |'
                 % (nm, how, comb, len(ds), v))
    L.append('')

    def val(nm):
        return vals[[n for n, _ in sets].index(nm)]

    L.append('按源域分解（正数 = 变差）：')
    L.append('')
    for dom, up, down in LAD2:
        m = meta_by_dom[dom]
        rawn = 'raw_D%d' % dom
        diln = 'D%d_ideal_r%.4f' % (dom, m['r'])
        lerpn = 'D%d_lerp_%.4f' % (dom, TARGET_F / SPEED[dom])
        L.append('- **D%d（%.0f Hz）**：未膨胀 %.2f → 带限膨胀到 80 Hz **%.2f**'
                 '（膨胀代价 %+.2f）｜同 ρ 的线性插值口径 %.2f（%+.2f）'
                 '｜真实 80 Hz 目标 %.2f'
                 % (dom, SPEED[dom], val(rawn), val(diln),
                    val(diln) - val(rawn), val(lerpn),
                    val(lerpn) - val(rawn), val('D9_real_80Hz')))
    L.append('')
    L.append('线性插值口径 vs 带限口径之差（= 插值低通伪影）：%s'
             % ', '.join('%s %+.2f' % (('D%d' % d),
                                      val('D%d_lerp_%.4f' % (d, TARGET_F / SPEED[d]))
                                      - val('D%d_ideal_r%.4f' % (d, 80.0 / SPEED[d])))
                         for d, u, dn in LAD2))
    L.append('')
    L.append('管道一致性：pipeline 口径的未膨胀 D8 %.2f vs 原始口径 %.2f'
             % (val('D8_pipe_raw'), val('raw_D8')))
    L.append('')

    np.savez(OUT_NPZ,
             names=np.array([n for n, _ in sets]),
             acc=np.array(vals),
             meta=np.array([[m['dom'], m['up'], m['down'], m['r'], m['n']]
                            for m in meta], dtype=float),
             train_acc=acc)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L).encode('ascii', 'replace').decode(), flush=True)
    print('[done] %.0fs -> %s' % (time.time() - t0, OUT_MD), flush=True)


if __name__ == '__main__':
    main()
