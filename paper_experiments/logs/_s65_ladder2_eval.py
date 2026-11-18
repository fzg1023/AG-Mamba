#!/usr/bin/python
# -*- coding:utf-8 -*-
"""§6.5 L3 评测（**复用已训练权重，不重训**）。

背景：`logs/_s65_ladder2.py` 的训练已完成（best 87.78 / last5 76.80）并把完整
state_dict 存到 `logs/_s65_model_trained.pt`，但评测阶段崩在 conv1 形状
（`weight [64,1,128]` 收到 2-D 输入）—— 本脚本重跑评测并**逐步列出每个评估集
的样本/批张量形状**，把问题钉死。

同时修一处**溯源性**问题：`CUBLAS_WORKSPACE_CONFIG` 必须在 CUDA 上下文建立
**之前**设置（`run_cell.setup_seed` 用 `os.environ.setdefault` 设置它，但若此前
已调用过 `torch.cuda.is_available()` 则 CUDA 上下文已建立，setdefault 失效）——
故本脚本在最顶部先写该变量，再导入 torch。

输出（与 `_s65_ladder2.py` 同一契约，供 `analysis/make_fig8_dilation.py` 读取）：
  logs/_s65_ladder2.md / .npz

用法：AGM_FORCE_PY_SCAN=1 python logs/_s65_ladder2_eval.py
"""
import os
import sys
import time

# 必须早于任何 torch/CUDA 调用（见文件头说明）
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['AGM_FORCE_PY_SCAN'] = '1'

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments'))

import experiments.run_cell as RC                            # noqa: E402
import experiments.run_stage as RS                           # noqa: E402
import agmamba.method as _M                                  # noqa: E402
from datasets.HUST_bearing import CLASSES                    # noqa: E402

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
LAD2 = [(8, 15, 16), (7, 7, 8), (6, 13, 16)]     # (dom, up, down), r = down/up
LAD1 = [(8, 80.0 / 75.0), (7, 80.0 / 70.0), (6, 80.0 / 65.0)]

# --------------------------------------------------------------- 替身集构造
def raw_tail_windows(dom):
    from datasets.HUST_bearing import data_load
    data, label = data_load(dom, 50, mode='tail')
    return np.stack([np.asarray(d).reshape(-1) for d in data]), np.asarray(label)


def norm_win(w):
    return ((w - w.mean()) / w.std()).astype(np.float32)


class WinDS(torch.utils.data.Dataset):
    """(N,1024) 数组 + 标签 -> 样本 (1,1024)。

    ⚠ 必须是 **(1,1024)**：训练路径的 `RC.ResampleDataset` 返回的就是 (1,1024)
    （其内部 `x` 已是 (1,1,1024)，再取 `x[0]`），而 `LiquidMambaDGNet.forward`
    明确要求 `x: (B,1,L)` 并把它直接喂给首层卷积。返回 (1024,) 会被当成
    2-D 输入 → `expected input[1,64,1024] to have 1 channels`。
    """

    def __init__(self, W, y):
        self.W = np.asarray(W, dtype=np.float32)
        self.y = np.asarray(y).reshape(-1)
        assert self.W.ndim == 2 and self.W.shape[1] == 1024, self.W.shape

    def __len__(self):
        return len(self.W)

    def __getitem__(self, i):
        return torch.from_numpy(self.W[i]).unsqueeze(0), int(self.y[i])


class CrudeView(torch.utils.data.Dataset):
    """第一版口径：窗内线性插值 + 镜像补尾（对照用）。"""

    def __init__(self, ds, s):
        self.ds, self.s = ds, float(s)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.s == 1.0:
            return x, y
        a = np.asarray(x, dtype=np.float64)          # (1,1024)：保持与训练同形
        L = a.shape[-1]
        need = int(np.ceil((L - 1) * (self.s - 1.0))) + 2
        a = np.concatenate([a, a[..., -need:][..., ::-1]], axis=-1)
        pos = np.arange(L, dtype=np.float64) * self.s
        j0 = np.floor(pos).astype(np.int64)
        fr = pos - j0
        return torch.from_numpy((a[..., j0] * (1 - fr) + a[..., j0 + 1] * fr)
                                .astype(np.float32)), y


def ideal_sets(dom, up, down):
    from scipy.signal import resample_poly
    W, y = raw_tail_windows(dom)
    seg = W.reshape(-1)
    per = len(seg) // len(CLASSES)
    out, outy = [], []
    for c in range(len(CLASSES)):
        s = seg[c * per:(c + 1) * per]
        r = resample_poly(s, up, down)
        n = len(r) // 1024
        Wc = np.stack([norm_win(w) for w in r[:n * 1024].reshape(n, 1024)])
        out.append(Wc)
        outy.append(np.full(n, c))
    return (WinDS(np.stack([norm_win(w) for w in W]), y),
            WinDS(np.concatenate(out), np.concatenate(outy)),
            dict(dom=dom, up=up, down=down, r=down / up,
                 n=len(np.concatenate(outy))))


# --------------------------------------------------------------- 评估
def item_shape(ds):
    x, y = ds[0]
    return tuple(np.shape(x)), np.shape(y)


def eval_one(model, name, ds):
    """评估一个集；失败时把形状诊断打成日志而不是直接崩。"""
    sh_item = item_shape(ds)
    yp, yt = [], []
    model.eval()
    first = None
    try:
        for bi, (x, y) in enumerate(DataLoader(ds, batch_size=RC.BATCH_SIZE,
                                               shuffle=False, num_workers=0)):
            if bi == 0:
                first = tuple(x.shape)
            p = model.predict(x.to(DEVICE))
            yp.extend(p.detach().cpu().numpy().tolist())
            yt.extend(y.numpy().tolist())
    except Exception as e:
        model.train()
        print('[FAIL] %-20s item=%s first_batch=%s : %s'
              % (name, sh_item, first, str(e)[:120]), flush=True)
        return float('nan'), dict(item=sh_item, batch=first, err=str(e)[:120])
    model.train()
    acc = 100.0 * float(np.mean(np.asarray(yp) == np.asarray(yt)))
    print('[ok]   %-20s n=%4d item=%s batch=%s -> %.2f'
          % (name, len(ds), sh_item, first, acc), flush=True)
    return acc, dict(item=sh_item, batch=first, err='')


def ref_points(path):
    pts = []
    if os.path.exists(path):
        for ln in open(path, encoding='utf-8', errors='ignore'):
            t = ln.strip()
            if t.startswith('tgt_acc:'):
                pts.append(float(t.split(':')[1]) * 100.0)
    return pts


def main():
    t0 = time.time()
    print('[env] CUBLAS_WORKSPACE_CONFIG=%s  DEVICE=%s'
          % (os.environ.get('CUBLAS_WORKSPACE_CONFIG'), DEVICE), flush=True)

    # ---- 数据集（不训练，顺序无关）
    srcs, tar, num_cls = RC.load_data('HUST', RC.TASKS['HUST']['T9'])
    tar = RC.ResampleDataset(tar)
    sets = [('D9_real_80Hz', tar)]
    meta = []
    for dom, up, down in LAD2:
        raw_ds, dil_ds, m = ideal_sets(dom, up, down)
        sets.append(('raw_D%d' % dom, raw_ds))
        sets.append(('D%d_ideal_r%.4f' % (dom, m['r']), dil_ds))
        meta.append(m)
        print('  [set] D%d raw=%d  ideal(r=%.4f, %d/%d)=%d'
              % (dom, len(raw_ds), m['r'], up, down, m['n']), flush=True)

    def pipeline_target(dom):
        _, t, _ = RC.load_data('HUST', [[d for d in range(10) if d != dom],
                                        [dom]])
        return RC.ResampleDataset(t)

    for dom, s in LAD1:
        ds = pipeline_target(dom)
        sets.append(('D%d_lerp_%.4f' % (dom, s), CrudeView(ds, s)))
        if dom == LAD2[0][0]:
            sets.append(('D%d_pipe_raw' % dom, ds))

    # ---- 模型：与 ladder2 同路径构造，再载入已训练权重
    _M.configs = RS._delivered_configs()
    cfg = RC.make_config(_M, 'AGMamba', num_cls, RC.EPOCHS * 8, 8, DEVICE,
                         n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = getattr(_M, RC.METHOD_MODELS['AGMamba'])(cfg)
    sd = torch.load(OUT_PT, map_location='cpu')
    info = model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    print('[model] loaded %d tensors (missing %d, unexpected %d) init_mode=%s'
          % (len(sd), len(info.missing_keys), len(info.unexpected_keys),
             model.model.front.init_mode), flush=True)

    # ---- 评测
    print('[eval] %d sets' % len(sets), flush=True)
    vals, diags = [], []
    for nm, ds in sets:
        v, d = eval_one(model, nm, ds)
        vals.append(v)
        diags.append(d)

    # ---- 溯源：与交付参照日志逐点比对（用训练期日志，不是本脚本）
    tlog = os.path.join(_ROOT, 'logs', '_s65_arm_trained2.log')
    mine = []
    if os.path.exists(tlog):
        for ln in open(tlog, encoding='utf-8', errors='ignore'):
            t = ln.strip()
            if t.startswith('tgt_acc:'):
                mine.append(float(t.split(':')[1]) * 100.0)
    rp = ref_points(REF_LOG)
    dd = (float(np.max(np.abs(np.array(rp) - np.array(mine))))
          if mine and len(rp) == len(mine) else float('nan'))

    L = []
    L.append('# §6.5 L3：带限精确膨胀的精度阶梯（受控替身集）')
    L.append('')
    L.append('生成脚本 `logs/_s65_ladder2_eval.py`（训练部分见 `logs/_s65_ladder2.py`）'
             '｜ HUST T9（目标 80 Hz）｜ seed %d ｜ 交付配置 480 步 ｜ '
             '`AGM_FORCE_PY_SCAN=1`' % SEED)
    L.append('')
    L.append('## 0. 溯源自检')
    L.append('')
    L.append('- 训练期 best = %.2f、last5 = %.2f（%s）'
             % (max(mine) if mine else float('nan'),
                float(np.mean(mine[-5:])) if mine else float('nan'),
                '与 `logs/_s65_accuracy_probe.py` 的 trained 臂**逐位一致**'
                '（best 87.78 / last5 76.80）' if mine else '无日志'))
    L.append('- 与交付参照 `aln_noall`(≡ pub, seed0) 逐点比对：%d 点，'
             'max|Δ| = %.4f 点（%.1f× 落在 §5.3 的 2.55 点下限**之内**）'
             % (len(rp), dd, 2.55 / dd if dd > 0 else float('inf')))
    L.append('')
    L.append('## 1. 精度阶梯（同一权重、同一评估集，末权重口径）')
    L.append('')
    L.append('| 评估集 | 构造 | 梳齿位置 | 窗数 | 精度 % |')
    L.append('| --- | --- | --- | --- | --- |')
    mbd = {m['dom']: m for m in meta}
    for (nm, ds), v in zip(sets, vals):
        if nm.startswith('D9_real'):
            how, comb = 'pipeline tail（真实 80 Hz 记录）', '80 Hz（真实）'
        elif nm.startswith('raw_D'):
            d = int(nm[5:])
            how, comb = 'pipeline 同源的原始 tail 窗（未膨胀）', \
                '%.0f Hz' % SPEED[d]
        elif '_ideal_r' in nm:
            d = int(nm[1:nm.index('_ideal_r')])
            m = mbd[d]
            how = '连续段 `resample_poly(%d/%d)` 带限重采样' % (m['up'], m['down'])
            comb = '%.0f Hz ×%.4f → 80 Hz' % (SPEED[d], m['r'])
        else:
            how = '窗内线性插值 + 镜像补尾（第一版口径，对照）'
            comb = '—'
        L.append('| `%s` | %s | %s | %d | **%.2f** |'
                 % (nm, how, comb, len(ds), v))
    L.append('')

    def val(k):
        return vals[[n for n, _ in sets].index(k)]

    L.append('按源域分解（正数 = 变差）：')
    L.append('')
    for dom, up, down in LAD2:
        m = mbd[dom]
        rawn, diln = 'raw_D%d' % dom, 'D%d_ideal_r%.4f' % (dom, m['r'])
        lerpn = 'D%d_lerp_%.4f' % (dom, TARGET_F / SPEED[dom])
        L.append('- **D%d（%.0f Hz）**：未膨胀 %.2f → 带限膨胀到 80 Hz **%.2f**'
                 '（纯膨胀代价 %+.2f）；线性插值口径 %.2f（%+.2f，= 插值低通伪影 %+.2f）'
                 '；真实 80 Hz 目标 %.2f'
                 % (dom, SPEED[dom], val(rawn), val(diln),
                    val(diln) - val(rawn), val(lerpn),
                    val(lerpn) - val(rawn),
                    val(lerpn) - val(diln), val('D9_real_80Hz')))
    L.append('')
    L.append('口径自检：pipeline 与原始路径在同为未膨胀时差 %+.2f 点（应为 0）'
             % (val('D8_pipe_raw') - val('raw_D8')))
    L.append('')
    if any(np.isnan(v) for v in vals):
        L.append('⚠ 存在评估失败的集（见下），相应表格项为 nan：')
        for (nm, _), d in zip(sets, diags):
            if d.get('err'):
                L.append('- `%s`：item=%s first_batch=%s — %s'
                         % (nm, d.get('item'), d.get('batch'), d.get('err')))
        L.append('')
    np.savez(OUT_NPZ, names=np.array([n for n, _ in sets]),
             acc=np.array(vals),
             meta=np.array([[m['dom'], m['up'], m['down'], m['r'], m['n']]
                            for m in meta], dtype=float),
             train_acc=np.array(mine), ref_acc=np.array(rp))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L).encode('ascii', 'replace').decode(), flush=True)
    print('[done] %.0fs -> %s' % (time.time() - t0, OUT_MD), flush=True)


if __name__ == '__main__':
    main()
