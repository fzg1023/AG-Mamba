#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§6.6 效率测量：21 方法的 参数量 / 单窗 FLOPs / 推理时延。

⚠ 本文件 2026-10-01 重写（原版已归档到
   `docs/legacy/exp_e6_efficiency_v0.2-era.py`）。原版失效的两个原因：
   1. `import run_dgfdb as R` / `import run_final_eval as FE` —— 这两个模块
      已改名为 `experiments/run_cell.py` / `experiments/run_stage.py`；
   2. 原版靠 `run_final_eval` 的 `DEFAULTS` + `auto_final.json` 变体来拼交付
      配置 —— 现在交付配置就是 `agmamba/configs.yaml` 本身
      （`gaborwide` / `gabor_alias=2.0` / `use_ssm=True` / `tau_mode=zero` /
      `w_mmd=w_adv=0`），不需要额外覆盖。
   另外原版只覆盖 13 个方法且其中 4 个（ADN/IEDGNet/DGNIS/CDDG）测不出，并注明
   "测量时 GPU 上有两个训练任务在跑，数值应视为上界"。本版**覆盖全部 21 方法**
   并在空闲 GPU 上重测。

测量协议（可写进正文 §5.5）
--------------------------
· 构造路径与主对比**完全一致**：`run_cell.load_data` + `run_cell.make_config`
  + `METHOD_MODELS[m](cfg)`，即"跑主对比时真正训练的那个对象"。
· 数据设置：HUST（9 源域、3 类、1024 点），输入张量 `(64, 1, 1024)`，`cuda`，
  `eval()` + `no_grad()`；20 次预热后 200 次前向，CUDA event 计时取**中位数**。
· 参数量 = `sum(p.numel() for p in obj.parameters())`（对"训练器类"即其全部子模块）。
· FLOPs 用 `torch.utils.flop_counter.FlopCounterMode` 实测（非解析估计），
  单样本一次前向。
· **前向入口透明化**：不同方法的前向 API 不统一（有的整网 `forward`，有的
  `predict(x)`，有的只暴露特征提取子模块）。本脚本按固定优先级探测并**在表中
  记录实际调用的入口名**，避免"拿子模块时延冒充整网时延"。

输出：docs/manuscript/_s66_efficiency.md
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import run_cell as R                                    # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DS = 'HUST'
TASK = 'T0'
# 与 run_cell.METHOD_MODELS 同步（顺序 = 主对比的批次顺序）
METHODS = ['CNNC', 'ADN', 'IEDGNet', 'DGNIS', 'WhiteningNet', 'CCN', 'CCDG',
           'CDDG', 'DDGFD', 'AGMamba', 'UDTL', 'WIDAN', 'MRA_SNN', 'DANN',
           'AGG', 'MMD', 'CORAL', 'Triplet_loss', 'Mixup', 'MLDG', 'DAEL']
# 前向入口探测优先级：先试"可单参调用"的子模块/方法名。
# 注：各方法 API 不统一，有的是整网 forward，有的是 predict(x)，
#     有的只暴露局部前向（CDDG 的 forward_penul_fv / forward_zd_fv）。
ENTRY_ATTRS = ['model', 'net', 'network', 'predict', 'forward',
               'forward_penul_fv', 'forward_zd_fv', 'forward1',
               '_forward', 'fe_inv', 'feature_extractor', 'fe',
               'encoder', 'encoder_h', 'backbone']
L = []
P = L.append


def build_model(method, num_cls, n_src, data_len, steps, iters):
    """按 run_cell/run_compare 的同一路径实例化（= 主对比里真正被训练的对象）。

    构造后统一 `.to(DEV)`：多数方法在 `__init__` 里已把子模块 `.to(device)`，
    但 **CDDG 只在 `train_model()` 里才 `self.to(self.device)`**，构造后参数仍在
    CPU —— 不补这一步会报
    `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)`。
    """
    mod = importlib.import_module(R.METHOD_MODULES[method])
    cfg = R.make_config(mod, method, num_cls, steps, iters, DEV, n_src=n_src)
    cfg.n_domains = n_src
    cfg.data_length = data_len
    cls = getattr(mod, R.METHOD_MODELS[method])
    obj = cls(cfg)
    return obj.to(DEV)


def find_entry(obj, x, verbose=True):
    """返回 (可调用对象, 入口名)。入口名进表，保证测量依据透明。

    逐个试 `ENTRY_ATTRS` 里的单参调用；对于需要额外参数的 `forward/_forward`
    （如 `forward(x, labels)`），再补一次两参调用 `(x, zeros)`。
    全部失败时把每个候选的异常打出来，便于定位。
    """
    errs = []
    for name in ENTRY_ATTRS:
        fn = getattr(obj, name, None)
        if fn is None or not callable(fn):
            continue
        for extra in ((), (torch.zeros(x.shape[0], dtype=torch.long,
                                      device=x.device),)):
            try:
                with torch.no_grad():
                    fn(x, *extra)
                return (lambda y, fn=fn: fn(y, *extra)), name
            except Exception as e:
                errs.append('%s%s: %s' % (name, '(x,*a)' if extra else '(x)',
                                          str(e)[:70]))
    if verbose:
        print('     候选失败明细：' + ' | '.join(errs[:6]), flush=True)
    raise RuntimeError('找不到可单参前向入口（候选 %s）' % '/'.join(ENTRY_ATTRS))


def n_params(obj):
    return sum(p.numel() for p in obj.parameters())


def flops_of(fn, x):
    """FlopCounterMode 实测一次前向的 FLOPs；失败返回 nan。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                fn(x)
        return float(fc.get_total_flops())
    except Exception:
        return float('nan')


def latency_ms(fn, x, warm=20, rep=200):
    """CUDA event 计时，返回 (中位数, 最小值, 第90百分位) ms。"""
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(rep):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            fn(x)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
    return float(np.median(ts)), float(np.min(ts)), float(np.percentile(ts, 90))


def gpu_clocks():
    """记录 GPU 时钟/功耗/温度——时延绝对值随时钟状态漂移，必须随结果一起留证。"""
    try:
        q = ('clocks.sm,clocks.max.sm,power.draw,temperature.gpu,'
             'utilization.gpu,pstate')
        r = subprocess.run(['nvidia-smi', '--query-gpu=' + q,
                            '--format=csv,noheader'], capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return 'nvidia-smi 不可用: %s' % str(e)[:60]


def ssm_backend_tag():
    """当前进程实际用的扫描后端（开关是模块导入时读的，见 models/agmamba_backbone.py）。"""
    return ('py_scan(纯 PyTorch，确定性；主对比强制使用)'
            if os.environ.get('AGM_FORCE_PY_SCAN', '') == '1'
            else 'official(mamba_ssm CUDA 内核；交付默认)')


def latency_probe_json(path, num_cls, n_src, data_len, steps, iters,
                       bs, warm, rep, repeats):
    """只测本文方法的时延（重复 repeats 次）+ 记录 GPU 时钟，写 JSON。"""
    clk_before = gpu_clocks()
    obj = build_model('AGMamba', num_cls, n_src, data_len, steps, iters)
    obj.eval()
    net = getattr(obj, 'model', obj)
    x = torch.randn(bs, 1, data_len, device=DEV)
    x1 = torch.randn(1, 1, data_len, device=DEV)
    fn, entry = find_entry(net, x)
    p = n_params(obj)
    f = flops_of(fn, x1)
    runs = []
    for i in range(repeats):
        med, mn, p90 = latency_ms(fn, x, warm, rep)
        runs.append({'median': med, 'min': mn, 'p90': p90})
        print('    run%d: med=%.3f min=%.3f p90=%.3f ms' % (i + 1, med, mn, p90),
              flush=True)
    out = {'backend': ssm_backend_tag(), 'entry': entry, 'params': p,
           'flops': f, 'bs': bs, 'warm': warm, 'rep': rep,
           'runs': runs,
           'median_of_medians': float(np.median([r['median'] for r in runs])),
           'spread': float(max(r['median'] for r in runs)
                           - min(r['median'] for r in runs)),
           'clocks_before': clk_before, 'clocks_after': gpu_clocks()}
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print('[probe] %s  backend=%s  med_of_med=%.3f ms'
          % (path, out['backend'], out['median_of_medians']), flush=True)


def group_params(net):
    """把交付网络按功能分组计参（仅对 AGMamba 有意义）。"""
    g = {}
    for name, p in net.named_parameters():
        key = ('frontend(conv1 bank)' if name.startswith('front.conv1')
               else 'frontend(other)' if name.startswith('front')
               else 'SSM blocks' if ('ssm' in name or 'mamba' in name
                                     or 'blocks' in name)
               else 'head' if name.startswith('head')
               else 'projection/norm/pos' if ('proj' in name or 'norm' in name
                                              or 'pos' in name)
               else 'other')
        g[key] = g.get(key, 0) + p.numel()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--warm', type=int, default=20)
    ap.add_argument('--rep', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(
        ROOT, 'docs', 'manuscript', '_s66_efficiency.md'))
    ap.add_argument('--json', default=None,
                    help='仅测本文方法时延并把结果写成 JSON（内部复核用）')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--num-cls', type=int, default=None)
    ap.add_argument('--n-src', type=int, default=None)
    a = ap.parse_args()

    # ---- 内部复核模式：只测本文方法（由主流程以不同 SSM 后端环境变量调起）----
    if a.json:
        num_cls = a.num_cls if a.num_cls is not None else 9
        n_src = a.n_src if a.n_src is not None else 9
        it = max(1, (450 + R.BATCH_SIZE - 1) // R.BATCH_SIZE)
        latency_probe_json(a.json, num_cls, n_src, R.DATA_LENGTH,
                           R.EPOCHS * it, it, a.bs, a.warm, a.rep, a.repeats)
        return

    print('device=%s  torch=%s' % (DEV, torch.__version__), flush=True)
    srcs, tar, num_cls = R.load_data(DS, R.TASKS[DS][TASK])
    n_src = len(srcs)
    data_len = R.DATA_LENGTH
    iters = max(1, (len(srcs[0]) + R.BATCH_SIZE - 1) // R.BATCH_SIZE)
    steps = R.EPOCHS * iters
    print('设置：%s/%s  源域=%d  类数=%d  窗长=%d  steps=%d'
          % (DS, TASK, n_src, num_cls, data_len, steps), flush=True)

    x = torch.randn(a.bs, 1, data_len, device=DEV)
    x1 = torch.randn(1, 1, data_len, device=DEV)

    P('# §6.6 效率测量（21 方法：参数量 / 单窗 FLOPs / 推理时延）')
    P('')
    P('由 `analysis/exp_e6_efficiency.py` 生成（2026-10-01 重写版）。')
    P('')
    P('- 构造路径与主对比**完全一致**（`run_cell.load_data` + `make_config` '
      '+ `METHOD_MODELS[m](cfg)`）')
    P('- 设置：`%s/%s`，%d 源域、%d 类、%d 点窗；配置 = '
      '`agmamba/configs.yaml`（交付配置：`gaborwide`/`alias=2.0`/`use_ssm`/'
      '`tau_mode=zero`/`w_mmd=w_adv=0`）' % (DS, TASK, n_src, num_cls, data_len))
    P('- 时延：`eval()` + `no_grad()`，批 %d，%d 次预热 + %d 次前向，'
      'CUDA event 计时取**中位数**；吞吐 = 批 64 / 中位数'
      % (a.bs, a.warm, a.rep))
    P('- **必须先 `eval()`**：train 模式下 `head_dropout`(0.3) 生效，时延与方差'
      '都会被高估（本脚本早期版本漏了这一步，已在重跑中修正）')
    P('- FLOPs：`torch.utils.flop_counter.FlopCounterMode` **实测**（单样本一次'
      '前向，非解析估计）')
    P('- **「入口」列 = 实际被计时的前向入口**。各方法 API 不统一，有的只暴露'
      '特征提取子模块；该列用于防止把"子模块时延"误读成"整网时延"')
    P('- 所有方法构造后统一 `.to(cuda)`（`CDDG` 原本要到 `train_model()` 才'
      '`self.to(device)`，否则参数留在 CPU）')
    P('')

    # ---------------- (a) AGMamba 参数分组 ----------------
    P('## (a) 本方法（交付配置）的参数分组与单窗 FLOPs')
    P('')
    obj = build_model('AGMamba', num_cls, n_src, data_len, steps, iters)
    net = None
    for attr in ('model', 'net', 'network'):
        sub = getattr(obj, attr, None)
        if isinstance(sub, nn.Module):
            net = sub
            break
    if net is not None:
        net.eval()                       # §5.4：时延/FLOPs 一律在 eval() 下测
        groups = group_params(net)
        tot = sum(groups.values())
        P('| 模块 | 参数量 | 占比 |')
        P('|---|---|---|')
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            P('| %s | %s | %.1f%% |' % (k, format(v, ','), 100.0 * v / tot))
        P('| **合计** | **%s** | 100%% |' % format(tot, ','))
        P('')
        fn_net, entry_net = find_entry(net, x1)
        f = flops_of(fn_net, x1)
        if f == f:
            P('单窗（%d 点）一次前向总 FLOPs = **%.1f MFLOPs**（入口 `%s`）。'
              % (data_len, f / 1e6, entry_net))
        else:
            P('单窗 FLOPs 实测失败（入口 `%s`）。' % entry_net)
        P('')
        sys.stdout.flush()
    else:
        P('（未找到 AGMamba 的 `model` 子模块，跳过参数分组。）')
        P('')

    # ---------------- (b) 21 方法 ----------------
    P('## (b) 参数量 / 单窗 FLOPs / 推理时延（%d 方法）' % len(METHODS))
    P('')
    P('| 方法 | 入口 | 参数量 | 单窗 FLOPs (M) | 中位数 (ms) | 最小值 (ms) '
      '| 第90百分位 (ms) | 吞吐@中位数 (窗/s) |')
    P('|---|---|---|---|---|---|---|---|')
    rows = []
    for m in METHODS:
        t0 = time.time()
        try:
            o = build_model(m, num_cls, n_src, data_len, steps, iters)
            o.eval()                     # §5.4：一定在 eval() 下测（否则 dropout 生效）
            p = n_params(o)
            fn, entry = find_entry(o, x)
            f = flops_of(fn, x1)
            med, mn, p90 = latency_ms(fn, x, a.warm, a.rep)
            thr = a.bs / (med / 1e3)
            rows.append((m, entry, p, f, med, mn, p90, thr))
            P('| %s | `%s` | %s | %s | %.2f | %.2f | %.2f | %.0f |'
              % (m, entry, format(p, ','),
                 ('%.1f' % (f / 1e6)) if f == f else '–',
                 med, mn, p90, thr))
            print('  %-14s entry=%-10s params=%-10s flops=%-8s med=%.2f ms (%.0fs)'
                  % (m, entry, format(p, ','),
                     ('%.1fM' % (f / 1e6)) if f == f else '–', med,
                     time.time() - t0), flush=True)
            del o, fn
        except Exception as e:
            P('| %s | – | – | – | – | – | – | – |' % m)
            print('  %-14s FAILED: %s' % (m, str(e)[:110]), flush=True)
        torch.cuda.empty_cache()
    P('')
    P('> 时延在**空闲 GPU** 上测得（主对比批次已于 2026-10-01 12:21 结束），'
      '批次跑动中测得的旧值（见 `_s66_data.md`）不再使用。')
    P('')

    # ---------------- (c) SSM 分支的时延占比 ----------------
    P('## (c) 选择性扫描（SSM 分支）的时延占比')
    P('')
    try:
        import agmamba.method as M
        base = bool(M.configs.use_ssm)
        o1 = build_model('AGMamba', num_cls, n_src, data_len, steps, iters)
        o1.eval()
        net1 = getattr(o1, 'model', o1)
        fn1, e1 = find_entry(net1, x)
        med_on, mn_on, _ = latency_ms(fn1, x, a.warm, a.rep)
        p_on = n_params(o1)
        M.configs.use_ssm = False
        o2 = build_model('AGMamba', num_cls, n_src, data_len, steps, iters)
        o2.eval()
        net2 = getattr(o2, 'model', o2)
        fn2, e2 = find_entry(net2, x)
        med_off, mn_off, _ = latency_ms(fn2, x, a.warm, a.rep)
        p_off = n_params(o2)
        M.configs.use_ssm = base
        P('同一架构关掉 SSM 分支（`use_ssm=False`）：')
        P('')
        P('| 配置 | 参数量 | 中位数 (ms) | 最小值 (ms) |')
        P('|---|---|---|---|')
        P('| `use_ssm=True`（交付） | %s | %.2f | %.2f |'
          % (format(p_on, ','), med_on, mn_on))
        P('| `use_ssm=False` | %s | %.2f | %.2f |'
          % (format(p_off, ','), med_off, mn_off))
        P('')
        P('→ 扫描分支占时延 **%.1f%%**（中位数），参数量差 %s（占 %.1f%%）。'
          % (100.0 * (med_on - med_off) / med_on,
             format(p_on - p_off, ','), 100.0 * (p_on - p_off) / p_on))
        print('  SSM 时延占比 %.1f%%' % (100.0 * (med_on - med_off) / med_on),
              flush=True)
        del o1, o2, fn1, fn2
    except Exception as e:
        P('（SSM 对照失败：%s）' % str(e)[:120])
    P('')

    # ---------------- (d) 本文方法时延复核：两种 SSM 后端 × repeats 次 -------------
    P('## (d) 本文方法时延复核：两种 SSM 后端 × %d 次重复 + GPU 时钟' % a.repeats)
    P('')
    P('扫描后端的开关（`models/agmamba_backbone.py` 的 `AGM_FORCE_PY_SCAN`）是'
      '**模块导入时**读的，所以两种后端必须在两个进程里分别测。')
    P('')
    P('- `official` = `mamba_ssm` 官方 CUDA 选择性扫描内核 —— **交付默认**')
    P('- `py_scan` = 纯 PyTorch 分块扫描 —— 主对比为了**梯度逐位可复现**'
      '（官方内核反传含原子累加）而强制启用，代价只落在本方法上')
    P('')
    P('| 后端 | 参数量 | 单窗 FLOPs (M) | %d 次中位数 (ms) | 极差 (ms) '
      '| 中位数中的中位数 (ms) | 吞吐 (窗/s) |' % a.repeats)
    P('|---|---|---|---|---|---|---|')
    probe = {}
    for tag, force in [('official', None), ('py_scan', '1')]:
        js = os.path.join(ROOT, 'logs', '_latency_probe_%s.json' % tag)
        env = dict(os.environ)
        if force:
            env['AGM_FORCE_PY_SCAN'] = force
        else:
            env.pop('AGM_FORCE_PY_SCAN', None)
        cmd = [sys.executable, os.path.abspath(__file__), '--json', js,
               '--repeats', str(a.repeats), '--bs', str(a.bs),
               '--warm', str(a.warm), '--rep', str(a.rep),
               '--num-cls', str(num_cls), '--n-src', str(n_src)]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=1800)
            for line in r.stdout.split('\n'):
                if line.startswith('    run') or line.startswith('[probe]'):
                    print('  [%s] %s' % (tag, line.strip()), flush=True)
            if r.returncode != 0:
                print('  [%s] 子进程失败: %s' % (tag, r.stderr[-200:]), flush=True)
                continue
            d = json.load(open(js))
            probe[tag] = d
            med_mm = d['median_of_medians']
            P('| %s | %s | %s | %s | %.3f | **%.3f** | %.0f |'
              % (tag, format(d['params'], ','),
                 ('%.1f' % (d['flops'] / 1e6)) if d['flops'] == d['flops'] else '–',
                 ' / '.join('%.3f' % x['median'] for x in d['runs']),
                 d['spread'], med_mm, a.bs / (med_mm / 1e3)))
        except Exception as e:
            P('| %s | – | – | – | – | – | – |' % tag)
            print('  [%s] 异常 %s' % (tag, str(e)[:100]), flush=True)
    P('')
    for tag, d in probe.items():
        P('- `%s`：入口 `%s`，GPU 时钟（测前）`%s`、（测后）`%s`'
          % (tag, d['entry'], d['clocks_before'], d['clocks_after']))
    if len(probe) == 2:
        mo = probe['official']['median_of_medians']
        mp = probe['py_scan']['median_of_medians']
        P('')
        P('→ 官方内核 vs 纯 PyTorch 扫描：**%.2f ms vs %.2f ms（%.2f×）**。'
          '正文的效率表应引用 **official**（交付实现）；`py_scan` 仅为主对比'
          '的确定性折衷，需在脚注说明。' % (mo, mp, mp / mo))
    P('')

    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('[efficiency] %s  （成功 %d/%d 个方法）'
          % (a.out, len(rows), len(METHODS)), flush=True)


if __name__ == '__main__':
    main()
