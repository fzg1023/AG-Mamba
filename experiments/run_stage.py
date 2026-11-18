#!/usr/bin/python
# -*- coding:utf-8 -*-
"""最终评估批次驱动器（FiLM 跨数据集验证 + HUST 基线方法）。

阶段 (stage):
  film_hust_t0    HUST T0     base/film  补 seed 3,4（seed 0-2 已有，自动跳过）
  film_hust_rest  HUST T1..T9 base/film x seed 0..4
  film_pu         PU   T0..T3 base/film x seed 0..4
  film_phm        PHM2009 T0..T3 base/film x seed 0..4
  widan_hust      WIDAN  HUST T0..T9 x seed 0..4
  dann_hust       DANN   HUST T0..T9 x seed 0..4
  udtl_hust       UDTL   HUST T0..T9 x seed 0..4
  film_all        以上三个 film 阶段依次执行
  hust_baselines  以上三个 HUST 基线阶段依次执行
  all             全部依次执行

特性:
  - 已存在且含结果的日志会**自动跳过并复用**（读取 tgt_acc 最大值），
    因此可安全地用同一脚本断点续跑。
  - 每个 (变体, 任务, 种子) 结束即打印一行，便于 tail -f 观察。

用法:
  python -u run_final_eval.py film_hust_t0
"""
import sys
import os
import re
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from agmamba import method as M  # noqa: F401  触发 configs 加载
try:
    from .run_cell import run_one
except ImportError:  # direct execution: python experiments/run_stage.py ...
    from run_cell import run_one
from agmamba import paths as _P

OUTDIR = _P.RESULTS
HUST_TASKS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
STD_TASKS = ['T0', 'T1', 'T2', 'T3']
SEEDS5 = [0, 1, 2, 3, 4]

# AG-Mamba 变体基线配置（其余开关全部复位）
DEFAULTS = {
    'film': False, 'w_cdan': 0.0, 'rsc_p': 0.0, 'mixup_alpha': 0.0,
    'w_jmmd': 0.0, 'ts_jitter': 0.0, 'front_init': 'none', 'front_mode': 'cnn',
    'freeze_init': False, 'wave_init': False,
    'w_mmd': 0.5, 'w_adv': 0.1,
    'tau_estimator': 'autocorr', 'tau_f_lo': 3, 'tau_f_hi': 120,
    'tau_mode': 'normal',
    'front_chan': 128, 'front_ksize': 64,
    'n_blocks': 3, 'd_model': 128, 'd_inner': 128,
    'steps_override': None, 'use_scheduler': False,
    'gabor_sigma': 0.5, 'gabor_f_lo': 0.005, 'gabor_f_hi': 0.45,
    'gabor_dc_frac': 0.5,
    'gabor_alias': 0.0,
    'use_ssm': True,
    'ema_decay': 0.0,
    'ema_start': 100,
}

VARIANTS = [
    {'name': 'base'},
    {'name': 'film', 'film': True},
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def read_acc(path):
    """读取日志中 tgt_acc 的历史最大值（等价 run_one 的 best 口径）。

    兼容两种格式：'tgt_acc: 0.9123'（统一格式）与 'tgt_acc 0.9123'（早期 MRA-SNN）。
    """
    if not os.path.exists(path):
        return None
    vals = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'tgt_acc:?\s*([0-9.]+)', line)
            if m:
                vals.append(float(m.group(1)))
    return max(vals) if vals else None


def lm_logpath(data, task, name, seed):
    return _P.log_path(data, task, 'AGMamba', seed, arm=name)


def base_logpath(data, task, method, seed):
    return _P.log_path(data, task, method, seed)


def _acc_of(path):
    a = read_acc(path)
    return None if a is None else a * 100.0


# ---------------------------------------------------------------------------
# AG-Mamba 变体
# ---------------------------------------------------------------------------
def run_variants(data, tasks, seeds, variants=VARIANTS):
    res = {v['name']: {} for v in variants}
    t_start = time.time()
    for task in tasks:
        for seed in seeds:
            for v in variants:
                vv = dict(v)
                name = vv.pop('name')
                path = lm_logpath(data, task, name, seed)
                acc = _acc_of(path)
                if acc is not None:
                    print('[skip] {} -> {:.2f}%'.format(
                        os.path.basename(path), acc), flush=True)
                else:
                    for k, val in DEFAULTS.items():
                        setattr(M.configs, k, val)
                    for k, val in vv.items():
                        setattr(M.configs, k, val)
                    t0 = time.time()
                    try:
                        a = run_one('AGMamba', data, task, seed, 'cuda', path)
                        acc = a * 100.0
                    except Exception as e:
                        print('[FAIL] {} : {}'.format(
                            os.path.basename(path), e), flush=True)
                        acc = float('nan')
                    print('[{}] {} -> {:.2f}% ({:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), os.path.basename(path),
                        acc, time.time() - t0), flush=True)
                res[name][(task, seed)] = acc
        # 每完成一个任务打印一次该任务小结
        print('  --- {} / {} ---'.format(data, task), flush=True)
        for name in res:
            a = np.array([res[name][(task, s)] for s in seeds], dtype=float)
            print('      {:6s}: {:.2f} +- {:.2f}  {}'.format(
                name, np.nanmean(a), np.nanstd(a),
                [round(x, 2) for x in a]), flush=True)
        print('      elapsed {:.0f}s'.format(time.time() - t_start), flush=True)

    # 总体汇总（以第一个变体为参照，兼容 base 以外的命名）
    print('=' * 78, flush=True)
    print('SUMMARY {} ({} tasks x {} seeds)'.format(data, len(tasks), len(seeds)), flush=True)
    ref_name = list(res.keys())[0]
    base_all = np.array([res[ref_name][k] for k in res[ref_name]], dtype=float)
    for name in res:
        all_v = np.array([res[name][k] for k in res[name]], dtype=float)
        per_task = [np.nanmean([res[name][(t, s)] for s in seeds])
                    for t in tasks]
        print('{:6s}: overall {:.2f} +- {:.2f} | per-task {} | {:+.2f} vs {}'.format(
            name, np.nanmean(all_v), np.nanstd(all_v),
            [round(float(x), 2) for x in per_task],
            np.nanmean(all_v) - np.nanmean(base_all), ref_name), flush=True)
    print('=' * 78, flush=True)
    return res


# ---------------------------------------------------------------------------
# 基线方法
# ---------------------------------------------------------------------------
def run_baseline(method, data, tasks, seeds=SEEDS5):
    res = {}
    t_start = time.time()
    for task in tasks:
        for seed in seeds:
            path = base_logpath(data, task, method, seed)
            acc = _acc_of(path)
            if acc is not None:
                print('[skip] {} -> {:.2f}%'.format(
                    os.path.basename(path), acc), flush=True)
            else:
                t0 = time.time()
                try:
                    a = run_one(method, data, task, seed, 'cuda', path)
                    acc = a * 100.0
                except Exception as e:
                    print('[FAIL] {} : {}'.format(
                        os.path.basename(path), e), flush=True)
                    acc = float('nan')
                print('[{}] {} -> {:.2f}% ({:.0f}s)'.format(
                    time.strftime('%H:%M:%S'), os.path.basename(path),
                    acc, time.time() - t0), flush=True)
            res[(task, seed)] = acc
        a = np.array([res[(task, s)] for s in seeds], dtype=float)
        print('  --- {} / {} / {} : {:.2f} +- {:.2f}  {} ---'.format(
            data, task, method, np.nanmean(a), np.nanstd(a),
            [round(x, 2) for x in a]), flush=True)

    all_v = np.array([res[k] for k in res], dtype=float)
    per_task = [np.nanmean([res[(t, s)] for s in seeds]) for t in tasks]
    print('=' * 78, flush=True)
    print('SUMMARY {} / {} : overall {:.2f} +- {:.2f} | per-task {} | {}s'.format(
        data, method, np.nanmean(all_v), np.nanstd(all_v),
        [round(float(x), 2) for x in per_task], time.time() - t_start), flush=True)
    print('=' * 78, flush=True)
    return res


# ---------------------------------------------------------------------------
# 阶段
# ---------------------------------------------------------------------------
def stage_film_hust_t0():
    return run_variants('HUST', ['T0'], SEEDS5)


def stage_film_hust_rest():
    return run_variants('HUST', HUST_TASKS[1:], SEEDS5)


def stage_film_pu():
    return run_variants('PU', STD_TASKS, SEEDS5)


def stage_film_phm():
    return run_variants('PHM2009', STD_TASKS, SEEDS5)


def stage_widan_hust():
    return run_baseline('WIDAN', 'HUST', HUST_TASKS)


def stage_dann_hust():
    return run_baseline('DANN', 'HUST', HUST_TASKS)


def stage_udtl_hust():
    return run_baseline('UDTL', 'HUST', HUST_TASKS)


def stage_mra_hust():
    """MRA-SNN 基线在 HUST 十任务 × 5 种子（补齐 HUST 全方法表中的空位）。

    原先只有 PHM2009 的单种子筛选结果；本阶段补齐与其余基线同协议的
    HUST 结果（60 epoch、5 种子），日志名 HUST_T{k}_MRA_SNN_seed{s}.log。
    """
    return run_baseline('MRA_SNN', 'HUST', HUST_TASKS)


def stage_tau_spec_hust():
    """τ₀ 估计器对照（HUST T0 x 5 种子）:
      tau_auto = autocorr lag[8,300]（当前默认，诊断显示在 HUST 上退化）
      tau_spec = envspec bin[2,220]（诊断显示方向正确、斜率≈-1）
    """
    variants = [
        {'name': 'tau_auto', 'tau_estimator': 'autocorr',
         'tau_f_lo': 3, 'tau_f_hi': 120},
        {'name': 'tau_spec', 'tau_estimator': 'envspec',
         'tau_f_lo': 2, 'tau_f_hi': 220},
    ]
    return run_variants('HUST', ['T0'], SEEDS5, variants)


def stage_tau_spec_phm():
    """τ₀ 估计器对照（PHM2009 T0 x 5 种子，验证跨数据集鲁棒性）。"""
    variants = [
        {'name': 'tau_auto', 'tau_estimator': 'autocorr',
         'tau_f_lo': 3, 'tau_f_hi': 120},
        {'name': 'tau_spec', 'tau_estimator': 'envspec',
         'tau_f_lo': 2, 'tau_f_hi': 220},
    ]
    return run_variants('PHM2009', ['T0'], SEEDS5, variants)


def stage_ablate_tau_hust():
    """E3 因果消融（HUST T0 x 5 种子）—— 检验「液体时间常数」是真机制还是包装。

      abl_base    = 不干预（同批次对照，不含 film）
      abl_zero    = τ₀ ≡ 0，等价于冻结 γ=0，**彻底切断** γ·τ₀ 通路
      abl_shuffle = τ₀ 在 batch 内随机置换（保留量级，打乱样本对应）
      abl_const   = τ₀ 替换为 batch 均值（保留通路，去掉样本级信息）

    判读：若 abl_zero 与 abl_base 无显著差异 -> τ₀ 通路对性能无贡献。
    """
    variants = [
        {'name': 'abl_base', 'tau_mode': 'normal'},
        {'name': 'abl_zero', 'tau_mode': 'zero'},
        {'name': 'abl_shuffle', 'tau_mode': 'shuffle'},
        {'name': 'abl_const', 'tau_mode': 'const'},
    ]
    return run_variants('HUST', ['T0'], SEEDS5, variants)


def stage_abl_const_verify():
    """abl_const 的 10 种子验证（HUST T0）。

    5 种子时 abl_const 为 +0.84%（p≈0.20），样本量不足以定论。
    本阶段用 **10 个全新种子 + 同批次对照**（新变体名 abl10_*，不复用旧结果，
    避免跨批次漂移污染配对）。

      abl10_base  = tau_mode 'normal'（对照）
      abl10_const = tau_mode 'const'（τ₀ 在 batch 内取均值，抹掉样本级波动）
    """
    variants = [
        {'name': 'abl10_base', 'tau_mode': 'normal'},
        {'name': 'abl10_const', 'tau_mode': 'const'},
    ]
    return run_variants('HUST', ['T0'], list(range(10)), variants)


def stage_abl_const_phm():
    """abl_const 的 10 种子验证（PHM2009 T0）—— 检验跨数据集普适性。"""
    variants = [
        {'name': 'abl10_base', 'tau_mode': 'normal'},
        {'name': 'abl10_const', 'tau_mode': 'const'},
    ]
    return run_variants('PHM2009', ['T0'], list(range(10)), variants)


def stage_widan_init_hust():
    """验证 WIDAN 优势来源：宽核小波初始化 vs 随机初始化。

    诊断依据（diag_frontend_compare.py）：
      · WIDAN 前端参数量是 AG-Mamba 前端的 **63.7 倍**（3.85M vs 60K）
      · WIDAN 的 Laplace_fast 核过零率 **33.9**、峰值频率 **0.268**（接近 Nyquist）
        -> 频率参数语义错位，该初始化已混叠失效，不是有效低频带通先验
    若关掉小波初始化后性能不变 -> 优势来自容量，而非物理先验。

    任务选 T0（低方差 ±0.54）与 T9（高区分度，WIDAN 唯一低于 98% 的任务）。
    """
    import importlib
    mod = importlib.import_module('WIDAN')
    variants = [('wid_wavelet', True), ('wid_rand', False)]
    tasks = ['T0', 'T9']
    seeds = list(range(10))
    res = {tag: {} for tag, _ in variants}
    t_start = time.time()
    for task in tasks:
        for seed in seeds:
            for tag, wi in variants:
                mod.configs.wavelet_init = wi
                path = os.path.join(OUTDIR, 'HUST_{}_WIDAN_{}_seed{}.log'.format(
                    task, tag, seed))
                acc = _acc_of(path)
                if acc is not None:
                    print('[skip] {} -> {:.2f}%'.format(
                        os.path.basename(path), acc), flush=True)
                else:
                    t0 = time.time()
                    try:
                        a = run_one('WIDAN', 'HUST', task, seed, 'cuda', path)
                        acc = a * 100.0
                    except Exception as e:
                        print('[FAIL] {} : {}'.format(
                            os.path.basename(path), e), flush=True)
                        acc = float('nan')
                    print('[{}] {} -> {:.2f}% ({:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), os.path.basename(path), acc,
                        time.time() - t0), flush=True)
                res[tag][(task, seed)] = acc
        print('  --- HUST / {} / WIDAN ---'.format(task), flush=True)
        for tag, _ in variants:
            a = np.array([res[tag][(task, s)] for s in seeds], dtype=float)
            print('      {:12s}: {:.2f} +- {:.2f}  {}'.format(
                tag, np.nanmean(a), np.nanstd(a),
                [round(x, 2) for x in a]), flush=True)

    print('=' * 78, flush=True)
    print('SUMMARY WIDAN init 消融 (HUST T0+T9 x 10 seeds)', flush=True)
    for tag, _ in variants:
        allv = np.array([res[tag][k] for k in res[tag]], dtype=float)
        per_task = [np.nanmean([res[tag][(t, s)] for s in seeds]) for t in tasks]
        print('{:12s}: overall {:.2f} +- {:.2f} | per-task {}'.format(
            tag, np.nanmean(allv), np.nanstd(allv),
            [round(float(x), 2) for x in per_task]), flush=True)
    print('elapsed {:.0f}s'.format(time.time() - t_start), flush=True)
    print('=' * 78, flush=True)
    return res


def stage_widan_capacity_hust():
    """WIDAN 容量消融：固定小波初始化，缩放 ResNet18 通道数。

    已有证据（widan_init_hust）：
      · 关掉小波初始化 -> T0 −5.24、T9 −20.02（p<1e-5）
      · 但容量（3.85M vs AG-Mamba 的 60K）贡献未测

    本阶段固定 wavelet_init=True，按 widen 缩放通道：
      cap_full    widen=1.00  (~3.85M)
      cap_half    widen=0.50  (~0.96M)
      cap_quarter widen=0.25  (~0.24M)

    判读：缩容后性能仍高 -> 初始化是主因；若崩塌 -> 容量与初始化协同。
    """
    import importlib
    mod = importlib.import_module('WIDAN')
    variants = [('cap_full', 1.0), ('cap_half', 0.5), ('cap_quarter', 0.25)]
    tasks = ['T0', 'T9']
    seeds = list(range(10))
    res = {tag: {} for tag, _ in variants}
    t_start = time.time()
    for task in tasks:
        for seed in seeds:
            for tag, wd in variants:
                mod.configs.wavelet_init = True
                mod.configs.widen = wd
                path = os.path.join(
                    OUTDIR, 'HUST_{}_WIDAN_{}_seed{}.log'.format(task, tag, seed))
                acc = _acc_of(path)
                if acc is not None:
                    print('[skip] {} -> {:.2f}%'.format(
                        os.path.basename(path), acc), flush=True)
                else:
                    t0 = time.time()
                    try:
                        a = run_one('WIDAN', 'HUST', task, seed, 'cuda', path)
                        acc = a * 100.0
                    except Exception as e:
                        print('[FAIL] {} : {}'.format(
                            os.path.basename(path), e), flush=True)
                        acc = float('nan')
                    print('[{}] {} -> {:.2f}% ({:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), os.path.basename(path), acc,
                        time.time() - t0), flush=True)
                res[tag][(task, seed)] = acc
        print('  --- HUST / {} / WIDAN capacity ---'.format(task), flush=True)
        for tag, _ in variants:
            a = np.array([res[tag][(task, s)] for s in seeds], dtype=float)
            print('      {:12s}: {:.2f} +- {:.2f}  {}'.format(
                tag, np.nanmean(a), np.nanstd(a),
                [round(x, 2) for x in a]), flush=True)

    print('=' * 78, flush=True)
    print('SUMMARY WIDAN capacity 消融 (HUST T0+T9 x 10 seeds)', flush=True)
    for tag, wd in variants:
        allv = np.array([res[tag][k] for k in res[tag]], dtype=float)
        per_task = [np.nanmean([res[tag][(t, s)] for s in seeds]) for t in tasks]
        print('{:12s} widen={:.2f}: overall {:.2f} +- {:.2f} | per-task {}'.format(
            tag, wd, np.nanmean(allv), np.nanstd(allv),
            [round(float(x), 2) for x in per_task]), flush=True)
    print('elapsed {:.0f}s'.format(time.time() - t_start), flush=True)
    print('=' * 78, flush=True)
    return res


def stage_gaborwide_hust():
    """宽频段 Gabor 滤波器组 + 加宽 CNN 前端（不用小波、不用 ResNet18）。

    设计依据（来自 widan_init_hust / widan_capacity_hust 的消融结论）：
      · 初始化：对数均匀覆盖 f∈[0.005,0.45]，而原小波只有 [0.01,0.08] 窄带；
        50% 低频通道保留直流（WIDAN 核 max|mean|=0.50，非零均值）
      · 容量：层宽 64/128/256/512，约 965K 参数（vs 原 60K，16x）
      · 骨干：普通 1D CNN（非 ResNet）

    对照：gw_wide_gabor（本方案）vs gw_wide_rand（随机初始化同容量，
    用于分离初始化贡献）；基准用已有 HUST_T0_AGMamba_fc_base_seed*（60K）。
    """
    variants = [
        {'name': 'gw_wide_gabor', 'front_mode': 'gaborwide',
         'front_chan': 512, 'front_init': 'gabor'},
        {'name': 'gw_wide_rand', 'front_mode': 'gaborwide',
         'front_chan': 512, 'front_init': 'random'},
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(5)), variants)


def stage_gabor_alias_hust():
    """受控相位混叠 Gabor：复现 WIDAN 核「空间有结构、频谱弱偏好」的特性。

    动机：同一 965K 容量下，WIDAN 的 Laplace 核在内插与外推上**同时最优**
    （96.29 / 80.36），而解析 Gabor 与随机初始化各只能赢一项：
      rand  91.51 / 78.84   （外推好）
      gabor 95.64 / 66.04   （内插好）
    诊断发现 WIDAN 核过零率 33.9（远超解析 Gabor 的 6.0），频谱集中度 0.023。
    本阶段用相位调制 phi += alias*sin(2*pi*fm*t) 把频谱"抹开"：
      alias=1 -> 过零率 15.3, 集中度 0.0122
      alias=2 -> 过零率 25.5, 集中度 0.0089（接近 WIDAN 的 0.0226/33.9）
    """
    variants = [
        {'name': 'gw_alias1', 'front_mode': 'gaborwide', 'front_chan': 512,
         'front_init': 'gabor', 'gabor_dc_frac': 0.3, 'gabor_alias': 1.0},
        {'name': 'gw_alias2', 'front_mode': 'gaborwide', 'front_chan': 512,
         'front_init': 'gabor', 'gabor_dc_frac': 0.3, 'gabor_alias': 2.0},
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(5)), variants)


def stage_alias_sweep():
    """α 细扫（Fig. 11）：补齐 α∈{0, 0.5, 1.5, 3.0}（T0/T9 × 5 种子）。

    统一 dc_frac=0.3（与 gw_alias1/2 同配置）；gw_alias1(α=1)、gw_alias2(α=2)
    已缓存，run_variants 会自动跳过，合起来给出 6 点 α 曲线。
    """
    base = {'front_mode': 'gaborwide', 'front_chan': 512,
            'front_init': 'gabor', 'gabor_dc_frac': 0.3,
            'steps_override': 480}
    variants = [dict(base, name=n, gabor_alias=a) for n, a in [
        ('gw_sw_a00', 0.0), ('gw_sw_a05', 0.5),
        ('gw_sw_a15', 1.5), ('gw_sw_a30', 3.0)]
    ] + [
        dict(base, name='gw_alias1', gabor_alias=1.0),
        dict(base, name='gw_alias2', gabor_alias=2.0),
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(5)), variants)


# ---------------------------------------------------------------------------
# v0.3：α 细扫（**交付配置**版）
# ---------------------------------------------------------------------------
# 顺序刻意把 α=2.0 放最前：它与主对比 `pub` 臂应当**逐点一致**，
# 跑完第一个单元就能做同源性自检（不一致就及早停，不浪费后面 55 个单元）。
ALPHA_ARMS = [('as_a20', 2.0), ('as_a00', 0.0), ('as_a05', 0.5),
              ('as_a10', 1.0), ('as_a15', 1.5), ('as_a30', 3.0)]


def _pub_logpath(task, seed):
    """主对比 `pub` 臂日志的**绝对路径**。

    必须走绝对路径：α 阶段用 `AGM_RESULTS=<隔离目录>` 运行，
    `_P.log_path(..., arm='pub')` 会被重定向到隔离目录、因而找不到 pub 日志。
    """
    return os.path.join(_ROOT, 'results', 'HUST', 'AGMamba', 'pub', task,
                        'HUST_%s_AGMamba_fc_pub_seed%d.log' % (task, seed))


def _delivered_configs():
    """从 `agmamba/configs.yaml` 重载**交付配置**（= 主对比 `pub` 臂同源）。

    交付配置 = 无对齐头（w_mmd = w_adv = 0）、无液体项（tau_mode='zero'）、
    受控混叠 Gabor 前端（gaborwide/512/gabor/dc0.3/alias2.0/f_lo 0.0025/ksize128）、
    余弦退火（use_scheduler=True）。
    """
    import yaml as _yaml
    import torch as _torch
    from utils.DictObj import DictObj
    with open(os.path.join(_ROOT, 'agmamba', 'configs.yaml'),
              encoding='utf-8') as f:
        c = DictObj(_yaml.load(f, Loader=_yaml.FullLoader))
    if c.use_cuda and _torch.cuda.is_available():
        c.device = 'cuda'
    return c


def _points(path):
    """读一个日志的全部评估点（%）。与 `run_compare.parse_accs` 同一解析规则。"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'(?:tgt_acc|tgt_train_acc|tar-Acc):?\s*([0-9.]+)', line)
            if m:
                v = float(m.group(1))
                out.append(v * 100.0 if v <= 1.5 else v)
    return out


def stage_alias_sweep_clean():
    """α 细扫（v0.3 协议）：**交付配置**下的 6 点 α 曲线（HUST T0/T9 × 5 种子）。

    与旧 `alias_sweep`（v0.2）的两处关键区别
    ----------------------------------------
    1. **不套 `DEFAULTS`**。旧版 DEFAULTS 是 v0.2 探针配置：`w_mmd=0.5 /
       w_adv=0.1 / tau_mode='normal' / use_scheduler=False / gabor_f_lo=0.005 /
       front_ksize=64` —— 与 v0.3 交付配置不一致（交付 = 无对齐头 + 无液体项 +
       余弦退火 + f_lo 0.0025 + ksize 128）。本阶段每个变体都从
       `configs.yaml` **重载交付配置，只改 `gabor_alias`**。
    2. **读数用 `last5`**（§5.1 主口径），不是旧版的 `max()`（peak）。

    规模：6 点 × 2 任务（T0/T9）× 5 种子 = 60 单元；HUST 预算 480 步
    （= 主对比同一预算，`run_cell.EPOCHS × iters_per_epoch`）。

    **自检**：`as_a20`（α=2.0）应与主对比 `pub` 臂**逐点一致** ——
    同为交付配置 + 480 步 + 余弦退火 + 同种子 + 确定性初始化。

    ⚠ **务必用独立结果目录 + 强制纯 PyTorch 扫描**，否则这些臂会被
    `run_compare.discover_units` 扫成"AGMamba 的新臂"污染 P-统一汇总，且与主对比
    不可比：
        AGM_FORCE_PY_SCAN=1 AGM_RESULTS=$PWD/results_probe_alpha \\
            python -u run_stage.py alias_sweep_clean
    （`AGM_FORCE_PY_SCAN=1` 与主对比批次一致：官方 CUDA 内核反传含原子累加，
      不设该变量时实测 60 点里 53 点与 pub 臂不同、mid-training 差 9.78 分。）
    """
    import agmamba.method as _M
    tasks, seeds = ['T0', 'T9'], list(range(5))
    res = {}
    t_start = time.time()
    for name, alpha in ALPHA_ARMS:
        res[name] = {}
        for t in tasks:
            for s in seeds:
                path = lm_logpath('HUST', t, name, s)
                pts = _points(path)
                if len(pts) >= 60:
                    print('[skip] {} ({} 点)'.format(os.path.basename(path),
                                                    len(pts)), flush=True)
                else:
                    _M.configs = _delivered_configs()
                    _M.configs.gabor_alias = float(alpha)
                    t0 = time.time()
                    try:
                        run_one('AGMamba', 'HUST', t, s, 'cuda', path)
                        pts = _points(path)
                    except Exception as e:
                        print('[FAIL] {} : {}'.format(
                            os.path.basename(path), str(e)[:120]), flush=True)
                        pts = []
                    l5 = (float(np.mean(pts[-5:])) if len(pts) >= 5
                          else float('nan'))
                    print('[{}] {} a={:.1f} last5={:.2f} n={} ({:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), os.path.basename(path), alpha,
                        l5, len(pts), time.time() - t0), flush=True)
                    # α=2.0 的即时同源性自检（vs 主对比 pub 臂）
                    if name == 'as_a20' and len(pts) >= 60:
                        pb = _points(_pub_logpath(t, s))
                        if len(pb) >= 60:
                            dd = (max(abs(x - y) for x, y in zip(pts, pb))
                                  if len(pts) == len(pb) else float('nan'))
                            print('    [自检] a=2.0 vs pub({}/s{}) 逐点 max|Δ|={:.6f}'
                                  '  -> {}'.format(t, s, dd,
                                                   '一致' if dd < 1e-6 else '不一致✗'),
                                  flush=True)
                res[name][(t, s)] = pts
        print('  --- {} (alpha={}) done, elapsed {:.0f}s ---'.format(
            name, alpha, time.time() - t_start), flush=True)

    # ---- 汇总（last5 主口径）----
    print('=' * 78, flush=True)
    print('SUMMARY alpha sweep (clean/delivered config), last5 readout', flush=True)
    for name, alpha in sorted(ALPHA_ARMS, key=lambda kv: kv[1]):
        vals = [float(np.mean(v[-5:])) for v in res[name].values()
                if len(v) >= 5]
        per = {t: [float(np.mean(res[name][(t, s)][-5:])) for s in seeds
                   if len(res[name][(t, s)]) >= 5] for t in tasks}
        print('  a={:<4} {:14s} pooled={:.2f} (n={})  T0={:.2f}  T9={:.2f}'.format(
            alpha, name, float(np.mean(vals)) if vals else float('nan'),
            len(vals),
            float(np.mean(per['T0'])) if per['T0'] else float('nan'),
            float(np.mean(per['T9'])) if per['T9'] else float('nan')),
            flush=True)

    # ---- 自检：α=2.0 vs 主对比 pub 臂 ----
    dmax, nchk = -1.0, 0
    for key, pts in res.get('as_a20', {}).items():
        pb = _points(_pub_logpath(key[0], key[1]))
        if len(pts) >= 60 and len(pb) >= 60:
            nchk += 1
            dmax = max(dmax, abs(float(np.mean(pts[-5:]))
                                 - float(np.mean(pb[-5:]))))
    if nchk:
        print('  [自检] as_a20 vs pub 臂：n={}  max|Δlast5|={:.4f}  -> {}'
              .format(nchk, dmax,
                      '一致 ✓' if dmax < 1e-6 else '不一致 ✗（需查配置差异）'),
              flush=True)
    else:
        print('  [自检] 数据不足，未比对', flush=True)
    print('=' * 78, flush=True)
    return res


def stage_gabor_soft_hust():
    """A+B 组合：软化物理先验（降低直流比例 + 混合初始化）。

    依据 gw_wide_gabor / gw_wide_rand 的对比结论：
      · 扩容有效（平均 +5.53）
      · 但**强先验在外推上有害**：T9 gabor 66.04 vs rand 78.84（p=0.011，显著）
        同时 T0 gabor 95.64 vs rand 91.51（p=0.051，边缘显著）
      -> 先验强度应与分布偏移性质匹配：内插宜强、外推宜弱

    A 降先验强度 : gabor_dc_frac 0.5 -> 0.2
    B 混合初始化 : 50% 通道 Gabor + 50% 随机
    """
    variants = [
        {'name': 'gw_dc20', 'front_mode': 'gaborwide', 'front_chan': 512,
         'front_init': 'gabor', 'gabor_dc_frac': 0.2},
        {'name': 'gw_mix', 'front_mode': 'gaborwide', 'front_chan': 512,
         'front_init': 'mix', 'gabor_dc_frac': 0.2},
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(5)), variants)


def stage_ssm_ablate_hust():
    """SSM 分支消融（gw_alias2 前端固定，同批次配对）。

    背景：同容量（965K）横向对比显示 WIDAN 用「前端 + 全局池化（无 SSM）」
    达到 96.29/80.36，而我们的「前端 + 液体 SSM」是 95.16/79.56。
    
    若去掉 SSM 后性能不降反升 -> 液体 SSM 是负担，方法可简化为「前端 + 对齐」
    若明显下降 -> SSM 有真实贡献，值得保留并优化

    两个变体除了 use_ssm 外全部相同（含 gabor_alias=2.0, dc_frac=0.3）。
    """
    common = dict(front_mode='gaborwide', front_chan=512, front_init='gabor',
                  gabor_dc_frac=0.3, gabor_alias=2.0)
    variants = [
        dict(name='gw_a2_ssm', use_ssm=True, **common),
        dict(name='gw_a2_nossm', use_ssm=False, **common),
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(5)), variants)


def stage_ssm_liquid_ablate():
    """P0+P1 联合消融：同批次、同前端、三变体，分离 SSM 与液体机制的贡献。

    P0（同批次复核）: 上一轮 gw_a2_ssm(00:00) vs gw_a2_nossm(06:44) 相隔 6 小时，
                      属于跨批次，需要同批次复核 SSM 的 +2.36 是否稳定。
    P1（去液体化）  : τ₀≡0 等价于移除 γ·τ₀ 项 -> Δ 退化为标准的输入依赖选择性 Δ。
                      用于回答「SSM 的贡献是否来自液体机制」。

    变体（前端全部固定为 gw_alias2: gaborwide + chan512 + gabor + dc0.3 + alias2.0）:
      v3_full     : SSM ✅ + 液体 ✅   （完整版）
      v3_nossm    : SSM ❌ + 液体 ——   （P0）
      v3_noliquid : SSM ✅ + τ₀≡0     （P1，等价于标准 Mamba Δ）
    """
    common = dict(front_mode='gaborwide', front_chan=512, front_init='gabor',
                  gabor_dc_frac=0.3, gabor_alias=2.0)
    variants = [
        dict(name='v3_full', use_ssm=True, tau_mode='normal', **common),
        dict(name='v3_nossm', use_ssm=False, tau_mode='normal', **common),
        dict(name='v3_noliquid', use_ssm=True, tau_mode='zero', **common),
    ]
    return run_variants('HUST', ['T0', 'T9'], list(range(10)), variants)


def stage_align_ablate_hust():
    """对齐头消融：**交付前端固定**，分别关闭源间 MMD 与 GRL 对抗项。

    补的是局限 L5 的空档——交付配置用的 w_mmd=0.5 / w_adv=0.1 从未在交付
    前端上被消融过（6.4 节现有的对齐头证据来自早期窄带前端）。

    前端与其它开关完全等于交付配置（gaborwide + chan512 + gabor-init +
    f_lo=0.0025 + ksize128 + dc0.3 + alias2.0 + use_ssm + tau_mode='zero'），
    预算为 480 步探针协议（与表 6/14/15 同协议，可与那些行直接对照）。

      aln_full  : w_mmd 0.5, w_adv 0.1   （同批次对照）
      aln_nommd : w_mmd 0.0, w_adv 0.1   （只关源间 MMD）
      aln_noadv : w_mmd 0.5, w_adv 0.0   （只关对抗域判别）
      aln_noall : w_mmd 0.0, w_adv 0.0   （两者都关，退化为纯分类）

    四臂除这两个权重外逐位相同：w=0 仅把对应项从 loss 中移除，不改变前向
    路径、数据次序或随机数消耗，故同种子配对严格成立（§5.4 的配对协议）。
    """
    common = dict(front_mode='gaborwide', front_chan=512, front_init='gabor',
                  gabor_f_lo=0.0025, front_ksize=128,
                  gabor_dc_frac=0.3, gabor_alias=2.0,
                  use_ssm=True, tau_mode='zero')
    variants = [
        dict(name='aln_full', w_mmd=0.5, w_adv=0.1, **common),
        dict(name='aln_nommd', w_mmd=0.0, w_adv=0.1, **common),
        dict(name='aln_noadv', w_mmd=0.5, w_adv=0.0, **common),
        dict(name='aln_noall', w_mmd=0.0, w_adv=0.0, **common),
    ]
    return run_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)), variants)


def _run_clean_variants(data, tasks, seeds, variants, check_arm=None,
                        check_ref=None):
    """在**交付配置**上逐变体覆盖后运行（v0.3 probe 的统一路径）。

    与 `run_variants` 的唯一区别：**不套 v0.2 的 `DEFAULTS`**，而是每个变体都从
    `agmamba/configs.yaml` 重载交付配置，只应用该变体显式列出的字段。
    旧路径里 `DEFAULTS` 的 `use_scheduler=False` 会悄悄关掉余弦退火（变常数 lr）、
    `gabor_f_lo=0.005` / `front_ksize=64` 也与交付配置不符 —— 都会让消融与主对比
    （`pub` 臂）不可比。

    check_arm / check_ref：指定一个变体与一个参照日志（callable(task, seed)->path）
    做**逐点同源性自检**，用于确认该臂 = 交付配置。
    """
    import agmamba.method as _M
    res = {}
    for v in variants:
        vv = dict(v)
        name = vv.pop('name')
        res[name] = {}
        t_all = time.time()
        for t in tasks:
            for s in seeds:
                path = lm_logpath(data, t, name, s)
                pts = _points(path)
                if len(pts) >= 60:
                    print('[skip] {} ({} 点)'.format(os.path.basename(path),
                                                    len(pts)), flush=True)
                else:
                    _M.configs = _delivered_configs()
                    for k, val in vv.items():
                        setattr(_M.configs, k, val)
                    t0 = time.time()
                    try:
                        run_one('AGMamba', data, t, s, 'cuda', path)
                        pts = _points(path)
                    except Exception as e:
                        print('[FAIL] {} : {}'.format(
                            os.path.basename(path), str(e)[:120]), flush=True)
                        pts = []
                    l5 = (float(np.mean(pts[-5:])) if len(pts) >= 5
                          else float('nan'))
                    print('[{}] {} last5={:.2f} n={} ({:.0f}s)'.format(
                        time.strftime('%H:%M:%S'), os.path.basename(path),
                        l5, len(pts), time.time() - t0), flush=True)
                    if (check_arm and name == check_arm and len(pts) >= 60
                            and check_ref is not None):
                        pb = _points(check_ref(t, s))
                        if len(pb) >= 60:
                            dd = (max(abs(x - y) for x, y in zip(pts, pb))
                                  if len(pts) == len(pb) else float('nan'))
                            print('    [自检] {} vs 参照({}/s{}) 逐点 max|Δ|={:.6f}'
                                  ' -> {}'.format(name, t, s, dd,
                                                  '一致' if dd < 1e-6
                                                  else '不一致✗'), flush=True)
                res[name][(t, s)] = pts
        print('  --- {} done, elapsed {:.0f}s ---'.format(
            name, time.time() - t_all), flush=True)

    print('=' * 78, flush=True)
    print('SUMMARY {} (clean/delivered config), last5 readout'.format(data),
          flush=True)
    for v in variants:
        name = v['name']
        vals = [float(np.mean(x[-5:])) for x in res[name].values()
                if len(x) >= 5]
        cells = []
        for t in tasks:
            tv = [float(np.mean(res[name][(t, s)][-5:])) for s in seeds
                  if len(res[name][(t, s)]) >= 5]
            cells.append('%s=%s' % (t, ('%.2f' % np.mean(tv)) if tv else 'na'))
        print('  {:10s} pooled={:.2f} (n={})  {}'.format(
            name, float(np.mean(vals)) if vals else float('nan'), len(vals),
            '  '.join(cells)), flush=True)
    print('=' * 78, flush=True)
    return res


def stage_align_ablate_clean():
    """对齐头消融（v0.3 交付配置）：w_mmd / w_adv 四组合，HUST T0/T4/T9 × 5 种子。

    与 `stage_align_ablate_hust` 的唯一区别：**不套 v0.2 的 `DEFAULTS`**
    （旧路径会把余弦退火关掉、f_lo/ksize 也退化成旧值）。本阶段每臂都从
    `configs.yaml` 重载交付配置，**只改 `w_mmd` / `w_adv`**，其余逐位相同 ——
    w=0 仅把对应项从 loss 移除，不改前向路径、数据次序与随机数消耗，
    故同种子配对严格成立（§5.4）。

      aln_noall : w_mmd 0.0, w_adv 0.0   ← **= 交付配置**
      aln_full  : w_mmd 0.5, w_adv 0.1   （v0.2 的带对齐配置）
      aln_nommd : w_mmd 0.0, w_adv 0.1
      aln_noadv : w_mmd 0.5, w_adv 0.0

    **自检**：`aln_noall` 应与主对比 `pub` 臂**逐点一致**（T0/T4/T9 × 5 种子 = 15 单元）。

    运行方式（必须与主对比同源，且结果隔离）：
        AGM_FORCE_PY_SCAN=1 AGM_RESULTS=$PWD/results_probe_align \\
            python -u run_stage.py align_ablate_clean
    """
    variants = [
        dict(name='aln_noall', w_mmd=0.0, w_adv=0.0),
        dict(name='aln_full', w_mmd=0.5, w_adv=0.1),
        dict(name='aln_nommd', w_mmd=0.0, w_adv=0.1),
        dict(name='aln_noadv', w_mmd=0.5, w_adv=0.0),
    ]
    return _run_clean_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)),
                               variants, check_arm='aln_noall',
                               check_ref=_pub_logpath)


def stage_tau_ssm_ablate_clean():
    """τ（液体时间常数）与 SSM 分支的 clean 消融（v0.3 交付配置）。

    交付配置为 `tau_mode='zero'`（τ₀ ≡ 0，即**标准选择性离散化**）。本阶段在交付
    配置上只改 `tau_mode` / `use_ssm`，其余字段逐位相同：

      tau_zero    : tau_mode='zero'    ← **= 交付配置**（自检基准）
      tau_normal  : tau_mode='normal'  （估计 τ₀ 并按 Δ = softplus(WΔu + bΔ + γτ₀) 注入）
      tau_shuffle : τ₀ 在 batch 内随机置换（保留量级，破坏样本对应）
      ssm_off     : use_ssm=False（退化为纯前端 + 双路池化 + 头）

    判读：
      · `tau_zero` 应与主对比 `pub` 臂**逐点一致**（同源性自检，验证 clean 路径）。
      · `tau_normal` / `tau_shuffle` 不优于 `tau_zero` → 液体先验无收益；
        若两者彼此无差异 → τ₀ 的**取值**不承载信息（物理时间尺度是包装）。
      · `ssm_off` 明显低于 `tau_zero` → 选择性状态空间记忆不可或缺。

    ⚠ **配对口径（与对齐消融不同，必须如实说明）**：
      对齐四臂 w=0 只移除 loss 项，前向路径与随机数消耗不变，故同 (task, seed)
      是**严格配对**（§5.4）。本阶段改的是**架构**（新增/删除 τ 估计器与 SSM 模块），
      建模期 RNG 消耗会变，因此只能按 **(task, seed) 共享数据划分**配对，
      不可主张"逐位可复现"。

    运行方式（必须与主对比同源，且结果隔离）：
        AGM_FORCE_PY_SCAN=1 AGM_RESULTS=$PWD/results_probe_tau \
            python -u run_stage.py tau_ssm_ablate_clean
    """
    variants = [
        dict(name='tau_zero', tau_mode='zero'),
        dict(name='tau_normal', tau_mode='normal'),
        dict(name='tau_shuffle', tau_mode='shuffle'),
        dict(name='ssm_off', use_ssm=False),
    ]
    return _run_clean_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)),
                               variants, check_arm='tau_zero',
                               check_ref=_pub_logpath)


# ---------------------------------------------------------------------------
# v0.3：前端设计消融（初始化 / 低频扩展 / 容量）—— **交付配置**版
# ---------------------------------------------------------------------------
# 取代已作废的 v0.2 探针数据（`_s63_data.md`，来源 `results_dgfdb/` 已删除）：
# 旧口径为 best-epoch + 2880 步 + 带对齐头(w_mmd 0.5/w_adv 0.1) + tau_mode=normal
# + dc_frac 0.5 + ksize 64/f_lo 0.005 + 关余弦退火，与交付配置不可比，
# 且 W3 已明文禁止其回流正文。
FRONTEND_ARMS = [
    ('fr_rand',   dict(front_init='random'),
     '首层随机初始化（同形状 Kaiming），其余逐位同交付配置'),
    ('fr_mix',    dict(front_init='mix'),
     '首层一半 Gabor + 一半随机'),
    ('fr_lap',    dict(front_init='laplace'),
     '首层改用 WIDAN 参考实现的 Laplace 族（64 原子 × 核长 128）：'
     '§3.3 / Fig. 2 平面上「宽核 Laplace 族」那一区的对照'),
    ('fr_narrow', dict(gabor_f_lo=0.005, front_ksize=64),
     '低频扩展反向消融：回到 f_lo 0.005 + 首层核 64（v0.2 探针档）'),
    ('fr_cap768', dict(front_chan=768),
     '前端宽度 512 -> 768（层宽 96/192/384/768）'),
    ('fr_cap960', dict(front_chan=960),
     '前端宽度 512 -> 960（层宽 120/240/480/960，总量≈WIDAN 3.85M 量级）'),
]


def stage_frontend_ablate_clean():
    """§6.3 前端设计三组消融（v0.3 交付配置）：初始化、低频扩展、容量。

    三项各回答一个设计问题（α 扫描已由 `alias_sweep_clean` 完成）：

      初始化对照 : 解析 Gabor 首层先验是否必要？随机 / 半混合 vs 交付的 Gabor
      低频扩展   : 交付档（f_lo 0.0025 + 首层核 128，把覆盖下延至最慢转速轴频）
                   相对窄带短核档（0.005 / 64）是否带来精度收益
      容量       : 前端宽度是不是杠杆？512 -> 768 / 960

    每个变体都从 `agmamba/configs.yaml` **重载交付配置**，只覆盖该臂显式列出的
    字段（`_run_clean_variants`，不套 v0.2 的 `DEFAULTS`）；主口径 `last5`，
    规模 = 5 臂 × T0/T4/T9 × 5 种子 = **75 单元**，HUST 480 步（= 主对比预算），
    实测 ~200-280 s/单元 -> 全程约 4.5-5.5 h。

    **配对口径（必须如实说明）**：本阶段改的是**前端结构与初始化**——
    `front_chan` 改变参数形状、`front_ksize` 改变核形状，而 Gabor 首层是
    `copy_` 写入、随机/混合首层要走 Kaiming（消耗 RNG），因此**下游各层拿到的
    随机权重也随之不同**。故这些臂**不是严格配对**，只共享 (task, seed) 数据
    划分；结论按 §5.3 的可复现下限 2.55 点判读，与 τ/SSM 臂同一口径。

    **参照臂不必重跑**：交付配置的 480 步参照就是 `results_probe_align` 的
    `aln_noall`（= 主对比 `pub`，已核 15 单元逐点 max|Δ| = 0.000000）。
    摘要脚本 `analysis/summary_front_ablate.py` 会再次核对该同源性。

    运行方式（必须与主对比同源，且结果隔离）：
        AGM_FORCE_PY_SCAN=1 AGM_RESULTS=$PWD/results_probe_front \
            python -u run_stage.py frontend_ablate_clean
    """
    variants = [dict(name=nm, **over) for nm, over, _ in FRONTEND_ARMS]
    return _run_clean_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)),
                               variants)


def stage_final10_hust():
    """P2：最终配置在 HUST 全 10 个 LODO 任务上 × 5 种子，与 WIDAN 正面对比。

    配置依据（来自 v3 消融，HUST T0 × 10 种子 同批次配对）:
      · 宽频段受控混叠 Gabor 前端（965K）: 相对原 60K 窄带前端 +9.4 平均
      · 标准 Mamba 选择性 Δ（use_ssm=True）: +2.40 (p=0.0002, 9/10 正向) ✅
      · 液体机制（γ·τ₀）: -0.38 (p=0.38, 3+/10-) ❌ 无贡献
        -> 因此 tau_mode='zero'（τ₀≡0，退化为标准选择性 Δ）
        且方差更小（±0.60 vs ±1.19）

    WIDAN 基准（同协议，10 任务 × 5 种子）: 97.80 +- 4.09
    """
    final_cfg = dict(front_mode='gaborwide', front_chan=512, front_init='gabor',
                     gabor_dc_frac=0.3, gabor_alias=2.0,
                     use_ssm=True, tau_mode='zero')
    variants = [dict(name='v4_noL', **final_cfg)]
    return run_variants('HUST', HUST_TASKS, SEEDS5, variants)


def stage_surpass_a():
    """批次 A：四候选筛选（T0/T4/T9 × 5 种子），目标超越 WIDAN。

    基线 v4_noL（final10_hust）三任务对照 WIDAN: T0 -2.71, T4 -2.14, T9 待评估。
    候选统一: gaborwide + gabor-init + dc0.3 + alias2 + use_ssm + tau_mode=zero
      sa_cap768     : 前端 512 -> 768（WIDAN 容量曲线参考增益 +1.9/+5.0）
      sa_cap1024    : 前端 512 -> 1024（~3.9M，总量对齐 WIDAN 3.85M）
      sa_lowf512    : f_lo 0.005->0.0025 + 首层核 64->128（T0 故障基频低于频带下限）
      sa_cap768_lowf: 768 + 低频扩展合并
    """
    common = dict(front_mode='gaborwide', front_init='gabor',
                  gabor_dc_frac=0.3, gabor_alias=2.0,
                  use_ssm=True, tau_mode='zero')
    variants = [
        dict(name='sa_cap768', front_chan=768, **common),
        dict(name='sa_cap1024', front_chan=1024, **common),
        dict(name='sa_lowf512', front_chan=512, gabor_f_lo=0.0025,
             front_ksize=128, **common),
        dict(name='sa_cap768_lowf', front_chan=768, gabor_f_lo=0.0025,
             front_ksize=128, **common),
    ]
    return run_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)), variants)


# 批次 B/C 的统一前端配置（低频扩展版），供 stage 与 autopilot 共用
SURPASS_COMMON = dict(front_mode='gaborwide', front_init='gabor',
                      front_chan=512, gabor_f_lo=0.0025, front_ksize=128,
                      gabor_dc_frac=0.3, gabor_alias=2.0,
                      use_ssm=True, tau_mode='zero')

# 批次 B 的四个变体（autopilot 决策时复用同一配置表）
SURPASS_B_VARIANTS = [
    ('sb_long3x', {'steps_override': 1440, 'use_scheduler': True}),
    ('sb_a25', {'gabor_alias': 2.5}),
    ('sb_ssmbig', {'n_blocks': 4, 'd_model': 192, 'd_inner': 192}),
    ('sb_long3x_ssmbig', {'steps_override': 1440, 'use_scheduler': True,
                          'n_blocks': 4, 'd_model': 192, 'd_inner': 192}),
]


# EMA 试点任务：best−final 落差最大的四个任务
EMA_PILOT_TASKS = ['T0', 'T5', 'T8', 'T9']


def stage_ema_pilot():
    """pilot B：交付配置 + EMA 影子权重（目标：抬高非预言口径）。

    背景：sc_steps6x 的 best 97.86 但 last10 仅 95.34（T0 −3.4、T9 −9.0），
    后期源域 CE→0（过拟合）使轨迹达峰后回落；WIDAN_long 同预算
    best 98.30 / last10 95.03。EMA 不改架构与超参，只改权重轨迹，
    且推理时即用影子权重（predict → _eval_net），故日志 tgt_acc 即 EMA 精度。
    对照＝已有的 sc_steps6x（同名种子、同配置）。
    """
    variants = [
        dict(name='em999', steps_override=2880, use_scheduler=True,
             ema_decay=0.999, ema_start=100, **SURPASS_COMMON),
    ]
    return run_variants('HUST', EMA_PILOT_TASKS, list(range(5)), variants)


def stage_cross_final():
    """跳数据集：交付配置在 PU / PHM2009 上的 4 任务 × 5 种子。

    两条臂：
      xg_std : 同架构/损失，但用各数据集默认的 60 epoch 预算
               （PU 2,640 步、PHM 1,500 步）——与基线同协议，公平比较。
      xg_6x  : 与 HUST 交付版一致（steps_override 2880 + 余弦），
               用于检查“更多预算是否也帮到跳数据集”。
    日志名：{PU,PHM2009}_T{k}_AGMamba_fc_{xg_std,xg_6x}_seed{s}.log
    """
    spec = _load_auto_spec('auto_final.json')
    v = spec['variants'][0]
    base = dict(v.get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    variants = [
        dict(name='xg_std', **std),
        dict(name='xg_6x', **base),
    ]
    res = {}
    for data in ('PU', 'PHM2009'):
        res[data] = run_variants(data, STD_TASKS, SEEDS5, variants)
    return res


# ---------------------------------------------------------------------------
# 归因实验：弥散 vs 容量/带宽（PU）+ 转速x负载混合协议（PHM2009 低负载）
# ---------------------------------------------------------------------------

# PU 上的 2x2：弥散 α ∈ {0, 2} × 前端宽度 ∈ {128, 512}，其余固定为交付配置。
# 预算与 Table 9 的 xg_std 相同（各数据集默认 60 epoch，PU = 2,640 步），
# 用于判定 §6.2 中 T2/T3 的 +8.32/+4.14 有多少来自弥散、多少来自容量/带宽。
PU_AC_VARIANTS = [
    ('puac_a0_c128', 0.0, 128),
    ('puac_a2_c128', 2.0, 128),
    ('puac_a0_c512', 0.0, 512),
    ('puac_a2_c512', 2.0, 512),
]


def stage_pu_alpha_capacity():
    """PU 归因 2x2（4 任务 × 5 种子 × 4 臂 = 80 次运行）。

    日志名：PU_T{k}_AGMamba_fc_puac_a{α}_c{宽度}_seed{s}.log
    """
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    base.pop('steps_override', None)      # 与 xg_std / base 同预算
    variants = []
    for nm, alias, ch in PU_AC_VARIANTS:
        v = dict(base)
        v.update(name=nm, gabor_alias=alias, front_chan=ch)
        variants.append(v)
    return run_variants('PU', STD_TASKS, SEEDS5, variants)


# PHM2009 工况索引（见 datasets/PHM_2009_Gearbox.total_conditions）：
#   0=30Hz_High_1  2=30Hz_Low_1  4=35Hz_High_1  6=35Hz_Low_1
#   8=40Hz_High_1 10=40Hz_Low_1 12=45Hz_High_1 14=45Hz_Low_1
# 主协议（T0–T3）只用 0/4/8/12 = 纯转速阶梯 @ High 负载。
# 本阶段加两条协议：
#   协议 B（L0–L3）：30/35/40/45 Hz 全部取 Low 负载 —— 纯转速阶梯，但负载水平不同；
#   协议 C（M0–M3）：目标是 Low 负载且其转速在源域（全 High 负载）中不出现 ——
#                    转速与负载**两个轴同时外推**，3 源域机器与主协议一致。
PHM_TASKS_EXT = {
    'L0': [[6, 10, 14], [2]],
    'L1': [[2, 10, 14], [6]],
    'L2': [[2, 6, 14], [10]],
    'L3': [[2, 6, 10], [14]],
    'M0': [[4, 8, 12], [2]],
    'M1': [[0, 8, 12], [6]],
    'M2': [[0, 4, 12], [10]],
    'M3': [[0, 4, 8], [14]],
}
PHM_EXT_TASKS = ['L0', 'L1', 'L2', 'L3', 'M0', 'M1', 'M2', 'M3']
PHM_EXT_BASELINES = ['WIDAN', 'UDTL', 'DANN']


def stage_phm_mixed_factor():
    """PHM2009 转速x负载协议（协议 B 低负载阶梯 + 协议 C 双因素外推）。

    臂：xg_std（交付配置、各数据集默认预算）、base（窄带前端内部对照），
    以及 WIDAN/UDTL/DANN 三个最有信息量的基线。
    共 (2 + 3) × 8 任务 × 5 种子 = 200 次运行。
    日志名：PHM2009_{Lk,Mk}_AGMamba_fc_{xg_std,base}_seed{s}.log 与
            PHM2009_{Lk,Mk}_{METHOD}_seed{s}.log
    """
    import run_dgfdb
    run_dgfdb.TASKS['PHM2009'].update(PHM_TASKS_EXT)

    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    variants = [dict(name='xg_std', **std), {'name': 'base'}]

    res = {}
    for task in PHM_EXT_TASKS:
        print('\n==== PHM2009 {} ===='.format(task), flush=True)
        res[task] = run_variants('PHM2009', [task], SEEDS5, variants)
        for method in PHM_EXT_BASELINES:
            res['{}_{}'.format(task, method)] = run_baseline(
                method, 'PHM2009', [task], SEEDS5)
    return res


def stage_uo_cross():
    """第四基准（UO 时变转速，4 个转速域 × 5 种子）：xg_std / base + 三个基线。

    UO 与其余三个基准的差别在于转速在样本内**连续变化**（编码器给出瞬时转速，
    域按平均转速分位切分），因此它检验的是"非平稳转速"这一种偏移，而不是
    转速x负载的混合偏移（后者由 phm_mixed_factor 的 L*/M* 协议覆盖）。
    日志名：UO_T{k}_AGMamba_fc_{xg_std,base}_seed{s}.log 与
            UO_T{k}_{METHOD}_seed{s}.log
    """
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    variants = [dict(name='xg_std', **std), {'name': 'base'}]
    res = {'ours': run_variants('UO', STD_TASKS, SEEDS5, variants)}
    for method in PHM_EXT_BASELINES:
        res[method] = run_baseline(method, 'UO', STD_TASKS, SEEDS5)
    return res


def stage_nondet_ctl():
    """可复现性对照：**同配置、同种子**重跑，量化运行间非确定性（误差底噪）。

    发现：PU 上 xg_std 与 puac_a2_c512（配置与种子完全相同）的日志前 3–4 步
    逐位一致，之后在第 4 位小数处分叉，最终 4 任务均值差 0.55、逐种子 |差| 均值
    1.64（最大 4.86）。本阶段用带后缀的副本臂再测若干单元，给出可发表的底噪。

    单元：PU T0/T2 x 2 次重跑；HUST T9（交付预算）× 2 次重跑。
    """
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    rep_std = [dict(name='rep_a', **std), dict(name='rep_b', **std)]
    res = {}
    res['PU'] = run_variants('PU', ['T0', 'T2'], SEEDS5, rep_std)
    rep_full = [dict(name='rep_a', **base), dict(name='rep_b', **base)]
    res['HUST'] = run_variants('HUST', ['T9'], list(range(5)), rep_full)
    return res


def stage_phm_capacity():
    """容量对齐实验：把本方法容量对齐到 WIDAN（3.98M）后，三个基准对称覆盖。

    容量选择（口径同表 18，含仅训练用的判别器）：
        front_chan=512 -> 1,406,357（交付配置，推理路径 1,372,940）
        front_chan=960 -> 3,883,685（推理路径 ≈3.85M，对 WIDAN 3,981,193 为 0.97×）
    除容量外配置完全不变（gaborwide + gabor + ks 128 + f_lo 0.0025 + dc 0.3 +
    alias 2.0 + SSM + tau zero）。

    四组：
      ① PHM2009 12 任务（High 阶梯 T0–T3 / Low 阶梯 L0–L3 / 双因素 M0–M3），
         各数据集默认 60 epoch 预算 -> 臂名 cap960；
      ② PU T0–T3，默认 60 epoch（2,640 步）-> cap960；
      ③ HUST T0/T4/T9，交付预算 2,880 步 + 余弦 -> 臂名 cap960x；
      ④ HUST T0/T4/T9，探针预算 480 步、无调度 -> 臂名 cap960p，
         补全表 8 容量曲线（sa_lowf512 512ch → sa_cap768_lowf 768ch → 本臂 960ch
         → sa_cap1024 1024ch，全部同一核组与协议）。
    共 60 + 20 + 15 + 15 = 110 次运行。
    日志名：{PU,PHM2009,HUST}_{TASK}_AGMamba_fc_{cap960,cap960x,cap960p}_seed{s}.log
    """
    import run_dgfdb
    run_dgfdb.TASKS['PHM2009'].update(PHM_TASKS_EXT)

    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)          # 各数据集默认预算
    cap_std = dict(std)
    cap_std['front_chan'] = 960
    cap_full = dict(base)                    # 保留 steps_override=2880 + 余弦
    cap_full['front_chan'] = 960
    probe = {k: v for k, v in std.items() if k != 'use_scheduler'}
    probe['front_chan'] = 960

    res = {}
    print('\n#### 组 ① PHM2009（12 任务，默认预算）', flush=True)
    res['PHM'] = run_variants(
        'PHM2009', STD_TASKS + PHM_EXT_TASKS, SEEDS5,
        [dict(name='cap960', **cap_std)])
    print('\n#### 组 ② PU（4 任务，默认预算）', flush=True)
    res['PU'] = run_variants(
        'PU', STD_TASKS, SEEDS5, [dict(name='cap960', **cap_std)])
    print('\n#### 组 ③ HUST（T0/T4/T9，交付预算 2,880 步 + 余弦）', flush=True)
    res['HUST_full'] = run_variants(
        'HUST', ['T0', 'T4', 'T9'], SEEDS5, [dict(name='cap960x', **cap_full)])
    print('\n#### 组 ④ HUST（T0/T4/T9，探针预算 480 步）', flush=True)
    res['HUST_probe'] = run_variants(
        'HUST', ['T0', 'T4', 'T9'], SEEDS5, [dict(name='cap960p', **probe)])
    return res


def stage_det_ctl():
    """非确定性根因诊断：同一配置训练两次，分别开/关 SSM 分支，比较日志是否逐位一致。

    已测：交付配置（含 SSM）两次运行的日志在第 3–4 步起在第 4 位小数处分叉，
    最终逐种子 |Δ| 均值约 2–3 点（PU）/2 点（HUST T9，交付预算）。
    本阶段用 PU T2 seed0 做最小诊断（**同一配置各跑两次**，逐行比对日志）：
      det_nossm_a / det_nossm_b : use_ssm=False（其余同交付配置）
      det_ssm_a   / det_ssm_b   : use_ssm=True （对照）
    若某对两次运行逐位一致、而另一对在第 4 位小数处分叉，即可锁定非确定性来源。
    日志名：PU_T2_AGMamba_fc_det_{nossm,ssm}_{a,b}_seed0.log
    """
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    no_ssm = dict(std)
    no_ssm['use_ssm'] = False
    with_ssm = dict(std)
    with_ssm['use_ssm'] = True
    res = {}
    for tag, cfg in (('nossm', no_ssm), ('ssm', with_ssm)):
        variants = [dict(name='det_{}_a'.format(tag), **cfg),
                    dict(name='det_{}_b'.format(tag), **cfg)]
        res[tag] = run_variants('PU', ['T2'], [0], variants)
    return res


def stage_det_ctl2():
    """确定性开关复测：与 det_ctl 相同的两次重跑，但要求 AGM_DET=1 启动。

    det_ctl 的结论是无 SSM 时两次运行仍差 3.57 点（有 SSM 时差 3.14 点），
    说明选择性扫描核不是（唯一）来源。本阶段在 `AGM_DET=1`（强制 cuDNN 确定性
    卷积算法）下重跑同一诊断：
      · 若两对都逐位一致 -> 非确定性来自 cuDNN 卷积算法选择，且已存在可复现模式；
      · 若仍分叉        -> 需继续排查（卷积以外的算子或损失）。
    日志名：PU_T2_AGMamba_fc_det2_{nossm,ssm}_{a,b}_seed0.log
    """
    if os.environ.get('AGM_DET', '0') != '1':
        print('[det_ctl2] 警告：未设置 AGM_DET=1，本次结果不能用于判定确定性',
              flush=True)
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    res = {}
    for tag, ssm in (('nossm', False), ('ssm', True)):
        cfg = dict(std)
        cfg['use_ssm'] = ssm
        variants = [dict(name='det2_{}_a'.format(tag), **cfg),
                    dict(name='det2_{}_b'.format(tag), **cfg)]
        res[tag] = run_variants('PU', ['T2'], [0], variants)
    return res


def stage_det_ctl3():
    """空闲 GPU 上的确定性复测：检验非确定性是否来自"共享 GPU"。

    已知事实：
      · det_ctl（默认设置，GPU 同时有另一任务）：有/无 SSM 两对分别差 3.14 / 3.57 点；
      · det_ctl2（AGM_DET=1，同样共享 GPU）：仍差 2.00 / 4.14 点 → 确定性开关无效。
    假设：cuDNN 在同一问题上会因可用显存/占用不同而选择不同（但各自确定的）算法，
    因此共享 GPU 时同配置两次运行的数值不一致、空闲时则一致。
    本阶段要求在**空闲 GPU** 上运行（队列脚本会等待其它训练结束）：
      · 若两对逐位一致 -> 非确定性是共享 GPU 的产物，可写成"独占运行时可复现"；
      · 若仍分叉     -> 属于代码库本身的数值非确定性，只能作为局限报告。
    日志名：PU_T2_AGMamba_fc_det3_{nossm,ssm}_{a,b}_seed0.log
    """
    spec = _load_auto_spec('auto_final.json')
    base = dict(spec['variants'][0].get('kwargs', {}))
    std = dict(base)
    std.pop('steps_override', None)
    res = {}
    for tag, ssm in (('nossm', False), ('ssm', True)):
        cfg = dict(std)
        cfg['use_ssm'] = ssm
        variants = [dict(name='det3_{}_a'.format(tag), **cfg),
                    dict(name='det3_{}_b'.format(tag), **cfg)]
        res[tag] = run_variants('PU', ['T2'], [0], variants)
    return res


def stage_surpass_b():
    """批次 B：优化预算 / SSM 容量 / 混叠精调（T0/T4/T9 × 5 种子）。

    依据批次 A：前端容量扩容无效、低频扩展仅 T4 +1.3（within-batch 干净的
    768 对对比）。本轮检验三条新假设（统一前端 = 低频扩展版）:
      sb_long3x        : 训练预算 x3 + cosine（当前仅 60 epoch，疑欠优化）
      sb_a25           : alias 2.5（T9 定向）
      sb_ssmbig        : SSM 容量 ~2x（n_blocks 4, d_model 192）
      sb_long3x_ssmbig : 二者交互
    """
    variants = [{**SURPASS_COMMON, 'name': nm, **kw}
                for nm, kw in SURPASS_B_VARIANTS]
    return run_variants('HUST', ['T0', 'T4', 'T9'], list(range(5)), variants)


def _load_auto_spec(fname):
    import json
    with open(fname, 'r', encoding='utf-8') as f:
        return json.load(f)


# 无对齐头交付配置（AF6x）：与 sc_steps6x 逐位相同，仅 w_mmd = w_adv = 0
# 依据 align_ablate_hust（2026-09-27）：对齐项在交付前端上峰值中性、末十为负。
ALIGNFREE_KW = dict(front_mode='gaborwide', front_init='gabor', front_chan=512,
                    gabor_f_lo=0.0025, front_ksize=128, gabor_dc_frac=0.3,
                    gabor_alias=2.0, use_ssm=True, tau_mode='zero',
                    w_mmd=0.0, w_adv=0.0)


# 原交付前端配置（对齐权重走 DEFAULTS：w_mmd 0.5 / w_adv 0.1）
DELIVERED_KW = dict(front_mode='gaborwide', front_init='gabor', front_chan=512,
                    gabor_f_lo=0.0025, front_ksize=128, gabor_dc_frac=0.3,
                    gabor_alias=2.0, use_ssm=True, tau_mode='zero')


def stage_af_vs_ref_cross():
    """PU / PHM2009：同批次四臂对照“有/无对齐头”（按用户 2026-09-28 要求）。

    目的：为“是否删掉对齐头”提供同批次配对证据（跳批次比较在 §5.4 被作废：
    同配置跨批次平均绝对差 1.07、最大 2.89 点，与待测效应量同量级）。

    四臂（前端与其它开关逐位相同，仅损失与预算不同）：
      xg2_std : 对齐开（w_mmd 0.5 / w_adv 0.1）+ 各数据集默认 60 epoch 预算
      af_std  : 对齐关（0 / 0）+ 同预算
      xg2_6x  : 对齐开 + 2880 步 + 余弦
      af_6x   : 对齐关 + 2880 步 + 余弦
    每任务 5 种子 × 4 臂。PU 先跑、PHM2009 后跑（run_variants 按任务打印汇总，
    故每完成一个任务即可读部分结果）。
    """
    arms = [dict(name='xg2_std', **DELIVERED_KW),
            dict(name='af_std', **ALIGNFREE_KW),
            dict(name='xg2_6x', steps_override=2880, use_scheduler=True,
                 **DELIVERED_KW),
            dict(name='af_6x', steps_override=2880, use_scheduler=True,
                 **ALIGNFREE_KW)]
    res = {}
    for data in ('PU', 'PHM2009'):
        res[data] = run_variants(data, STD_TASKS, SEEDS5, arms)
    return res


def stage_af_final_hust():
    """无对齐头交付配置：HUST 全 10 任务 × 5 种子（2880 步 + 余弦）。

    与 `stage_auto_final`（sc_steps6x）逐位相同，只去掉 MMD 与 GRL 对抗
    两项损失。用于替换§6.1–6.2 的头条数字。
    """
    variants = [dict(name='af6x', steps_override=2880, use_scheduler=True,
                     **ALIGNFREE_KW)]
    return run_variants('HUST', HUST_TASKS, SEEDS5, variants)


def stage_af_cross_final():
    """无对齐头交付配置的跳数据集臂：PU / PHM2009，4 任务 × 5 种子 × 2 预算。

      af_std : 各数据集默认 60 epoch 预算（PU 2,640 步、PHM 1,500 步）
      af_6x  : 2880 步 + 余弦（与 HUST 交付版一致）
    对应旧版的 xg_std / xg_6x（表 9–11 的跳数据集结论）。
    """
    variants = [dict(name='af_std', **ALIGNFREE_KW),
                dict(name='af_6x', steps_override=2880, use_scheduler=True,
                     **ALIGNFREE_KW)]
    res = {}
    for data in ('PU', 'PHM2009'):
        res[data] = run_variants(data, STD_TASKS, SEEDS5, variants)
    return res


def stage_auto_batchc():
    """自动批次 C：从 auto_batchc.json 读取变体列表（autopilot 决策生成）。"""
    spec = _load_auto_spec('auto_batchc.json')
    variants = [dict(name=v['name'], **v.get('kwargs', {}))
                for v in spec['variants']]
    tasks = spec.get('tasks', ['T0', 'T4', 'T9'])
    seeds = spec.get('seeds', list(range(5)))
    return run_variants('HUST', tasks, seeds, variants)


def stage_auto_final():
    """自动决赛：从 auto_final.json 读取单变体，全 10 任务 × 5 种子。

    已存在的 (task, seed) 日志自动跳过（复用批次 B/C 已测任务）。
    """
    spec = _load_auto_spec('auto_final.json')
    v = spec['variants'][0]
    variants = [dict(name=v['name'], **v.get('kwargs', {}))]
    return run_variants('HUST', HUST_TASKS, SEEDS5, variants)


def stage_auto_widan():
    """WIDAN 同预算复核：从 auto_widan.json 读 steps_override。

    日志写 HUST_{task}_WIDAN_long_seed{seed}.log（不覆盖标准基准）。
    """
    import importlib

    mod = importlib.import_module('WIDAN')
    spec = _load_auto_spec('auto_widan.json')
    so = spec.get('steps_override')
    tasks = spec.get('tasks', HUST_TASKS)
    for task in tasks:
        vals = []
        for seed in SEEDS5:
            path = os.path.join(
                OUTDIR,
                'HUST_{}_WIDAN_long_seed{}.log'.format(task, seed))
            acc = _acc_of(path)
            if acc is not None:
                print('[skip] {} -> {:.2f}%'.format(
                    os.path.basename(path), acc), flush=True)
            else:
                if so:
                    mod.configs.steps_override = int(so)
                t0 = time.time()
                try:
                    a = run_one('WIDAN', 'HUST', task, seed, 'cuda', path)
                    acc = a * 100.0
                except Exception as e:  # noqa: BLE001
                    print('[FAIL] {} : {}'.format(
                        os.path.basename(path), e), flush=True)
                    acc = float('nan')
                print('[{}] {} -> {:.2f}% ({:.0f}s)'.format(
                    time.strftime('%H:%M:%S'), os.path.basename(path), acc,
                    time.time() - t0), flush=True)
            vals.append(acc)
        a2 = np.array(vals, dtype=float)
        print('  --- HUST / {} --- WIDAN_long: {:.2f} +- {:.2f}  {}'.format(
            task, np.nanmean(a2), np.nanstd(a2),
            [round(x, 2) for x in a2]), flush=True)
    print('### ALL DONE @ {}'.format(time.strftime('%F %T')), flush=True)


STAGES = {
    'film_hust_t0': stage_film_hust_t0,
    'film_hust_rest': stage_film_hust_rest,
    'film_pu': stage_film_pu,
    'film_phm': stage_film_phm,
    'widan_hust': stage_widan_hust,
    'dann_hust': stage_dann_hust,
    'udtl_hust': stage_udtl_hust,
    'mra_hust': stage_mra_hust,
    'tau_spec_hust': stage_tau_spec_hust,
    'tau_spec_phm': stage_tau_spec_phm,
    'ablate_tau_hust': stage_ablate_tau_hust,
    'abl_const_verify': stage_abl_const_verify,
    'abl_const_phm': stage_abl_const_phm,
    'widan_init_hust': stage_widan_init_hust,
    'widan_capacity_hust': stage_widan_capacity_hust,
    'gaborwide_hust': stage_gaborwide_hust,
    'gabor_soft_hust': stage_gabor_soft_hust,
    'gabor_alias_hust': stage_gabor_alias_hust,
    'alias_sweep': stage_alias_sweep,
    'alias_sweep_clean': stage_alias_sweep_clean,
    'ssm_ablate_hust': stage_ssm_ablate_hust,
    'ssm_liquid_ablate': stage_ssm_liquid_ablate,
    'align_ablate_hust': stage_align_ablate_hust,
    'align_ablate_clean': stage_align_ablate_clean,
    'tau_ssm_ablate_clean': stage_tau_ssm_ablate_clean,
    'frontend_ablate_clean': stage_frontend_ablate_clean,
    'af_final_hust': stage_af_final_hust,
    'af_cross_final': stage_af_cross_final,
    'af_vs_ref_cross': stage_af_vs_ref_cross,
    'final10_hust': stage_final10_hust,
    'surpass_a': stage_surpass_a,
    'surpass_b': stage_surpass_b,
    'ema_pilot': stage_ema_pilot,
    'cross_final': stage_cross_final,
    'pu_alpha_capacity': stage_pu_alpha_capacity,
    'phm_mixed_factor': stage_phm_mixed_factor,
    'uo_cross': stage_uo_cross,
    'phm_capacity': stage_phm_capacity,
    'capacity_match': stage_phm_capacity,
    'nondet_ctl': stage_nondet_ctl,
    'det_ctl': stage_det_ctl,
    'det_ctl2': stage_det_ctl2,
    'det_ctl3': stage_det_ctl3,
    'auto_batchc': stage_auto_batchc,
    'auto_final': stage_auto_final,
    'auto_widan': stage_auto_widan,
}
GROUPS = {
    'film_all': ['film_hust_t0', 'film_hust_rest', 'film_pu', 'film_phm'],
    'hust_baselines': ['widan_hust', 'dann_hust', 'udtl_hust'],
    'tau_spec': ['tau_spec_hust', 'tau_spec_phm'],
    'tau_diag': ['tau_spec_hust', 'tau_spec_phm', 'ablate_tau_hust'],
    'abl_const': ['abl_const_verify', 'abl_const_phm'],
    'widan_probe': ['widan_init_hust', 'widan_capacity_hust'],
    'gaborwide': ['gaborwide_hust'],
    'gabor_soft': ['gabor_soft_hust'],
    'gabor_alias': ['gabor_alias_hust'],
    'alpha_sweep_v03': ['alias_sweep_clean'],
    'align_ablate_v03': ['align_ablate_clean'],
    'tau_ssm_ablate_v03': ['tau_ssm_ablate_clean'],
    'frontend_ablate_v03': ['frontend_ablate_clean'],
    'ssm_ablate': ['ssm_ablate_hust'],
    'ssm_liquid': ['ssm_liquid_ablate'],
    'align_ablate': ['align_ablate_hust'],
    'final10': ['final10_hust'],
    'attribution': ['pu_alpha_capacity', 'phm_mixed_factor'],
    'all': ['film_hust_t0', 'film_hust_rest', 'film_pu', 'film_phm',
            'widan_hust', 'dann_hust', 'udtl_hust',
            'tau_spec_hust', 'tau_spec_phm', 'ablate_tau_hust',
            'abl_const_verify', 'abl_const_phm', 'widan_init_hust',
            'widan_capacity_hust'],
}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'film_hust_t0'
    if stage in GROUPS:
        todo = GROUPS[stage]
    elif stage in STAGES:
        todo = [stage]
    else:
        print('unknown stage: {}'.format(stage))
        print('available: {}'.format(sorted(list(STAGES) + list(GROUPS))))
        return

    os.makedirs(OUTDIR, exist_ok=True)
    print('### stages: {}  start {}'.format(todo, time.strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    for s in todo:
        print('\n>>> STAGE {} @ {}'.format(s, time.strftime('%H:%M:%S')), flush=True)
        STAGES[s]()
    print('\n### ALL DONE @ {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)


if __name__ == '__main__':
    main()
