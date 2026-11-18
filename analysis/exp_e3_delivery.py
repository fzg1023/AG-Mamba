#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付配置版 E3：训练前后首层频谱与核漂移诊断。

该脚本只用于机制一致性检查，不用于替换主实验结果。它固定当前交付配置，
比较 Gabor alpha=0、AG alpha=2 和 Laplace 三种首层初始化，在 HUST T0/T4/T9
上各运行三个种子。每个单元保存训练前后首层权重、频谱统计、核漂移和末五个
目标准确率，支持中断后按单元续跑。
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import agmamba.method as M  # noqa: E402
from run_cell import (  # noqa: E402
    BATCH_SIZE, EPOCHS, ResampleDataset, InfiniteDataLoader,
    load_data, make_config, setup_seed, TASKS,
)
from run_stage import _delivered_configs  # noqa: E402

OUT = os.path.join(ROOT, "results_probe_mechanism")
CELL_DIR = os.path.join(OUT, "cells")
TASKS_E3 = ["T0", "T4", "T9"]
SEEDS = [0, 1, 2]
VARIANTS = {
    "gabor_a0": {"front_init": "gabor", "gabor_alias": 0.0},
    "ag_a2": {"front_init": "gabor", "gabor_alias": 2.0},
    "laplace": {"front_init": "laplace"},
}


def stats(w):
    """Return per-bank spectral coverage and temporal-shape statistics."""
    p = w.detach().float().cpu().squeeze(1)
    power = torch.fft.rfft(p, n=4096, dim=-1).abs().pow(2).numpy()
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    freq = np.fft.rfftfreq(4096)
    cdf = np.cumsum(power, axis=1)
    bw95 = np.array([freq[np.searchsorted(row, 0.95)] for row in cdf])
    centroid = (power * freq[None, :]).sum(axis=1)
    entropy = -(power * np.log(power + 1e-12)).sum(axis=1)
    zcr = ((p[:, :-1] * p[:, 1:]) < 0).float().mean(dim=1).numpy()
    return {
        "bw95_mean": float(bw95.mean()),
        "bw95_std": float(bw95.std()),
        "centroid_mean": float(centroid.mean()),
        "entropy_mean": float(entropy.mean()),
        "zcr_mean": float(zcr.mean()),
    }


def train_one(variant, task_name, seed):
    setup_seed(seed)
    M.configs = _delivered_configs()
    for key, value in VARIANTS[variant].items():
        setattr(M.configs, key, value)

    srcs, target, num_cls = load_data("HUST", TASKS["HUST"][task_name])
    srcs = [ResampleDataset(s) for s in srcs]
    target = ResampleDataset(target)
    iters = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
    steps = EPOCHS * iters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_config(M, "AGMamba", num_cls, steps, iters, device,
                      n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = M.AGMamba(cfg)
    w0 = model.model.front.conv1.weight.detach().cpu().clone()

    logger = logging.getLogger("e3_delivery_{}_{}_{}".format(variant, task_name, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = os.path.join(OUT, "logs",
                            "{}_{}_s{}.log".format(variant, task_name, seed))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    loaders = [InfiniteDataLoader(s, BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(target, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)
    result = model.train_model(zip(*loaders), [test_loader], logger)
    w1 = model.model.front.conv1.weight.detach().cpu().clone()
    accs = np.asarray(result["acces"], dtype=float).reshape(-1)
    delta = w1 - w0
    drift = float(delta.norm() / w0.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(
        w0.reshape(1, -1), w1.reshape(1, -1)).item())
    record = {
        "variant": variant,
        "task": task_name,
        "seed": seed,
        "steps": int(steps),
        "last5_acc": float(np.mean(accs[-5:]) * 100.0),
        "best_acc": float(np.max(accs) * 100.0),
        "drift_rel_l2": drift,
        "cosine": cosine,
        "init_stats": stats(w0),
        "trained_stats": stats(w1),
    }
    return w0.numpy(), w1.numpy(), record


def main():
    os.makedirs(CELL_DIR, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        for task in TASKS_E3:
            for seed in SEEDS:
                stem = "{}_{}_s{}".format(variant, task, seed)
                path = os.path.join(CELL_DIR, stem + ".npz")
                if os.path.exists(path):
                    with np.load(path, allow_pickle=False) as z:
                        rows.append(json.loads(str(z["record"])))
                    print("[skip]", stem, flush=True)
                    continue
                w0, w1, record = train_one(variant, task, seed)
                np.savez_compressed(path, w0=w0, w1=w1,
                                    record=json.dumps(record))
                rows.append(record)
                print("[done] {} last5={:.2f} drift={:.4f} bw95={:.4f}->{:.4f}".format(
                    stem, record["last5_acc"], record["drift_rel_l2"],
                    record["init_stats"]["bw95_mean"],
                    record["trained_stats"]["bw95_mean"]), flush=True)

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("[saved]", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
