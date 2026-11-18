#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUST 严格嵌套留一工况选 α（T0/T9 首批）。

外层目标域完全不参与选参：对每个 outer target，在其余 9 个工况内做
inner leave-one-condition-out；α 候选固定为当前论文扫描集合，按 inner
validation 的 last5 平均准确率选择。随后用全部非目标工况重新训练 5 个
seed，并只在 outer target 上评估。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.HUST_bearing import HUST_DG  # noqa: E402
from run_cell import (  # noqa: E402
    BATCH_SIZE, EPOCHS, InfiniteDataLoader, ResampleDataset,
    make_config, setup_seed,
)
import agmamba.method as M  # noqa: E402
from run_stage import _delivered_configs  # noqa: E402

DOMAINS = list(range(10))
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
TUNE_SEED = 0
FINAL_SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(ROOT, "results_nested_hust")


def task_id(domain):
    return "T{}".format(domain)


def read_last5(accs):
    a = np.asarray(accs, dtype=float).reshape(-1)
    return float(np.mean(a[-5:]) * 100.0)


def train_eval(tag, train_domains, eval_domain, alpha, seed, outer,
               phase):
    """Train on source domains and evaluate only on eval_domain."""
    assert eval_domain not in train_domains
    setup_seed(seed)
    M.configs = _delivered_configs()
    M.configs.gabor_alias = float(alpha)

    spec = [list(train_domains), [eval_domain]]
    srcs, target, num_cls = HUST_DG(spec, "mean-std", 50, 50).train_test_data_split()
    srcs = [ResampleDataset(s) for s in srcs]
    target = ResampleDataset(target)
    iters = max(1, (len(srcs[0]) + BATCH_SIZE - 1) // BATCH_SIZE)
    steps = EPOCHS * iters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_config(M, "AGMamba", num_cls, steps, iters, device,
                      n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = M.AGMamba(cfg)

    log_dir = os.path.join(OUT, "outer_{}".format(task_id(outer)), phase)
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "{}_eval{}_a{}_s{}.log".format(
        tag, task_id(eval_domain), str(alpha).replace('.', 'p'), seed))
    logger = logging.getLogger("nested_{}_{}_{}_{}".format(
        outer, phase, tag, seed))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(logfile, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    loaders = [InfiniteDataLoader(s, BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(target, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)
    result = model.train_model(zip(*loaders), [test_loader], logger)
    accs = np.asarray(result["acces"], dtype=float).reshape(-1)
    return {
        "outer": task_id(outer),
        "phase": phase,
        "tag": tag,
        "train_domains": [task_id(x) for x in train_domains],
        "eval_domain": task_id(eval_domain),
        "alpha": float(alpha),
        "seed": int(seed),
        "last5": read_last5(accs),
        "best": float(np.max(accs) * 100.0),
        "log": logfile,
    }


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_outer(outer):
    outer_dir = os.path.join(OUT, "outer_{}".format(task_id(outer)))
    inner_path = os.path.join(outer_dir, "inner_scores.json")
    inner = load_json(inner_path) or []
    source = [d for d in DOMAINS if d != outer]

    for alpha in ALPHAS:
        for val in source:
            key = (float(alpha), int(val))
            exists = any(float(x["alpha"]) == key[0] and
                         x["inner_val"] == task_id(key[1]) for x in inner)
            if exists:
                continue
            train_domains = [d for d in source if d != val]
            rec = train_eval(
                "inner_{}_val{}".format(str(alpha).replace('.', 'p'), task_id(val)),
                train_domains, val, alpha, TUNE_SEED, outer, "inner")
            rec["inner_val"] = task_id(val)
            inner.append(rec)
            save_json(inner_path, inner)
            print("[inner] outer={} alpha={} val={} last5={:.2f}".format(
                task_id(outer), alpha, task_id(val), rec["last5"]), flush=True)

    scores = {}
    for alpha in ALPHAS:
        vals = [x["last5"] for x in inner if float(x["alpha"]) == alpha]
        scores[str(alpha)] = {
            "n": len(vals),
            "mean_last5": float(np.mean(vals)) if vals else None,
            "folds": vals,
        }
    selected = max(ALPHAS, key=lambda a: (scores[str(a)]["mean_last5"], -a))
    selection = {
        "outer": task_id(outer),
        "outer_target_excluded_from_all_inner_runs": True,
        "candidate_alphas": ALPHAS,
        "tune_seed": TUNE_SEED,
        "selection_metric": "mean of inner-fold last5 target accuracy",
        "scores": scores,
        "selected_alpha": selected,
    }
    save_json(os.path.join(outer_dir, "selected_config.json"), selection)
    print("[select] outer={} alpha={} scores={}".format(
        task_id(outer), selected,
        {k: round(v["mean_last5"], 3) for k, v in scores.items()}), flush=True)

    final_path = os.path.join(outer_dir, "final_scores.json")
    final = load_json(final_path) or []
    for seed in FINAL_SEEDS:
        if any(int(x["seed"]) == seed for x in final):
            continue
        rec = train_eval("final", source, outer, selected, seed, outer, "final")
        final.append(rec)
        save_json(final_path, final)
        print("[final] outer={} alpha={} seed={} last5={:.2f}".format(
            task_id(outer), selected, seed, rec["last5"]), flush=True)
    return selection, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", default="T0,T9",
                    help="outer targets, e.g. T0,T9")
    args = ap.parse_args()
    targets = [int(x[1:]) for x in args.outer.split(",") if x]
    all_out = {}
    for outer in targets:
        selection, final = run_outer(outer)
        all_out[task_id(outer)] = {"selection": selection, "final": final}
    save_json(os.path.join(OUT, "summary_T0_T9.json"), all_out)
    print("[saved]", os.path.join(OUT, "summary_T0_T9.json"), flush=True)


if __name__ == "__main__":
    main()
