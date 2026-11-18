#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-selection exploratory alpha sweep for HUST T4/T8.

This deliberately writes to an isolated AGM_RESULTS directory.  It is not
used for model selection; it only completes the target-domain sensitivity
curves for Fig. 16(b) after the protocol has frozen alpha=2.0.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agmamba import method as M
from agmamba import paths as P
try:
    from .run_cell import run_one
    from .run_stage import ALPHA_ARMS, _delivered_configs, _points
except ImportError:  # direct execution from the experiments directory
    from run_cell import run_one
    from run_stage import ALPHA_ARMS, _delivered_configs, _points


def main():
    tasks = ["T4", "T8"]
    seeds = list(range(5))
    summary = {}
    started = time.time()

    for arm, alpha in ALPHA_ARMS:
        summary[arm] = {}
        for task in tasks:
            vals = []
            for seed in seeds:
                path = P.log_path("HUST", task, "AGMamba", seed, arm=arm)
                pts = _points(path)
                t0 = time.time()
                if len(pts) < 60:
                    M.configs = _delivered_configs()
                    M.configs.gabor_alias = float(alpha)
                    try:
                        run_one("AGMamba", "HUST", task, seed, "cuda", path)
                        pts = _points(path)
                    except Exception as exc:
                        print("[FAIL] {}: {}".format(os.path.basename(path), str(exc)[:160]), flush=True)
                        pts = []
                last5 = float(np.mean(pts[-5:])) if len(pts) >= 5 else float("nan")
                summary[arm]["{}_s{}".format(task, seed)] = last5
                print("[{}] {} alpha={:.1f} last5={:.2f} n={} ({:.0f}s)".format(
                    time.strftime("%H:%M:%S"), os.path.basename(path), alpha,
                    last5, len(pts), time.time() - t0), flush=True)
                vals.append(last5)
            print("  --- {} alpha={:.1f}: mean={:.2f} sd={:.2f} values={} ---".format(
                task, alpha, float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)),
                [round(x, 2) for x in vals]), flush=True)
        print("  === alpha={:.1f} done; elapsed {:.0f}s ===".format(alpha, time.time() - started), flush=True)

    out = os.path.join(P.RESULTS, "alpha_summary_t4_t8.npz")
    np.savez(out, **{arm: np.array([summary[arm]["{}_s{}".format(t, s)]
                                      for t in tasks for s in seeds], dtype=float)
                      for arm, _ in ALPHA_ARMS})
    print("SUMMARY_FILE {}".format(out), flush=True)


if __name__ == "__main__":
    main()
