# 未收录的脚本（v0.2 遗留 / 已被取代 / 非实验性）

本包只收录**定稿口径**用到的文件（**49 个** = 47 个脚本 + 2 个口径快照）。
仓库 `experiments/`、`analysis/`、`logs/`、`tools/` 下共 **110 个脚本**，去向如下：

| 去向 | 数量 | 说明 |
|---|---|---|
| **本包收录** | 47 | 见 `MANIFEST.csv`（另含 `docs/` 两个非脚本快照） |
| **归 `paper_figures/`** | 10 | 出图脚本，与实验脚本分开归集 |
| **本文件排除** | 53 | `analysis/` 36 + `experiments/queues/` 7 + `tools/` 10 |
| 合计 | **110** | ✓ 与仓库逐一对账 |

另有 `experiments/arms/` 的 4 个 `.json`（臂定义，非脚本，见第三节）。

排除原因是：它们属于 v0.2 时期、已被同名新脚本取代、或与实验无关。
**它们仍在仓库里原样存在**——本包只做归集，未移动/删除任何仓库文件。

判断依据（三条，任一命中即排除）：

1. **产出的记录文件已被明令禁止引用**（`_s6*_data.md`、`_s62*`、`_s65_data/e2/e3*`、`_two_rule_tables.md`、`_fig16_*` 等）；
2. **脚本依赖已删除的模块**（`run_dgfdb` / `run_final_eval`）→ 实际**跑不起来**；
3. **按旧编号出图**，与定稿图号冲突（新编号见 §0 的置换表）。

---

## 一、`analysis/` —— 36 个

### 1.1 依赖已删模块、跑不起来（4）

| 脚本 | 产出 | 为什么排除 |
|---|---|---|
| `exp_e1_closure.py` | `_s65_data.md`、`e1e2.npz` | v0.2 口径；`_s65_data.md` 已禁引用 |
| `exp_e2_morph.py` | `_s65_e2.md`、`e2.npz` | 同上 |
| `exp_e3_recalc.py` | `_s65_e3_recalc.md`、`e3_stats.npz` | 同上 |
| `exp_e3_trained.py` | `_s65_e3.md`、`e3.npz` | **import `run_dgfdb` / `run_final_eval`（已删除）→ ImportError** |

> 已被 `analysis/exp_s65_dilation.py`（§6.5 定稿版）取代：后者去掉退役指标 η(s)，改用交付配置基座。

### 1.2 产出 v0.2 口径记录、已被取代（6）

| 脚本 | 产出 | 取代者 |
|---|---|---|
| `summary_63.py` | `_s63_data.md`（best-epoch / 2880 步 / dc=0.5） | `summary_alpha_sweep.py` + `summary_front_ablate.py` |
| `summary_64.py` | `_s64_data.md` | `summary_align_ablate.py` + `summary_tau_ssm_ablate.py` |
| `summary_af_vs_ref.py` | `_s62_af_vs_ref.md` | `summary_compare.py` |
| `summary_attribution.py` | `_s62b_attribution.md` | `summary_compare.py`（P-统一口径） |
| `summary_align.py` | （对齐旧摘要） | `summary_align_ablate.py` |
| `summary_hust_all.py` | （`summary_*_all.py` 系列） | `make_61_tables.py` + `summary_compare.py` |

> 仓库里另有 `docs/legacy/summary_*_all.py.v0.2-era`、`make_61_tables_v0.2-era.py`，属同一批归档。

### 1.3 表格脚本按 peak / last-ten 口径（4）

| 脚本 | 产出 |
|---|---|
| `make_two_rule_tables.py` | `_two_rule_tables.md` |
| `make_two_rule_tables2.py` | `_two_rule_tables.md` |
| `make_single_rule_tables.py` | （peak / last-ten） |
| `make_hust_table_rows.py` | （last10） |

### 1.4 探索性一次性分析（12，只打印、无固定产物）

`analyze_abl10.py`、`analyze_best_vs_final.py`、`analyze_best_vs_final2.py`、
`analyze_budget_plateau.py`、`analyze_lastk.py`、`analyze_tau_trajectory.py`、
`analyze_v3.py`、`capacity_means.py`、`compare_v4_widan.py`、
`check_lastten_cover.py`、`paired_all.py`、`paired_deployment.py`

> `analyze_lastk.py` / `check_lastten_cover.py` / `paired_*.py` 都属 **last-ten 口径**时期；
> 其位置已被 `analysis/readout_window_study.py`（§5.4 / 附录 A）与 `summary_compare.py` 的严格配对检验取代。

### 1.5 按旧编号出图，与定稿图号冲突（10）

`make_fig1.py`（工作稿合成，已被 `make_fig1_panels.py` 取代）、
`make_fig4.py`、`make_fig567.py`、`make_fig8.py`、`make_fig9.py`、
`make_fig10.py`、`make_fig11.py`、`make_fig12.py`、`make_fig13.py`、`make_fig14_15.py`

> 定稿的 10 个出图脚本不在本包，而在 **`paper_figures/analysis/`**
> （`make_fig1_orbit` / `make_fig2_design_space` / `make_fig3_arch` / `make_fig4_filterbank` /
> `make_fig5_cross` / `make_fig6_frontend` / `make_fig7_audit` / `make_fig8_dilation` /
> `make_fig9_efficiency` / `make_fig10_hust`）。
> ⚠ `make_fig1_panels.py` 还会写出**旧编号**的 `fig2.pdf` / `fig3.pdf`，重跑会在 `figures/` 里
> 造出误导性文件名——这正是 `paper_figures/` 里改用裁剪版 `make_fig1_orbit.py` 的原因。

---

## 二、`tools/` —— 10 个

| 脚本 | 为什么排除 |
|---|---|
| `run_chain.sh`、`run_after_phm2009.sh`、`_watch_next.sh` | v0.2 时期跨数据集串行编排，硬编码旧流程 |
| `archive_dataset.py`、`archive_agmamba_results.py` | 结果归档运维（非实验） |
| `inventory_results.py`、`inventory_results2.py`、`inventory_tables.py` | 清单盘点运维 |
| `sync_from_legacy.py` | 旧仓库同步 |
| `check_manuscript.py` | 写作辅助（早期版本），已被手稿内的自查脚本取代 |

---

## 三、`experiments/` 的早期编排附件 —— 7 个脚本 + 4 个 JSON

- `experiments/queues/`（7 个 `.sh`）—— 2026-09-28 的早期队列编排，对应 v0.2 的 stage：
  `run_align_queue.sh`、`run_afref_queue.sh`、`run_attribution_queue.sh`、`run_capacity_queue.sh`、
  `run_idle_det_queue.sh`、`run_alignfree_queue.sh`、`check_configs.sh`。
- `experiments/arms/`（4 个 `.json`：`arms.json` / `auto_batchc.json` / `auto_final.json` / `auto_widan.json`）
  —— 喂给 `run_stage.py` 的 `auto_batchc` / `auto_final` / `auto_widan` 三个 v0.2 决赛 stage。

---

## 四、附：不随包的"数据"（不是脚本，但要清楚）

| 内容 | 规模 | 说明 |
|---|---|---|
| `results/` | 452.4 MB / 2364 文件 | 主对比训练日志，摘要脚本的输入 |
| `results_probe_front` / `_align` / `_alpha` / `_tau` | 各 5.2–7.8 MB | 探针批次日志 |
| `results_probe_snr` | 1.3 MB / 27 | SNR 网格 |
| `results_geom` + `weights_geom` | 1.1 MB / 8 + 权重 | 特征几何 |
| v0.2 记录文件 | 16 个 `_s6*_data.md` / `_s62*` / `_fig16_*` | 已禁引用；仍留在仓库（附录 C.4 的负面记录需要它们作溯源） |

> 记录文件与图件的归集见姊妹包 `paper_figures/`（含 `EXCLUDED.md`）。
