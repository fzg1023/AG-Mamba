# AG-Mamba 实验索引与口径约定（v1，doc-only）

> 本文档**不改变任何实验**，只把"每张表/图用的是哪套协议、哪个前端、哪种读数、哪些任务与种子、由哪个阶段跑出"写清楚。
> 目的：让读者（以及我们自己）能一眼判断**哪些数字可以互相比较、哪些绝对不能**。

---

## 1. 三套协议（这是全文最重要的区分）

| 记号 | 名称 | 训练预算 | 调度器 | 前端 | 用途 |
|---|---|---|---|---|---|
| **P-交付** | 交付协议 | 2,880 步 | 余弦退火至零 | **F-交付**：512 ch，$K_s{=}128$，$f_{\rm lo}{=}0.0025$，`dc_frac` 0.3，α = 2.0 | §6.1–6.2 主结果 |
| **P-探针** | 探针协议 | 480 步 | **无** | **F-探针**：512 ch，$K_s{=}64$，$f_{\rm lo}{=}0.005$，`dc_frac` 0.3，α = 2.0 | §3.3、§6.3–6.5 机制实验 |
| **P-发布** | 各方法发布预算 | 基线各自的发布协议（多为 60 epoch = HUST 480 步） | 随方法 | 随方法 | §6.1 参照组、Table 3 |

> ⚠️ **P-探针 与 P-交付 的差异有三处**（预算、调度器、**前端**），因此两者的**绝对值不可比**；两者之间只允许比较"同一批次内部的配对差值"。
> 另：P-探针的日志是在**前身损失**（对齐项开启）下跑的，而交付配置已改为无对齐目标 —— 见局限 L5。

## 2. 四种读数

| 记号 | 定义 | 用在哪 |
|---|---|---|
| `peak` | 一次运行内所有评估点的最大值（含检查点选择成分） | Table 3（参照组记录）、Table 13、Table 14、Table 16、Table 17 末列、Table 21 种子散布 |
| `final-ten` | 固定调度**末十个检查点均值**（本文 §5.2 的正式读数） | Table 4、5、6、9、10、11、12 |
| `@step` | 两条 2,880 步轨迹在**匹配步数**处的读数（无检查点选择） | Table 15 |
| 分析量 | 不是精度（合成信号响应、核统计、FLOPs/时延） | Table 17–20、Fig. 1–2、5–10 |

## 3. 任务集与种子

| 记号 | 含义 | 用在哪 |
|---|---|---|
| **全 10** | HUST T0–T9（T0 = 20 Hz … T9 = 80 Hz） | Table 3、4、5、15、21 |
| **锚点 3** | HUST **T0 / T4 / T9**（跨带外推 / 带内 / 带外推） | Table 8、13（HUST 列）、14（对齐块）、16 |
| **锚点 2** | HUST **T0 / T9** | Table 6、7 |
| **跨数据集** | PU T0–T3、PHM2009 T0–T3 | Table 9–13（PU/PHM 列） |
| 种子 | 5 种子为主；Table 7 参照行 10 种子、Table 14 部分 10 种子、Table 19 每格 1 种子 | — |

---

## 4. 逐表/逐图索引

| ID | 块 | 协议 | 读数 | 任务 × 种子 | 源阶段（臂名） |
|---|---|---|---|---|---|
| Table 1 | E0 | — | — | 基准说明 | — |
| Table 2 | E0 | — | — | 基线分类说明 | — |
| Table 3 | E0 | P-发布 | `peak` | 全 10 × 5 | `mra_hust` + 21 个基线日志 + 本方法行 |
| Table 4 | E0 | P-交付 | `final-ten` | 全 10 × 5 | `sc_steps6x`、`WIDAN_long`、`base`/`film` |
| Table 5 | E0 | P-交付 | `final-ten` | 全 10 × 5 | 同 Table 4（配对检验） |
| Table 6 | E1 | P-探针 | `final-ten`（附 `peak`） | 锚点 2 × 5 | `alias_sweep`（α=0…3） |
| Table 7 | E1 | P-探针 | `peak` | 锚点 2 × 5/10 | `widan_init_hust`、`widan_capacity_hust`、`gaborwide_hust`、`gabor_soft_hust`、`gabor_alias_hust` |
| Table 8 | E1 | P-探针 | `peak` | 锚点 3 × 5 | `surpass_a`（`sa_*`）、容量行 |
| Table 9 | E0 | P-交付（PU 发布 60 ep） | `final-ten` | PU × 5 | `cross_final`（`xg_std`/`xg_6x`/`base`） |
| Table 10 | E0 | P-交付（PHM 发布 60 ep） | `final-ten` | PHM × 5 | 同 Table 9 |
| Table 11 | E0 | P-交付 | `final-ten` | PU/PHM × 5 | 同 Table 9–10（配对检验） |
| Table 12 | E0 | P-交付 | `final-ten` | PHM × 5 | `phm_mixed_factor`（协议 B/C） |
| Table 13 | E1/E3 | P-交付 | **`peak`** | 锚点 3（n=15）+ PU/PHM（n=20）× 5 | `pu_alpha_capacity`、`phm_capacity`（`cap960*`） |
| Table 14 | E2/E3 | P-探针 | `peak`（Δ） | 锚点 2 / 1 / 3 × 5–10 | `ssm_ablate_hust`、`ssm_liquid_ablate`、`ablate_tau_hust`、`abl_const_verify`、`abl_const_phm`、`tau_spec_*`、**`align_ablate_hust`** |
| Table 15 | E3 | P-交付 | `@step` | 全 10 × 5 | `sc_steps6x`、`WIDAN_long` |
| Table 16 | E3 | 混合（P-交付 / P-探针） | `peak` | 锚点 3（+全 10） × 5 | `ema_pilot`、`surpass_b/c`、`sd_12x`、`cap960` |
| Table 17 | 分析 | — | 闭包 η(s) | 合成梳（无训练） | 前向分析 + Table 6 的 T9 `peak` |
| Table 18 | 分析 | — | 合成响应相似度 | 合成冲击串（无训练） | 前向分析 |
| Table 19 | E1 | P-探针 | 核统计 ζ/χ/η | 训练前后（每格 1 种子） | 核审计（权重已存档） |
| Table 20 | 分析 | — | 参数量/FLOPs/时延 | 推理（无训练） | `exp_e6_efficiency.py` |
| Table 21 | E0/E3 | P-交付 | `peak` + 训练开销 | 全 10 × 5 | 日志解析 |
| Fig. 1–2 | 分析 | — | — | HUST 谱 / 核统计平面 | 前向分析 |
| Fig. 3 | E1 | P-探针 | `peak` | 锚点 2 × 5 | `gaborwide_hust` 等 |
| Fig. 4 | — | — | — | 架构图 | `make_fig4.py` |
| Fig. 5–7 | 分析 | — | 原子/覆盖 | 无训练 | `make_fig567.py` |
| Fig. 8–10 | 分析 | — | 原子/Bessel 梳/核统计 | 无训练 | `make_fig8/9/10.py` |
| Fig. 11 | E1 | P-探针 | `final-ten` | 锚点 2 × 5 | `alias_sweep` |
| Fig. 12–13 | E1 | 分析 + P-探针 | 闭包/核审计 | 合成 + 每格 1 种子 | `make_fig12/13.py` |
| Fig. 14 | E0 | P-交付 | `final-ten` | PHM + PU + HUST | `make_fig14_15.py` |
| Fig. 15 | 装置 | P-交付 | 同上 | PU T0/T2、HUST T9 | `nondet_ctl`、`det_ctl*`、`rep_a/b` |
| Fig. 16–17 | E0 | P-交付 | t-SNE / 混淆矩阵 | PU T2、HUST T9（seed 0） | `run_tsne_cm.py`（权重 `agm_ckpt/`） |

---

## 5. 臂名注册表（历史沉积名 → 含义）

| 臂名前缀 | 含义 | 协议 |
|---|---|---|
| `base` / `film` | 朴素窄带前端（对照）/ 加 FiLM 的变体 | P-探针 |
| `sa_*` | 批次 A 候选：容量 768/1024、低频扩展 | P-探针 |
| `sb_*`、`sc_*` | 批次 B/C 候选：预算、α=2.5、SSM 扩容；`sc_steps6x` = **交付配置（前身损失）** | P-交付 |
| `xg_std` / `xg_6x` | 跳数据集交付臂（发布预算 / 2,880 步） | P-交付 |
| `v3_*`、`v4_noL` | SSM/液体三联消融；`v4_noL` = 交付配置的探针对应物 | P-探针 |
| `abl*`、`tau_*` | τ₀ 处理与估计器消融 | P-探针 |
| `cap960*`、`puac_*`、`phm_*` | 容量对齐（960 ch）与两数据集上的容量/弥散阶梯 | P-交付 |
| `det_ctl*`、`rep_a/b`、`nondet_ctl` | 可复现性控制（装置本身） | P-交付 |
| **`af6x` / `af_std` / `af_6x`** | **干净版交付臂：无对齐头（判别器不构建）+ 无液体项（τ₀ 通路不构建）**（2026-09-28 起，已注册进 `arms.json`） | P-交付 / P-探针 |
| **`xg2_*`** | 无对齐判定的**同批次**有对齐对照臂（w_mmd 0.5 / w_adv 0.1） | P-交付 |

---

## 6. 由本索引直接得出的三条结论

1. **只有 §6.1–6.2（Table 4/5/9–12/15/21）与 §6.6（Table 20）是"交付配置"的数字**，其余机制表都是探针配置 —— 稿中已由 §5.2 声明，但读者需要索引才看得清。
2. **Table 13 与 Table 6 的读数不同**（前者 `peak`、后者 `final-ten`），且任务集不同（前者含 T4 与 PU/PHM）→ 两张表不能互相引用数值，只能各自内部比较。
3. **Table 14 的 `peak` 口径**与 §5.2 的 `final-ten` 规则不一致，属于"机制审计沿用历史口径"，已在表题注明；若要彻底一致，需要重跑（见 `实验设计_v1_单一协议.md` 的 Stage C，暂缓）。

---

## 7. 干净版（交付主干）代码级约定

"干净"不是只把权重置 0，而是**相关模块在配置下根本不构建**，以保证
"我们没有用对齐/液体项"这句话在代码层面成立、且省去无用的前向与显存：

| 开关 | 取值 | 行为 |
|---|---|---|
| `w_adv` | `0.0` | `AGMamba.disc` 不创建、不进优化器、前向不做 GRL（`method.py`） |
| `w_mmd` / `w_jmmd` / `w_cdan` | `0.0` | 三个全为 0 时**整段对齐逻辑跳过**，不做 z 的域切分与核计算 |
| `tau_mode` | `'zero'` | `use_tau = False`：`γ`、`tau_proj`、`FiLM` 均不创建；前向**不调用包络/FFT 估计器**，`Δ = softplus(W_dt x + b)` |

对应的可复现臂：`af6x`（HUST 全 10 × 2880 步）、`af_std` / `af_6x`
（PU/PHM2009）；有对齐对照为 `xg2_std` / `xg2_6x`。驱动阶段：
`af_final_hust`、`af_cross_final`、`af_vs_ref_cross`。

---

## 7. 一致性欠账清单（索引暴露出来的，尚未修）

| # | 问题 | 影响 | 修法 | 成本 |
|---|---|---|---|---|
| U1 | Table 13 用 `peak`，§5.2 规则是 `final-ten` | 读者会误以为与 Table 4 同口径 | 表题已注明；彻底统一需重跑 960 通道臂 | 中 |
| U2 | Table 14 用 `peak`（Δ） | 与 §5.2 不一致 | 同上；或改名为"机制审计表（peak 口径）"并前移表题说明 | 低 |
| U3 | **Table 16 把 P-del 与 P-probe 行混在一张表**（参照行预算不同） | 行间 Δ 不可直接比 | 拆成两张表，或每行标注其协议列 | 低 |
| U4 | Table 7 同表混用 5 与 10 种子 | 标准差不可比 | 表题已注明"reference rows pool ten seeds"；可拆行 | 低 |
| U5 | "Pooled" 在不同表里含义不同（T0/T9 vs T0/T4/T9 vs 全 10） | 跨表引用会错 | A.6 已给任务集列；表题补一句 | 低 |
| U6 | 探针表（6.3–6.5）用的是**前身损失**（对齐开启） | 与交付配置的损失不同 | 局限 L5 已声明；彻底修需 Stage C | 高 |
| U7 | Fig. 3/11 的探针绝对精度 vs §6.1 的交付表 | 数字不可比 | §5.2 + A.6 已声明 | 已处理 |

> 结论：**U1–U5 属文档可修**（本次已完成 A.6 索引，其余可在 30 分钟内补完）；**U6 需重跑**（Stage C）。

---

## 8. 结果与权重目录规范（v2，已落地）

### 8.1 布局

```
results/<DATASET>/<METHOD>/<ARM>/<TASK>/<DATASET>_<TASK>_<METHOD>[_fc_<ARM>]_seed<k>.log
weights/<DATASET>/<METHOD>/<ARM>/<同名>.pt | _feat.npz
```

* `<METHOD>`：算法名（`AGMamba` / `WIDAN` / `UDTL` / `DANN` / `CCNC` …）；基线只有一个配置时 `<ARM>=main`。
* `<ARM>`：本方法的一个配置臂（`xg_std` / `v4_noL` / `aln_full` / …）；`main` 表示交付/默认配置。
* `<TASK>`：任务标识（`T0`…`T9`，或 `L1` / `M3` 等额外协议条件）。
* 命名**自描述**：不打开目录也能从文件名判断数据集、任务、算法、臂、种子。

### 8.2 唯一入口

`agmamba/paths.py` 是读写路径的唯一权威：

| 用途 | 调用 |
|---|---|
| 写日志 | `log_path(dataset, task, method, seed, arm='main')` |
| 读日志 | `find(basename)` —— 按文件名递归定位，返回规范路径 |
| 遍历 | `iter_logs(dataset=None, method=None, arm=None)` |
| 权重目录 | `weights_dir(dataset, method, arm='main')` |
| 概览 | 直接运行 `python agmamba/paths.py` |

根目录可用环境变量重定向，无需改代码：`AGM_RESULTS` / `AGM_WEIGHTS`。

### 8.3 `_archive/`：同名冲突的去歧义规则

旧仓库中同一 `(数据集, 任务, 算法, 种子)` 在多个扁平目录下存在**内容不同**的副本
（共 360 组，全部位于 `results_dgfdb` ↔ `results_dgfdb_tuned`/`_tuned2` 之间）。
旧脚本依赖目录优先级择一，合并为单一树后按文件名索引会产生歧义。处理方式：

1. 按旧读取方的真实优先级选定唯一版本，放入规范路径；
2. 落选版本移入 `results/_archive/superseded/<来源目录>/` 留档；
3. `_archive/` 被 `find()` 与 `iter_logs()` **显式跳过**。

优先级取自旧脚本的 `PORT_DIRS` / `DATA_DIR`：

| 算法族 | 优先级（高 → 低） |
|---|---|
| ORIG8（`AGG DANN MMD CORAL Triplet_loss Mixup MLDG DAEL`） | 数据集目录 → `tuned2` → `tuned` → `dgfdb` |
| BENCH9（`CNNC ADN IEDGNet DGNIS WhiteningNet CCN CCDG CDDG DDGFD`） | `tuned2` → `tuned` → `dgfdb` |
| `WIDAN` / `UDTL` | `results_widan` / `results_udtl` |
| 本方法各臂、HUST 基线 | `tuned2` → `tuned` → `dgfdb` |

数量：规范日志 2,908 条；归档 358 条；合计 3,266 条。

### 8.4 迁移后的对拍验收

对 22 个只读分析脚本在**旧树（扁平目录）**与**新树（规范布局）**分别运行并逐行比对，
归一化掉 `LM-DGNet` → `AG-Mamba` 改名与输出路径后：

* **20 个脚本输出完全一致**（含 §6.1–6.6 的全部底稿脚本）；
* `summary_hust_all.py`：内容一致，仅方法清单的字母序变化（`AGMamba` 现排在 `DANN` 之前，原名 `LM_DGNet` 在后）；
* `make_two_rule_tables.py`：**被取代的旧重复脚本**，其旧的 `dgfdb` 优先顺序现统一到与权威脚本
  `make_two_rule_tables2.py` 相同（即手稿口径，如 `CCN` 66.68 而非 58.01）；两者现互相一致。

配套修复（本轮发现并已修）：`compare_v4_widan.py` 中 `_findlog()` 被误用于**未格式化模板**；
`make_single_rule_tables.py` / `summary_pu_all.py` 的方法清单缺项；`run_tsne_cm.py` 沿用旧的
`run_dgfdb` / `run_final_eval` 模块名与扁平权重路径。

### 8.5 与旧树同步

新增实验仍在旧树运行时，用增量同步工具把它们按同一规范并入：

```bash
python tools/sync_from_legacy.py          # 增量同步日志；结束后打印当前布局概览
```

