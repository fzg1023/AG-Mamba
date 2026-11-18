# 统一协议对比（多方法公平对比）v1

> 本文档定义 `experiments/run_compare.py` 的**冻结协议**。任何改动都必须在此记
> 录（见 §7 变更记录），因为改动会使已有数字与新数字**不可比**。
>
> 与 `docs/PROTOCOL.md` 的三套协议（P-交付 / P-探针 / P-发布）**互不换算**：
> 那三套是方法内部消融用的，本文档是**多方法横向对比**用的第四套协议（记 **P-统一**）。

---

## 1. 为什么需要单独一套协议

历史表格（`docs/PROTOCOL.md` Table 3 等）里各方法的数字来自**各自发布协议**：
预算、调度、评估频率都不同，且读数用 `peak`（逐评估最大值）——含有
"检查点选择"成分，会系统性高估。要回答"我们的方法相对基线是否更好"，
必须让所有方法在**同一数据、同一预算、同一评估网格、同一读数**下重跑。

关键历史事实：**PHM2009 上大多数基线的旧日志根本无法支撑新读数**
（`ADN`/`CCN`/`CDDG` 等只记录结尾一次的 `tgt_train_acc`，没有逐轮目标域精度），
因此不能复用，必须重跑。

---

## 2. 冻结的公平性五要素

| # | 要素 | 取值 | 实现位置 |
|---|---|---|---|
| 1 | **数据与划分** | `run_cell.TASKS[dataset][task]`，`load_data` 同一划分；重采样长度 1024 | `run_cell.py` |
| 2 | **预算** | **60 epoch**，`steps = 60 × iters_per_epoch`（PHM2009 = 1500 步） | `run_compare.EPOCHS` |
| 3 | **评估网格** | `checkpoint_freq = iters_per_epoch` ⇒ **恰好 60 个评估点** | `run_compare.run_one` |
| 4 | **种子** | `{0,1,2,3,4}`，同种子跨方法**严格配对** | `run_compare.SEEDS` |
| 5 | **batch / 优化器** | batch 64；各方法沿用其原优化器与 lr（见 §4，不做调参差异） | `run_cell.make_config` |

> 第 5 条的口径说明：**不统一 lr**。原因是各基线用的优化器不同
> （Adagrad / SGD / Adam），强行统一 lr 会人为压低某些方法。lr 取
> `run_cell.LR_OVERRIDE`（各方法快扫调优后的值），这是一个**已公开、可复核**的
> 固定表，而不是"我们方法单独调参"。

### 2.1 各数据集的规模参数（由 `--plan` 实测确认）

`EPOCHS = 60` 恒定，但 `iters_per_epoch` 由该数据集的源域样本数决定，
因此不同数据集的 `steps` 不同；**评估点恒为 60 个**（每 epoch 一次），
`last5` 主口径在所有数据集上一致。

| 数据集 | 任务 | 源域数 | 类别 | 每源域样本 | 目标样本 | iters/epoch | steps | 单元数 |
|---|---|---|---|---|---|---|---|---|
| PHM2009 | T0–T3 | 3 | 8 | 1600 | 400 | 25 | 1500 | 440 |
| PU | T0–T3 | 3 | 14 | 2800 | 700 | 44 | 2640 | 440 |
| HUST | T0–T9 | 9 | 9 | 450 | 450 | 8 | 480 | 1100 |

> HUST 的单元数 = AGMamba 2×10×5 + 20 基线 ×10×5 = 100 + 1000 = 1100（若沿用 5 种子）。
> UO 待跑时用 `--plan` 确认后补入此表。
>
> ⚠ 每个数据集**开跑前必须先归档该数据集的旧结果**：旧日志中格式兼容的会被
> `[skip]` 复用，把不同协议的数字混入新表（PHM2009 与 PU 都已归档，见 §7）。

---

## 3. 调度臂（`--arms`）

`run_cell.make_config` 给基线的默认就是**阶梯衰减**：
`schedule = [20·iters, 40·iters]`、`cos = False`，即**在预算的 1/3、2/3 处把
lr 乘 0.5**。而 AG-Mamba 的发布配置（`agmamba/configs.yaml`）是**余弦退火至零**。

| 臂 | 含义 | AG-Mamba | 基线 |
|---|---|---|---|
| `unif` | **统一协议**：全部方法用阶梯衰减（milestones = 1/3、2/3，每档 ×0.5） | 覆盖 `sched_mode='step'` | 本即此行为 |
| `pub` | **发布配置**：各方法用自己的默认调度 | 余弦退火至零 | 与 `unif` **相同** |

因此 **`ARMS_MAY_DIFFER = {'AGMamba'}`**：只有 AG-Mamba 需要跑两个臂；
其余 13 个基线两臂等价，**只跑 `unif`**，避免算力翻倍
（`run_compare.py` 的 `build_overrides()` 返回 `None` 即跳过）。

PHM2009 单元数 = AGMamba 2×4×5 = 40，基线 20×4×5 = 400，**合计 440**。
方法总数 21 = 本方法 + 13 个移植基线（`baselines/*.py`）+ 7 个 origeight 基线（§4.1）。

---

## 4. 方法与学习率

| 方法 | lr | 优化器备注 |
|---|---|---|
| **AGMamba（本方法，干净版）** | 0.001 | Adam |
| CCDG / DDGFD | 0.02 | SGD 需更高 lr |
| ADN / IEDGNet / CCN | 0.01 | Adagrad |
| MRA_SNN | 0.005 | SNN 替代梯度需较大 lr（原版 0.01） |
| CNNC / DGNIS / WhiteningNet / CDDG / UDTL / WIDAN | 0.002 | 快扫调优 |
| DANN | 0.001 | 默认 |
| **AGG / MMD / CORAL / Triplet_loss / Mixup / MLDG / DAEL** | 0.001 | origeight 家族（§4.1），沿用 DANN 移植值 |

### 4.1 origeight 家族：7 个方法的移植说明

原仓库的 8 个自有方法（`docs/legacy/方法对比分析报告.md` §2）中，**DANN 已有独立模块
`baselines/DANN.py`**，其余 7 个此前**没有统一接口实现**（旧实现散落在 `utils/train.py`
的 `if args.method == ...` 分支与 `utils/{MLDG,DAEL}_train.py` 里，且依赖已废弃的
`datasets.PHM2009_DG` / `args` 参数式接口）。现移植为 `baselines/origeight.py`。

**保真依据**：该文档 §2 明确这 8 个方法「共享同一套轻量主干：4 层 1D CNN 特征提取器
（卷积核 25/15/5/5，输出 512 维）+ MLP 分类器（512→256→256→类别数）」——即
`models/extractor.py::Feature_extractor` + `models/classifier.py::Classifier`，
与 `baselines/DANN.py` 用的完全一致。故沿用同一主干属**复现原架构**。

| 方法 | 损失（与原始实现逐条对应） |
|---|---|
| AGG | $\mathcal{L}_{\rm CE}$，无任何域泛化项 |
| MMD | $\mathcal{L}_{\rm CE}+\tau\cdot\overline{\rm MMD}(f_i,f_j)$ |
| CORAL | $\mathcal{L}_{\rm CE}+\tau\cdot\overline{\rm CORAL}(f_i,f_j)$ |
| Triplet_loss | $\mathcal{L}_{\rm CE}+\tau\cdot{\rm Triplet}(f,y)$，**margin = 2** |
| Mixup | $\mathcal{L}_{\rm CE}+\sum_k\lambda_k\mathcal{L}_{\rm CE}(g(\tilde x),y_k)$，$\tilde x=\sum_k\lambda_k x_k$，$\lambda\sim{\rm Dir}(1,\dots,1)$ |
| MLDG | 内环 meta-train 一步 + 外环 meta-test 反传，$\nabla\leftarrow\nabla_{\rm in}+\beta\nabla_{\rm out}$ |
| DAEL | $\overline{\mathcal{L}_{\rm CE}}+\tau\cdot\overline{\rm MSE}\big(\overline{g_{j\neq k}(f_k)},g_k(f_k)\big)$ |

**原样保留的超参**：6 个非元学习方法**共用同一个 $\tau$ = `trade_off`（取 0.01，与原
代码单一 `args.trade_off`、以及已移植 DANN 的取值一致）；`TripletLoss(margin=2)` 为原
代码硬编码值（**不是** `models/loss.py` 的默认 0.3）。

**必要的一般化**（原实现硬编码 3 源域；HUST 有 9 源域）：CORAL/MMD 改为全部配对 $i<j$
的均值、Mixup 泛化为 $n_{\rm src}$ 分量 Dirichlet、DAEL 分类器数与源域数一致、MLDG
随机留 1 域作 meta-test。**$n_{\rm src}=3$ 时与原始实现逐位等价。**

> ⚠ **口径声明**：这 7 个是**本仓库的移植实现**，不是原作者调优后的代码。因此它们的
> 数字应表述为"我们的移植版本在统一协议下的结果"，**不得**与历史表（`docs/PROTOCOL.md`
> Table 3 等）中来自外部遗留目录的 ORIG8 数字混用或对照。MLDG 的 $\beta$ 取标准值 1.0，
> 与原代码同样名为 `trade_off` 但**语义不同**（元目标权重，非对齐损失权重），故未复用 0.01。

**本方法为"干净版"**（见 `docs/PROTOCOL.md` §7）：无域对齐头
（`w_mmd = w_adv = 0`，域判别器不构建）、无液体项（`tau_mode='zero'`，
τ₀ 估计器与 γ 调制不构建）。损失仅 $\mathcal{L} = \mathcal{L}_{\rm CE} + 0.01\|\Delta\|^2$。

---

## 5. 读数口径

### 5.1 主口径：`last5`（末 5 个评估点均值）

| 口径 | 定义 | 定位 |
|---|---|---|
| **`last5`（主）** | **末 5 个评估点（即最后 5 个 epoch）的均值** | **正文主表** |
| `last1` | 末个评估点（= 第 60 轮，**实际要部署的那个模型**） | **正文并列列出**（部署口径，最保守） |
| `last10` | 末 10 个评估点均值 | **正文并列列出**（稳健性口径） |
| `peak` | 一次运行内逐评估最大值 | **仅存分析文件，不进正文**（偏置随方法变化） |

**为何取 5**（不是拍脑袋，依据见 `analysis/readout_window_study.py` 与
`docs/manuscript/_readout_window_study.md`）：

1. **估计目标决定了窗口应偏小。** 协议固定 60 epoch，要交付的就是末轮模型，
   故 `last1` 才是估计目标；末-k 均值回答的是"若停止轮次随机落在末段 k 轮"的期望性能，
   作用只是**不让结论悬于恰好第 60 轮**。因此应取**能让结论稳定下来的最小 k**。
2. **k ≤ 3 不稳定。** PU 上 CCN 在 k=1、k=3 不显著而在 k=2、k=4 显著（p 值在 0.05 附近抖动）；
   PU 显著数 k=1→17/20、k=2→19/20、k=3→18/20（**非单调**）。k ≥ 4 起稳定。
3. **降噪边际收益已耗尽。** 种子间 std 中位数：PHM2009 k=1→3.13、k=5→2.30（−26%）、
   k=10→2.06（仅再 −8%）；PU k=1→3.25、k=5→2.11（−35%）、k=10→1.93（仅再 −6%）。
   且实测降幅只有"各轮独立"假设下的一半左右（**相邻轮次高度相关**），
   长窗口在"降噪"这个理由上站不住。
4. **窗口越长，偏离可交付模型越多，且偏离随方法变化。** 末 k 均值 − 末轮值：
   PU 上本方法 +0.61（k=5）→ +0.71（k=10），基线 −0.15 → −0.38；
   PHM2009 上本方法 ≈ 0、基线 +0.37。即长窗口不是中性加噪，而是**方向性改变方法间比较**。
5. **k=5 与 k=10 的结论完全一致**（PHM2009 均 20/20 显著；PU 均 19/20，未显著的均为 Mixup），
   只有量级差（PU Δ +2.80 vs +3.19）。既然结论相同，就取偏离更小者。

> 诚实声明（可在论文中引用）：**更大的窗口对本方法更有利**。PU 上 k ≥ 12 可得到 20/20 显著、Δ+3.39；
> 选 k=5 是主动放弃更有利的读数。最保守的部署口径 `last1` 下 PU 仅 17/20（CCN、CDDG、Mixup 未达显著）。
>
> **论文报三个口径：`last5`（主）＋ `last1`（部署口径）＋ `last10`（稳健性）。**
> 三者 Δ 方向完全一致；同时列出是因为它们回答的是三个不同问题（随机停轮的期望性能 /
> 实际交付模型 / 更平滑的末段），共列可同时回应"为什么选 5"与"结论是否依赖窗口"。
>
> `peak` 的统计结果列在 `docs/manuscript/_cmp_<DS>_stats.md` 与 `_cmp_<DS>_rankings.md`
> 作为核算底稿，**不进入正文与附录** —— 它的偏置随方法变化（实测 0.85–6.89 分，跨度 8 倍），
> 跨方法比较不公平，列进正文只会招来质疑（而 `last1` 已经承担了"最保守口径"的角色）。

### 5.2 日志标签的两种格式（解析器必须同时兼容）

| 标签 | 方法 |
|---|---|
| `tgt_acc: {:.4f}` | 本方法、DANN、MRA_SNN、UDTL、WIDAN，**以及新移植的 origeight 7 个** |
| `tgt_train_acc: …` | ADN、CCDG、CCN、CDDG、CNNC、DDGFD、DGNIS、IEDGNet、WhiteningNet（9 个移植方法） |

第二类的标签写着 `train`，但**记录的就是目标域精度**（目标 loader 是 `loaders[0]`），
是原代码复制粘贴遗留的命名。`run_compare.parse_accs` 同时匹配两者，并兼容旧格式
`tar-Acc`（数值 >1.5 视为已是百分数）。

> ⚠ **只认 `tgt_acc` 会让上述 9 个基线解析为空、末段读数变 NaN**，且因跳过条件
> 是 `len(traj) >= 10`，它们会被反复重跑却永远产出不了数字。这是 2026-09-28
> 修掉的一个真实缺陷（详见 §7）。

---

## 6. 运行方式与产物

```bash
# 只看计划（不运行）
python experiments/run_compare.py --dataset PHM2009 --methods all --plan

# 全量对比（300 单元）
python experiments/run_compare.py --dataset PHM2009 --methods all

# 只跑本方法两个调度臂
python experiments/run_compare.py --dataset PHM2009 --methods AGMamba

# 强制重跑已有日志
python experiments/run_compare.py --dataset PHM2009 --methods all --force
```

产物：

| 路径 | 内容 |
|---|---|
| `results/<DS>/<METHOD>/<arm>/<TASK>/<DS>_<TASK>_<METHOD>_fc_<arm>_seed<n>.log` | 逐轮日志（含 60 个 `tgt_acc`） |
| `docs/manuscript/_cmp_<DS>.csv` | 逐 (方法, 臂, 任务) 的 last5 / last1 / last10 均值±std |
| `docs/manuscript/_cmp_<DS>.md` | 同上，Markdown 表 + 跨任务平均 |

运行已有日志会自动跳过（`len(traj) >= 10` 判定）；`--force` 覆盖。

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-28 | 建立 P-统一；冻结五要素与 `unif`/`pub` 两臂；修复运行循环中"未跳过等价臂"的缺陷（原本会让 13 个基线各白跑一遍 `pub`，300 → 560 单元）；新增 `write_summary()` 自动产出 CSV/MD |
| 2026-09-28 | PHM2009 旧结果（含 `*_lr0.00*` 探针与 `DANN`/`UDTL`/`WIDAN` 的 `L*`/`M*` 协议）
归档至 `results/_archive/2026-09-28_preclean_PHM2009/`，以保证全部单元同批次 |
| 2026-09-28 | **修复 `parse_accs` 只认 `tgt_acc` 的缺陷**：9 个移植基线用 `tgt_train_acc`，
原本会被解析为空、末段读数变 NaN（见 §5.2） |
| 2026-09-28 | **修复 `cfg` 单例泄漏**：`run_cell.make_config` 返回模块级 `mod.configs`，
`unif` 臂写入的 `sched_mode='step'` 会泄漏到 `pub` 臂，导致两臂跑成同一个（静默失效）。
现两臂的调度在 `ARM_STATIC` 中**显式**写出，`make_config` 亦重置 `sched_mode` |
| 2026-09-28 | **修复可复现性**：`setup_seed` 增加 `cudnn.deterministic/benchmark`
与 `CUBLAS_WORKSPACE_CONFIG`；并定位到 `mamba_ssm` CUDA 扫描内核**反传含原子累加**
（梯度 max\|Δgrad\|≈4e-8 → 1500 步放大成 ~1.3 点漂移）。AG-Mamba 的运行需设
`AGM_FORCE_PY_SCAN=1` 改用纯 PyTorch 扫描（完全确定；只影响本方法，+~2.5h）。
审计工具：`tools/check_determinism.py` |
| 2026-09-28 | **补齐 7 个缺失基线**（AGG/MMD/CORAL/Triplet_loss/Mixup/MLDG/DAEL），
移植自 `utils/train.py` 与 `utils/{MLDG,DAEL}_train.py`（见 §4.1）。
方法总数 14 → 21，PHM2009 单元数 300 → **440** |
| 2026-09-28 | **修复汇总覆盖缺陷**：`write_summary` 原本只汇总"本次命令行指定的方法"，
分批跑（先本方法、后基线）时**后一批会把前一批的表覆盖掉**。现改为
`discover_units()` 从 `results/<DS>/` **扫描磁盘上已有的一切**，一次生成完整表 |
| 2026-09-28 | **修复 `TripletLoss` 的性能缺陷**：hard-mining 原为逐样本 Python 循环，
batch=192 时单步含 192 次 GPU 同步，实测 670 s/单元（DANN 仅 42 s）。
改为向量化（与 `baselines/DGNIS.py`、`baselines/IEDGNet.py` 已有副本一致），
**loss 与梯度逐位等价**（\|diff\| = 0，max\|Δgrad\| = 0），40 s/单元，提速 16.8× |
| 2026-09-28 | 新增 `experiments/run_compare.py` 的**逐方法进度报告**：每完成一个算法
即基于磁盘扫描输出一次完整排行榜（`logs/progress_<DS>.txt`），便于随时比较 |
| 2026-09-28 | 新增 `tools/run_compare_dataset.sh`（数据集通用启动器，内含
`AGM_FORCE_PY_SCAN=1`）与 `tools/run_after_phm2009.sh`（用 `pgrep` 等待前一批
真正结束后自动串联下一数据集，不依赖估时） |
| 2026-09-28 | **PU 加入 P-统一**：预算 60 epoch × 44 iters = **2640 步**，14 类，
单元数 440。旧 PU 结果（383 日志 / 70.3 MB，含 `lr0.002` 探针与 legacy 的
origeight 目录）归档至 `results/_archive/2026-09-28_preclean_PU/`。
冒烟验证：AGG 与 AGMamba 各 1 单元，`failed=0` |
| 2026-09-29 | 新增 `analysis/summary_compare.py`：**配对显著性检验**（同 `(task,seed)`
严格配对的双侧 Wilcoxon）。PHM2009 结果：**20/20 个基线全部显著落后**
（最强基线 `Mixup` Δ=+3.44, p=7e-4, 17/20；`MMD` Δ=+3.65, p<1e-4, 17/20）。
80 个（基线×任务）格子中仅 2 格为负（`Mixup` T3 −1.52、`WIDAN` T3 −2.73） |
| 2026-09-29 | **HUST 加入 P-统一**：10 任务（T0–T9）、**9 个源域**、480 步、单元数 1100。
旧 HUST 结果归档至 `results/_archive/2026-09-28_preclean_HUST/`（430 成员）。
**9 源域冒烟**特意覆盖 5 个未测路径：`Mixup` 的通用 Dirichlet 分支（legacy
实现硬编码 3 域）、`DAEL` 的 9 分类器、`MMD`/`CORAL` 的 36 个域对、`MLDG` 的
8+1 元划分、`AGMamba` 的 576 batch —— `failed=0` |
| 2026-09-29 | 新增 `tools/run_chain.sh <等待数据集> <下一个数据集>`：用 `pgrep`
轮询真实进程，前一批结束后自动串联下一批，不依赖估时 |
| 2026-09-29 | 新增 `tools/watch_agmamba.sh <DS...>`：**AGMamba 单点提醒钩子**。
命中后打印含 AGMamba 的完整排行榜 + 与最强基线差距，落盘
`logs/AGMAMBA_DONE_<DS>.txt` 后退出（退出即提醒）。兼容两种批次形态：
`--methods all`（有 `[method done] AGMamba`）与 `--methods AGMamba`
（单方法批次**没有**该行，只有 `### 完成 done=N`） |
| 2026-09-30 | **PU 完成**：440 单元，21 方法，耗时 23h47m（09-29 05:20 → 09-30 05:07）。
AGMamba `pub` **69.13** / `unif` **68.19**，最强基线 `Mixup` 65.00（Δ=+4.13/+3.19） |
| 2026-09-30 | **修复 `datasets_src` 硬编码（严重，静默失效类）**：
`run_cell.make_config` 原先硬编码 `cfg.datasets_src = ['s1','s2','s3']`。
9 个移植基线（ADN/CCN/CDDG/CNNC/DGNIS/IEDGNet/WhiteningNet）用
`len(configs.datasets_src)` 当 `num_domains`，而域判别类损失要求
`标签数 == logits 行数 == n_src × batch_size`。3 源域的 PHM2009/PU 上**恰好正确**，
故此前未被发现；HUST（9 源域）上 ADN 报
`Expected input batch_size (576) to match target batch_size (192)` 全 50 单元崩溃
（stub 日志仅 177 B）。已改为 `make_config(..., n_src=len(srcs))` 按真实划分生成，
5 个调用点全部同步。**影响**：PHM2009/PU 结果不受影响（源域数本就为 3，逐位等价）；
HUST 需**重跑 ADN/CCN/CDDG/DGNIS 四个方法**（各 50 单元）；UO（4 源域）此前会踩同一坑。
判定详情：`CNNC` 算出 `domain_labels` 但**从未使用** → 安全；`IEDGNet`/`WhiteningNet`
的 reshape 在 `use_domain_weight=False` 守卫内 → 安全 |
| 2026-09-30 | **主口径从 `last10` 改为 `last5`**（§5.1）。依据 `analysis/readout_window_study.py`
在 PHM2009/PU 上的窗口扫描：① 估计目标是末轮模型（`last1`），末-k 均值只是防"巧合第 60 轮"，
故应取**能稳定结论的最小 k**；② k ≤ 3 不稳定（PU 显著数 17→19→18→19 非单调，CCN 来回跳），
k ≥ 4 起稳定；③ 降噪边际收益已尽（PHM2009 k=5→10 仅再降 8% std，PU 仅再降 6%，
且实测降幅只有"各轮独立"假设的一半 → 相邻轮次高度相关）；④ 窗口越长偏离可交付模型越多且
**随方法变化**（PU 本方法 +0.61→+0.71 vs 基线 −0.15→−0.38）；⑤ k=5 与 k=10 结论完全一致
（PHM2009 均 20/20；PU 均 19/20，未显著者均为 Mixup），只有量级差。
**论文并列报三个口径**：`last5`（主）+ `last1`（部署口径，最保守）+ `last10`（稳健性）；
**`peak` 不进论文**（偏置随方法变化 0.85–6.89 分，跨方法不可比），仅在
`_cmp_<DS>_stats.md` / `_rankings.md` 留档。
⚠ 诚实披露：更大的窗口对本文方法**更有利**（PU k≥12 可得 20/20、Δ+3.39），选 k=5 是主动放弃有利读数 |
| 2026-09-30 | **HUST 批次断电续跑（18:01 中断）+ 两个崩溃修复的实证验证**。断电现场：
  MRA_SNN/T1/seed3 留下 44 个评估点的**半截日志**。据此新增**完整性闸门**：
  `run_compare.py` 的 `[skip]` 由 `len(tr) >= 10` 改为 `>= EPOCHS(60)` ——
  否则半截日志会冒充"已完成"被永久跳过，且末-k 读数取自错误的检查点区间
  （诊断工具 `tools/check_incomplete.py`；本次体检：完整 463 / 半截 1 / stub 200） |
| 2026-09-30 | **修复 `DGNIS` 硬编码 8 个故障分类头（严重）**：`fc1..fc8` 再按源域数切片，
  HUST（**9 源域**）只能切出 8 个 → `update()` 里 `self.fcs[8]` 抛
  `IndexError: list index out of range`（全 50 单元崩溃，且是**修完 `datasets_src`
  之后**才暴露的第二层坑）。改为按源域数构造。
  ⚠ **关键约束**：构造个数必须写 `max(8, num_domains)`，**不能**写 `num_domains`。
  首版按源域数建头后，PU 旧结果**不再可复现**（逐点最大差 25.1 分）。根因：
  `utils/DatasetClass.InfiniteDataLoader` 的 `RandomSampler` 用**全局 torch RNG**
  （无独立 generator），而它在**建模之后**才构造 —— 模型初始化消耗的 RNG 量会
  改变数据采样顺序。建满 8 个即与历史实现逐位同构（≤8 源域）；
  >8 源域才多建，不影响任何已有数据集。
  **实证**：`[SAME] DGNIS@PU len 60 vs 60 max|diff|=0.0000` → PU/PHM2009 **无需重跑** |
| 2026-09-30 | **HUST 修复的 7 单元实证**（`logs/_verify_fix.out`）：
  `ADN` 82.20 / `CCN` 72.13 / `CDDG` 73.94 / `DGNIS` 72.16（各 pts=60，原为全崩）；
  另用**重跑逐位比对**证明 `CNNC` / `IEDGNet` / `WhiteningNet` 在 9 源域下
  **未被污染**（`max|diff|=0.0000`）→ 其已有 HUST 结果可直接使用，
  不必像 ADN/CCN/CDDG/DGNIS 那样重跑 |
| 2026-09-30 | **批次口径统一**：清理 5 个 09-29 冒烟残留
  （`CORAL/MMD/MLDG/Mixup/DAEL` 的 T0/seed0，协议头虽一致但非本批产出），
  使 HUST 表中所有单元同源同批 |
| 2026-10-01 | **HUST 批次完成**：1100 单元，`done=642 skipped=458 failed=0`，耗时 960.4 min。
  断电前的 582 单元全部复用（逐单元头部核查协议一致），补跑 518 单元。 |
| 2026-10-01 | **汇总产物与 §5.1 对齐（消除"主口径"自相矛盾）**：
  `run_compare.write_summary` 早期把 `last10` 写成"主口径"，与 §5.1 冲突。
  已改为 **`last5`（主）+ `last1`（部署）+ `last10`（稳健）+ `peak`（留档）** 四列，
  并把完整性闸门（须满 EPOCHS 个评估点）一并接入；`report_progress` 的 cross、
  `[skip]` / 单单元日志行同步改为 `last5`。重生成 `_cmp_HUST.md` / `.csv` |
| 2026-10-01 | **新增 `analysis/make_all_methods_report.py`**：按**现行协议**汇总全方法
  基准（主口径 last5，逐任务 mean±std + 三口径对照）。取代历史
  `analysis/summary_*_all.py` 家族（那些是 **peak 口径**，且读归档前的旧日志）。
  旧的 `HUST_all_methods_report.md`（peak 口径、preclean 结果）已归档至
  `docs/legacy/HUST_all_methods_report_2026-09-28_preclean.md` 并加"勿引用"横幅：
  与现行结果差异显著（`ADN 88.02→84.19`、`AGG 91.51→88.58`、`DANN 92.68→84.62`） |
| 2026-10-01 | **收敛为单一事实源（消除报表重复/口径漂移）**：根目录的 `<DS>_all_methods_report.md`
  一度与本模块的输出重复，容易再次出现"两个事实源口径不一致、下游静默出错数字"。定案：
  **规范性表格一律以 `docs/manuscript/_<ds>_tables.md`（`make_61_tables.py` 生成）为准**；
  `make_all_methods_report.py` 降为**工具库（主要）+ 临时速览 CLI（次要）**，
  其 CLI **默认只打印到 stdout**，需显式 `--out` 才落盘。
  根目录 `HUST_all_methods_report.md` 已删除。**根目录现已无任何报表**（仅剩 `README.md`） |
| 2026-10-01 | **PU / PHM2009 旧报表归档 + 三个 v0.2 生成器退休**（B2 方案）：
  `PU_all_methods_report.md` / `PHM2009_all_methods_report.md`（均 best-epoch 口径）
  归档至 `docs/legacy/*_2026-09-28_preclean.md` 并加"勿引用"横幅；
  其生成器 `summary_pu_all.py` / `summary_cross_all.py` / `summary_cross_af.py`
  **实测为死代码**（沿用旧臂名 `xg_std`/`xg_6x`/`base` 与旧文件名规则
  `<DS>_{task}_{method}_seed{s}.log`，**无 `_fc_<arm>_` 段** → 读不到现行日志，
  重跑只吐一列 `–`），已移至 `docs/legacy/*.py.v0.2-era`（无任何模块 import 它们，已核）。
  同一管线产出的 `docs/manuscript/_s62_data.md`（§6.2 跨数据集底稿）**也是陈旧口径**
  （PU `xg_std` 74.21 vs 现行 69.12），已加横幅但**保留**（§6.2 待按 P-统一重做） |
| 2026-10-02 | **α 扫描完成（§6.3 (a)，v0.3 协议）**：新增 `run_stage.py alias_sweep_clean`
  + `analysis/summary_alpha_sweep.py` → `docs/manuscript/_s63_alpha.md`。
  60 单元（6 点 α × HUST T0/T9 × 5 种子），**交付配置 + 480 步**，`last5` 主口径。
  **同源性自检**：α=2.0 臂与主对比 `pub` 臂逐点 `max\|Δ\|=0.000000`（10/10）→
  整条曲线与主表同源。**结论**：最优 α = **2.0**（池化 85.11；0/0.5 → 76.8、
  1.0 → 81.7、3.0 → 82.7），全部替代点显著更差（池化 p ≤ 0.006，wins ≤ 1/10）；
  收益**集中在 T9**（α=0 → 62.02 vs α=2.0 → 76.07，**+14.1**；T0 仅差 2.7）。
  ⚠ **推翻旧结论**：v0.2 的 `_s63_data.md`（best-epoch/2880 步/带对齐头）曾显示
  α=3.0 在 T9 更高（82.09 vs 79.56）；新协议下 α=3.0 T9 = 72.67 < 76.07
  → **交付值 α=2.0 是最优**，无需改配置 |
| 2026-10-02 | ⚠ **probe 阶段的共性问题（新发现，影响 §6.3–6.5 全部消融）**：
  `run_stage.run_variants` 对**所有** stage 都先套 v0.2 的 `DEFAULTS`
  （`w_mmd=0.5 / w_adv=0.1 / tau_mode='normal' / use_scheduler=False /
  gabor_f_lo=0.005 / front_ksize=64`）—— 与 v0.3 交付配置不符。
  故 `alias_sweep`（旧）、`align_ablate_hust`、`ssm_ablate_hust`、`ablate_tau_hust`、
  `gabor_soft_hust`、`capacity_*` 等**都不可直接用于 v0.3 正文**；
  已按 `alias_sweep_clean` 的模式（每变体从 `configs.yaml` 重载 + 只改目标字段）
  提供干净的替代路径。**建议给 `run_variants` 加 `defaults=None` 参数统一改造** |
| 2026-10-02 | ⚠ **AGMamba 变体臂必须 `AGM_FORCE_PY_SCAN=1`**（与主对比一致）：
  官方 `mamba_ssm` CUDA 内核反传含原子累加；实测**同配置同种子**下 60 个评估点里
  **53 个不同**、mid-training 差 **9.78 分**（末段恰好收敛回同一水平，极易漏判）。
  探针臂另需 `AGM_RESULTS=<隔离目录>`，否则会被 `run_compare.discover_units`
  扫成"AGMamba 的新臂"，污染 `_cmp_<DS>.md/.csv` 与进度榜 |
| 2026-10-01 | ⚠ **待办（同类隐患，未处理）**：`analysis/summary_af_vs_ref.py` 与
  `analysis/summary_attribution.py` 仍按 **peak / last-ten** 口径产出
  `_s62_af_vs_ref.md` / `_s62b_attribution.md`；`analysis/make_two_rule_tables*.py`、
  `make_single_rule_tables.py` 同属 v0.2 的"两口径"家族。这些**不进 P-统一正文**，
  但若被引用需先核口径 |
| 2026-10-01 | **HUST 结论（配对 Wilcoxon，n=50，主口径 last5）**：
  **19/20 个基线显著领先**；唯一**不显著**的是 `WIDAN`（Δ=+2.06，wins 21/50，p=0.78）——
  三项口径方向一致（last1 Δ=+0.76、last10 Δ=+3.47），只有 `peak` 反号（−0.96）。
  优势最薄的格子：`AGG/MMD/CORAL/MRA_SNN` 在 **T1–T3 上 Δ≈0**（±2 分内）。
  → HUST **不得**表述为"全面领先"，须与 PU（不显著者 Mixup）同样单独披露 |
| 2026-10-01 | **⚠ WIDAN 上 Wilcoxon 与配对 t 检验结论不一致（必须如实披露）**：
  主口径 last5，Wilcoxon `p=0.7795`（**不显著**），配对 t `p=0.0387`（边缘显著），
  `wins=21/50`（几乎对半）。即显著性来自少数大正离群点、而非整体位移。
  **协议口径 = Wilcoxon → 判定为"不显著"**；`unif`/`pub` 两臂同样模式
  （pub: p_W=0.7647, p_t=0.0372, wins=19/50）。正文须两值并列，不可只报 t |
| 2026-10-01 | **`analysis/make_61_tables.py` 重写**（原版归档至
  `docs/legacy/make_61_tables_v0.2-era.py`）。原版三个致命问题：① 数据源是中间报表
  `HUST_all_methods_report.md`（报表改口径即**静默**出错数字）；② 口径是 best-epoch
  (peak)；③ 引用已消失的臂 `AGMamba_fc_sc_steps6x` / `WIDAN_long` / `WIDAN_std`
  （`results/` 与 `_archive/` 内均无）。新版**只读 `results/`**，主口径 last5，
  产出 `HUST-T1`（主对比，22 方法/臂 × 10 任务）、`HUST-T2`（关键对手 + 逐任务 Δ）、
  `HUST-T3`（三口径稳健性 + 配对检验，Wilcoxon 与配对 t **并列**）。
  **v0.2 的"决定性对比（480 vs 2880 预算）"不再生成**：P-统一已把所有方法预算固定为
  480 步，该对比语义已被取代；预算曲线请走 `analysis/exp_e6_efficiency.py`。
  `同号` 的定义与 `summary_compare.py` 对齐（`last5`/`last1`/`last10`/`peak` 四口径同向） |
| 2026-10-02 | **新增 `analysis/summary_align_ablate.py`**（§6.4 对齐头消融数值底稿）。读
  `results_probe_align/`（由 `run_stage.py align_ablate_clean` 产出），4 臂
  `aln_noall`(=交付配置, w_mmd/w_adv=0) / `aln_nommd`(0/0.1) / `aln_noadv`(0.5/0) /
  `aln_full`(0.5/0.1) × HUST T0/T4/T9 × 5 种子 = 60 单元。主口径 last5，并列
  last1/last10/peak；配对 Wilcoxon vs `aln_noall`（同 task×seed，严格配对，因 w=0 只
  移除 loss 项、不改前向路径与随机数消耗）。**同源性自检**：`aln_noall` 与主对比
  `pub` 臂逐点 `max\|Δ\|=0.000000`（15/15 单元）。输出 `docs/manuscript/_s64_align.md`。
  用法：`python analysis/summary_align_ablate.py [--dir results_probe_align]` |
