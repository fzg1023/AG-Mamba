#!/usr/bin/python
# -*- coding: utf-8 -*-
"""§6.8 特征几何 —— t-SNE（源/目标特征）+ 目标域混淆矩阵。

对照两条臂，**唯一变量 = 首层的解析先验**：
  ours : 交付配置原样（agmamba/configs.yaml：受控混叠 Gabor 组、512 通道、α = 2）
  rand : 只把 front_init 换成同形状的 Kaiming 随机核（形状/通道/损失/调度/预算全同）

这对应 §6.3 / Table 6 的 `fr_rand` 臂（池化 −9.24，0/15 胜；T9 −17.23）。

**协议**：与 §5.2 冻结协议一致 —— 两臂同预算、同调度、同种子（seed 0），
步数 = 60 × ceil(N_src/64)，即 PU **2640**、HUST **480**（**不是** v0.2 探针的
2,880 步）。配置基座是 `run_stage._delivered_configs()`（直接读 configs.yaml），
**不套 v0.2 的 `DEFAULTS`** —— 后者带着 w_mmd=0.5 / w_adv=0.1 /
tau_mode='normal' / front_chan=128 / gabor_alias=0.0 / use_scheduler=False，
会让本图与主对比不可比（W3 已禁其回流）。

权重与特征落盘，可断点复用；历史权重若与当前臂不一致会自动重训
（`_is_stale_ckpt`，键含 front_init / w_mmd / w_adv / tau_mode / front_mode /
front_chan / gabor_alias）。

用法:
  python -u run_tsne_cm.py              # 全部单元 + 出图
  python -u run_tsne_cm.py smoke        # 冒烟（每格 20 步，不覆盖正式权重）
  python -u run_tsne_cm.py plot         # 只用已有 npz 出图
  python -u run_tsne_cm.py run PU T2    # 追加单元

产物:
  agm_ckpt/{data}_{task}_{arm}_seed0.pt        权重（含 cfg 与最终精度）
  agm_ckpt/{data}_{task}_{arm}_seed0_feat.npz  特征（源/目标 z + 标签）
  results_tsne/{data}_{task}_{arm}_seed0.log   训练日志
  docs/manuscript/figures/fig11_geom_pu.{pdf,png}      图（PU T2）
  docs/manuscript/figures/fig12_geom_hust.{pdf,png}    图（HUST T9）
  docs/manuscript/figures/fig11a/b/c_panel.*           单面板导出
  docs/manuscript/figures/fig12a/b/c_panel.*
  docs/manuscript/_s68_geom.md                 数值汇总（供图注核对）

**版式**：面板图题一律置于坐标轴**下方**（`cap()`，与 Fig. 6/8/10 同规则）；
单面板图用显式 rect + bbox_inches='tight' 导出，便于排版拼版。
"""
import os
import sys
import gc
import time
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agmamba import method as M                     # noqa: E402  载入 configs
import run_cell as R                    # noqa: E402  TASKS / load_data / ResampleDataset
from run_stage import _delivered_configs   # noqa: E402 交付配置（configs.yaml）
from agmamba import paths as _P         # noqa: E402  结果 / 权重目录规范

CKPT = _P.WEIGHTS
LOGD = _P.RESULTS
FIGD = 'docs/manuscript/figures'
DATAMD = 'docs/manuscript/_s68_geom.md'      # ⚠ 不再是 v0.2 的 _fig16_data.md
SEED = 0
N_SRC_PER_DOMAIN = 150          # 每源域抽样的样本数（t-SNE 用）
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# 两臂的**唯一变量 = 首层的解析先验**：`ours` 直接用交付配置（不覆盖任何
# 开关），`rand` 只把 `front_init` 换成同形状的 Kaiming 随机核 —— 形状、
# 通道数、损失、调度、预算全部逐位相同。
#   · 对应 §6.3 / Table 6 的 `fr_rand` 臂（池化 −9.24，0/15 胜；T9 −17.23）。
#   · ⚠ 旧版用 `front_mode='cnn', front_chan=128, gabor_alias=0` 做对照，那同时
#     改了**带宽、弥散、容量**三项，不是单变量对照，已弃用。
OURS_KW: dict = {}
CTRL_KW = {'front_init': 'random'}
ARMS = {'ours': OURS_KW, 'rand': CTRL_KW}
ARM_LABEL = {
    'ours': 'delivered: analytic Gabor first layer (512 ch, $\\alpha$=2)',
    'rand': 'randomised first layer (same shape, Kaiming)',
}

# 任务表：每格 (data, task, steps_override)。**steps_override=None** 即用
# §5.2 冻结协议的预算：steps = 60 × ceil(N_src/64) → PU **2640**、HUST **480**。
CELLS = [
    ('PU', 'T2', None),      # PU 四任务之一
    ('HUST', 'T9', None),    # 极端外推（80 Hz）
]

# 图号与文件名（2026-10-03 定：§6.8 特征几何 → Fig. 11 / Fig. 12）
OUTNAME = {('PU', 'T2'): 'fig11_geom_pu',
           ('HUST', 'T9'): 'fig12_geom_hust'}
FIGNO = {('PU', 'T2'): 10, ('HUST', 'T9'): 11}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def build_cfg(data, task_name, arm, num_cls, n_src0, steps_override,
              smoke=False):
    """按 `run_cell.make_config` 的方式构造配置。

    n_src0 = 单个源域的样本数（用于算 iters_per_epoch → 默认步数）。
    """
    # ⚠ 基座 = **交付配置**（直接读 `agmamba/configs.yaml`），不套 v0.2 的
    # `DEFAULTS` —— 见文件头说明。每个单元都重载一次，避免上一格的
    # front_init='random' 泄漏到下一格。
    M.configs = _delivered_configs()
    for k, val in ARMS[arm].items():
        setattr(M.configs, k, val)
    M.configs.steps_override = (int(steps_override)
                                if steps_override is not None else None)
    iters = max(1, (int(n_src0) + R.BATCH_SIZE - 1) // R.BATCH_SIZE)
    steps = R.EPOCHS * iters
    cfg = R.make_config(M, 'AGMamba', num_cls, steps, iters, DEV,
                        n_src=len(R.TASKS[data][task_name][0]))
    cfg.n_domains = len(R.TASKS[data][task_name][0])
    if smoke:
        cfg.steps = 20
    return cfg


def load_splits(data, task_name):
    task = R.TASKS[data][task_name]
    srcs, tar, num_cls = R.load_data(data, task)
    srcs = [R.ResampleDataset(s) for s in srcs]
    tar = R.ResampleDataset(tar)
    return srcs, tar, num_cls


# 与"干净版"相关的关键开关：存档配置若与当前臂不一致，必须重训而不是
# 静默复用（历史教训：旧 t-SNE 权重是 w_mmd=0.5 的带对齐版，会被直接跳过）。
CKPT_KEYS = ('w_mmd', 'w_adv', 'tau_mode', 'front_mode', 'front_chan',
             'gabor_alias', 'front_init')


def _is_stale_ckpt(wpath, cfg):
    """返回不一致开关的描述串；一致或无法读取时返回 ''。"""
    try:
        old = torch.load(wpath, map_location='cpu', weights_only=False)
        og = old.get('cfg') or {}
    except Exception:
        return ''
    diffs = []
    for k in CKPT_KEYS:
        if k not in og:
            continue
        a, b = og[k], getattr(cfg, k, None)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            same = abs(float(a) - float(b)) < 1e-9
        else:
            same = (a == b)
        if not same:
            diffs.append('{}={}->{}'.format(k, a, b))
    return ','.join(diffs)


def train_cell(data, task, arm, seed, steps_override, smoke=False):
    tag = '{}_{}_{}_seed{}{}'.format(data, task, arm, seed, '_smoke' if smoke else '')
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    logpath = os.path.join(LOGD, data, 'AGMamba', arm, task, tag + '.log')
    wpath = os.path.join(_P.weights_dir(data, 'AGMamba', arm), tag + '.pt')
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    R.setup_seed(seed)
    srcs, tar, num_cls = load_splits(data, task)
    cfg = build_cfg(data, task, arm, num_cls, len(srcs[0]), steps_override,
                    smoke=smoke)

    if os.path.exists(wpath):
        stale = _is_stale_ckpt(wpath, cfg)
        if stale:
            print('[stale] {} 与当前臂不一致（{}）-> 重新训练'.format(
                wpath, stale), flush=True)
        else:
            print('[skip] {} 已存在'.format(wpath), flush=True)
            return logpath, wpath

    print('[cfg] {} / {} / {} : steps={} sched={} front={} ch={} alias={} '
          'num_cls={} n_src_dom={}'.format(
              data, task, arm, cfg.steps, cfg.use_scheduler, cfg.front_mode,
              cfg.front_chan, cfg.gabor_alias, num_cls, len(srcs)), flush=True)

    model = M.AGMamba(cfg)

    logger = logging.getLogger('tsne_{}'.format(tag))
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(logpath, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    train_loaders = [R.InfiniteDataLoader(s, R.BATCH_SIZE) for s in srcs]
    test_loader = DataLoader(tar, batch_size=R.BATCH_SIZE, shuffle=False,
                             num_workers=0)
    t0 = time.time()
    res = model.train_model(zip(*train_loaders), [test_loader], logger)
    train_s = time.time() - t0

    acc = np.array(res['acces'], dtype=float)[:, 0] * 100.0
    best = float(acc.max())
    last10 = float(acc[-10:].mean())
    print('[done] {} -> last-ten {:.2f} | best {:.2f} | {:.0f}s'.format(
        tag, last10, best, train_s), flush=True)

    torch.save({'model': model.model.state_dict(),
                # 干净版（w_adv = 0）下域判别器不构建 -> 不落盘
                'disc': (model.disc.state_dict()
                         if getattr(model, 'disc', None) is not None else None),
                'cfg': {k: getattr(cfg, k) for k in dir(cfg)
                        if not k.startswith('_') and not callable(getattr(cfg, k))},
                'data': data, 'task': task, 'arm': arm, 'seed': seed,
                'steps': int(cfg.steps), 'last_ten': last10, 'best': best,
                'train_seconds': train_s}, wpath)
    print('[save] {} ({:.1f} MB)'.format(wpath, os.path.getsize(wpath) / 1e6), flush=True)
    return logpath, wpath


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(data, task, arm, seed, n_src=N_SRC_PER_DOMAIN, smoke=False):
    tag = '{}_{}_{}_seed{}{}'.format(data, task, arm, seed, '_smoke' if smoke else '')
    wdir = _P.weights_dir(data, 'AGMamba', arm)
    fpath = os.path.join(wdir, tag + '_feat.npz')
    wpath = os.path.join(wdir, tag + '.pt')

    ckpt = torch.load(wpath, map_location=DEV, weights_only=False)
    srcs, tar, num_cls = load_splits(data, task)
    cfg = build_cfg(data, task, arm, num_cls, len(srcs[0]), int(ckpt['steps']))
    model = M.AGMamba(cfg)
    model.model.load_state_dict(ckpt['model'])
    model.model.to(DEV).eval()

    def fwd(loader):
        Z, Y, P = [], [], []
        for x, y in loader:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            logits, z, _, _ = model.model(x.to(DEV))
            Z.append(z.cpu().numpy().astype(np.float32))
            Y.append(np.asarray(y))
            P.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(Z), np.concatenate(Y), np.concatenate(P)

    Zs, Ys, Ds = [], [], []
    for j, ds in enumerate(srcs):
        idx = list(range(min(n_src, len(ds))))
        ld = DataLoader(Subset(ds, idx), batch_size=256, shuffle=False,
                        num_workers=0)
        z, y, _ = fwd(ld)
        Zs.append(z)
        Ys.append(y)
        Ds.append(np.full(len(y), j))
    Zs, Ys, Ds = np.concatenate(Zs), np.concatenate(Ys), np.concatenate(Ds)

    ld = DataLoader(tar, batch_size=256, shuffle=False, num_workers=0)
    Zt, Yt, Pt = fwd(ld)

    np.savez_compressed(fpath, z_src=Zs, y_src=Ys, d_src=Ds,
                        z_tgt=Zt, y_tgt=Yt, p_tgt=Pt,
                        num_cls=num_cls, last_ten=ckpt['last_ten'],
                        best=ckpt['best'], steps=ckpt['steps'])
    print('[feat] {:<38s} src {} | tgt {} | dim {} | last-ten {:.2f}'.format(
        os.path.basename(fpath), Zs.shape, Zt.shape, Zs.shape[1],
        ckpt['last_ten']), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return fpath


# ---------------------------------------------------------------------------
# 度量 + 出图
# ---------------------------------------------------------------------------
def sep_metrics(Zs, Ys, Zt, Yt, n_neighbors=10):
    """返回 (源/目标线性可分度 %, 目标特征的类别轮廓系数 %)。

    · 域可分度：逻辑回归 5 折交叉验证区分源/目标特征的准确率。
    · 类别轮廓系数：仅在**目标域**特征、以真实类别为标签计算（训练时目标
      标签完全未用，与混淆矩阵同为评估量），越高 = 目标特征自身的类簇越开。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(np.vstack([Zs, Zt]))
    As, At = sc.transform(Zs), sc.transform(Zt)
    X = np.vstack([As, At])
    ydom = np.r_[np.zeros(len(As)), np.ones(len(At))]
    dom = cross_val_score(LogisticRegression(max_iter=1000), X, ydom,
                          cv=5).mean() * 100.0
    sil = silhouette_score(At, Yt) * 100.0
    return dom, sil


def tsne_of(Zs, Zt, seed=0, perplexity=30):
    from sklearn.manifold import TSNE
    X = np.vstack([Zs, Zt])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    emb = TSNE(n_components=2, perplexity=perplexity, init='pca',
               learning_rate='auto', random_state=seed, max_iter=1000).fit_transform(X)
    return emb[:len(Zs)], emb[len(Zs):]


def cap(ax, text, y=-0.20, size=6.4):
    """面板图题置于坐标轴**下方**（与 Fig. 6/8/10 的 cap() 同规则）。"""
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=size, color='0.20', linespacing=1.35)


# 三个列面板的图题（合成图与单面板图共用）。行是「哪条臂」由左侧旋转标签给出。
CAP_COLS = [
    '(a)  joint t-SNE of source and target features\n'
    '      (sources shaded by source condition; target = red triangles)',
    '(b)  the same embedding coloured by class\n'
    '      (source filled, target open)',
    '(c)  target-domain confusion matrix\n'
    '      (counts; rows normalised in the shading)',
]


def draw_domain(ax, a, plt, Line2D):
    """列 (a)：源域（按工况索引着色）+ 目标域（红色三角）联合 t-SNE。"""
    n_dom = int(a['Ds'].max()) + 1
    dcmap = plt.get_cmap('viridis')
    for j in range(n_dom):
        m = a['Ds'] == j
        if m.any():
            ax.scatter(a['ex_s'][m], a['ey_s'][m], s=5,
                       color=dcmap(j / max(n_dom - 1, 1)), alpha=0.5,
                       linewidths=0)
    ax.scatter(a['ex_t'], a['ey_t'], s=13, c='#c00000', marker='^',
               alpha=0.9, linewidths=0)
    handles = [Line2D([], [], marker='o', linestyle='', markersize=4,
                      color=dcmap(j / max(n_dom - 1, 1)),
                      label='src {}'.format(j)) for j in range(n_dom)]
    handles.append(Line2D([], [], marker='^', linestyle='', markersize=5,
                          color='#c00000', label='target'))
    ax.legend(handles=handles, fontsize=6, loc='best', ncol=2,
              framealpha=0.9, handletextpad=0.2, columnspacing=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1', fontsize=8)
    ax.set_ylabel('t-SNE 2', fontsize=8)


def draw_class(ax, a, num_cls, plt):
    """列 (b)：同一嵌入按类别着色（源实心、目标空心）。"""
    cmap = plt.get_cmap('tab10' if num_cls <= 10 else 'tab20')
    cols = [cmap(k % cmap.N) for k in range(num_cls)]
    for k in range(num_cls):
        m = a['Ys'] == k
        if m.any():
            ax.scatter(a['ex_s'][m], a['ey_s'][m], s=4, color=cols[k],
                       alpha=0.35, linewidths=0)
        m = a['Yt'] == k
        if m.any():
            ax.scatter(a['ex_t'][m], a['ey_t'][m], s=18, facecolors='none',
                       edgecolors=cols[k], linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1', fontsize=8)


def draw_cm(ax, a, num_cls, plt):
    """列 (c)：目标域混淆矩阵（计数标注，行归一化用于底色）。"""
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(a['Yt'], a['Pt']):
        cm[t, p] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for t in range(num_cls):
        for p in range(num_cls):
            if cm[t, p]:
                ax.text(p, t, str(cm[t, p]), ha='center', va='center',
                        fontsize=5.5,
                        color='white' if cmn[t, p] > 0.55 else 'black')
    ax.set_xlabel('predicted', fontsize=8)
    ax.set_ylabel('true', fontsize=8)
    ax.set_xticks(range(num_cls)); ax.set_yticks(range(num_cls))
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=6)


def plot_pair(data, task, seed=SEED, tag_prefix='', out_fig=None, single=True):
    """为单个 (data, task) 画 2 行（ours / rand）× 3 列的图，返回数值记录。

    列 = (a) 联合 t-SNE（按源工况着色 + 目标红色三角）、(b) 同类着色、
    (c) 目标域混淆矩阵；行 = 交付配置 / 首层随机核。
    **面板图题一律置于坐标轴下方**（`cap()`），每行左侧旋转标签给出该臂的三个
    标量。另导出 6 张单面板图（a–c = 上行，d–f = 下行）便于排版拼版。
    """
    if out_fig is None:
        out_fig = os.path.join(FIGD, OUTNAME.get((data, task), 'fig11_geom_pu'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    recs, arts = [], {}
    for arm in ('ours', 'rand'):
        f = os.path.join(_P.weights_dir(data, 'AGMamba', arm),
                         '{}_{}_{}_seed{}{}_feat.npz'.format(
                             data, task, arm, seed, tag_prefix))
        if not os.path.exists(f):
            print('[warn] 缺特征文件 {}'.format(f), flush=True)
            return None
        d = np.load(f)
        Zs, Ys, Zt, Yt, Pt = (d['z_src'], d['y_src'], d['z_tgt'],
                              d['y_tgt'], d['p_tgt'])
        Ds = d['d_src']
        acc = float((Pt == Yt).mean() * 100.0)
        dom, knn = sep_metrics(Zs, Ys, Zt, Yt)
        E_s, E_t = tsne_of(Zs, Zt)
        ex_s, ey_s = E_s[:, 0], E_s[:, 1]
        ex_t, ey_t = E_t[:, 0], E_t[:, 1]
        arts[arm] = dict(Zs=Zs, Ys=Ys, Ds=Ds, Zt=Zt, Yt=Yt, Pt=Pt,
                         ex_s=ex_s, ey_s=ey_s, ex_t=ex_t, ey_t=ey_t)
        recs.append(dict(arm=arm, acc=acc, dom=dom, knn=knn,
                         last_ten=float(d['last_ten']), best=float(d['best']),
                         steps=int(d['steps']), n_src=len(Zs), n_tgt=len(Zt)))
        print('[metric] {}/{}/{}  acc {:.2f} | last-ten {:.2f} | best {:.2f} | '
              '域可分 {:.1f}% | 目标类轮廓 {:.1f}'.format(
                  data, task, arm, acc, recs[-1]['last_ten'], recs[-1]['best'],
                  dom, knn), flush=True)

    num_cls = int(np.load(os.path.join(
        _P.weights_dir(data, 'AGMamba', 'ours'),
        '{}_{}_ours_seed{}{}_feat.npz'.format(data, task, seed,
                                              tag_prefix)))['num_cls'])

    # ---- 合成图。图题放下端后必须留白：
    #   H = (top-bottom)/(2+hspace) = (0.925-0.115)/2.30 = 0.352
    #   图题偏移 0.20*H = 0.070 ；行间隙 hspace*H = 0.106 ✓
    #   下排轴底 0.115 -> 图题顶 0.045，两行字高 ~0.026 -> 底 0.019 > 0 ✓
    # ⚠ 宽度取 11.0 in（不是 13.6）：6 面板全宽图缩印到 7.5 in 刊宽后
    #   13.6 in 版的字号只剩 3–4 pt。11.0 in 时约 0.68×，面板接近方形。
    # 左侧 0.062 的栏给旋转行标签；不用 tight_layout（会把下端图题压掉）。
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, left=0.062, right=0.985, top=0.925,
                          bottom=0.115, hspace=0.30, wspace=0.20)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    for r, arm in enumerate(('ours', 'rand')):
        a, rec = arts[arm], recs[r]
        draw_domain(axes[r][0], a, plt, Line2D)
        draw_class(axes[r][1], a, num_cls, plt)
        draw_cm(axes[r][2], a, num_cls, plt)
        p0 = axes[r][0].get_position()
        fig.text(0.009, 0.5 * (p0.y0 + p0.y1),
                 '{}\nacc {:.1f}%  |  silhouette {:.1f}  |  '
                 'separability {:.1f}%'.format(ARM_LABEL[arm], rec['acc'],
                                               rec['knn'], rec['dom']),
                 fontsize=6.2, va='center', ha='left', rotation=90,
                 linespacing=1.7)

    for c, txt in enumerate(CAP_COLS):
        cap(axes[1][c], txt)

    fig.suptitle('t-SNE of the penultimate features and target-domain confusion '
                 'matrices ({} {}, seed {}; {} steps)'.format(
                     data, task, seed, recs[0]['steps']), fontsize=10)
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(out_fig + '.pdf')
    fig.savefig(out_fig + '.png', dpi=200)
    plt.close(fig)
    print('[fig] {}.pdf / .png'.format(out_fig), flush=True)

    # ---- 单面板导出（a/b/c = 上行，d/e/f = 下行；显式 rect + tight bbox）
    if single:
        pref = os.path.basename(out_fig).split('_')[0]      # fig10 / fig11
        RECT = [0.185, 0.30, 0.775, 0.60]
        for r, arm in enumerate(('ours', 'rand')):
            a = arts[arm]
            for c, drawer in enumerate((
                    lambda ax: draw_domain(ax, a, plt, Line2D),
                    lambda ax: draw_class(ax, a, num_cls, plt),
                    lambda ax: draw_cm(ax, a, num_cls, plt))):
                f = plt.figure(figsize=(4.6, 3.9))
                ax = f.add_axes(RECT)
                drawer(ax)
                cap(ax, CAP_COLS[c] + '\n      ' + ARM_LABEL[arm], y=-0.26)
                tag = 'abcdef'[r * 3 + c]
                for ext in ('pdf', 'png'):
                    f.savefig(os.path.join(
                        FIGD, '{}{}_panel.{}'.format(pref, tag, ext)),
                        dpi=400, bbox_inches='tight')
                plt.close(f)

    return recs


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    smoke = (cmd == 'smoke')

    if cmd == 'plot':
        cells = [(d, t) for d, t, _ in CELLS]
    elif cmd == 'run':
        if len(sys.argv) > 3:
            cells = [(sys.argv[2], sys.argv[3])]
            if cells[0] not in [(d, t) for d, t, _ in CELLS]:
                CELLS.append((cells[0][0], cells[0][1], None))
        else:
            cells = []
    else:
        cells = [(d, t) for d, t, _ in CELLS]

    if cmd != 'plot':
        for sdata, stask, steps in CELLS:
            if cmd == 'run' and (sdata, stask) not in cells:
                continue
            for arm in ('ours', 'rand'):
                train_cell(sdata, stask, arm, SEED, steps, smoke=smoke)
                extract_features(sdata, stask, arm, SEED, smoke=smoke)
        if smoke:
            print('[smoke] 跳过出图', flush=True)
            return

    # 出图：每个 (data, task) 一张。图号 2026-10-03 定：§6.8 → Fig. 11（PU T2）/
    # Fig. 12（HUST T9）；文件名语义化，不再用旧的 fig16/fig17。
    os.makedirs(FIGD, exist_ok=True)
    lines = ['# §6.8 特征几何数据（t-SNE + 混淆矩阵，seed {}）'.format(SEED), '',
             '> 协议：§5.2 冻结协议（步数 = 60 × ceil(N_src/64)）。臂 = 交付配置 /',
             '> 首层随机核（唯一变量 = 解析先验）。脚本 `experiments/run_tsne_cm.py`。', '']
    for data, task in cells:
        out = os.path.join(FIGD, OUTNAME.get((data, task),
                                            'fig11_geom_pu'))
        recs = plot_pair(data, task, SEED, out_fig=out)
        if recs is None:
            continue
        no = FIGNO.get((data, task), 10)
        lines.append('## Fig. {} — {} {}'.format(no, data, task))
        lines.append('| 臂 | 步数 | 运行末十 | 运行最优 | 目标精度 | 源/目标可分性 | 目标类轮廓 |')
        lines.append('|---|---|---|---|---|---|---|')
        for r in recs:
            lines.append('| {} | {} | {:.2f} | {:.2f} | {:.1f} | {:.1f} | {:.1f} |'.format(
                r['arm'], r['steps'], r['last_ten'], r['best'], r['acc'],
                r['dom'], r['knn']))
        lines.append('')
    with open(DATAMD, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('[md] {}'.format(DATAMD), flush=True)


if __name__ == '__main__':
    main()
