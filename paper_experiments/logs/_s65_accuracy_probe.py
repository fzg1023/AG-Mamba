#!/usr/bin/python
# -*- coding:utf-8 -*-
"""§6.5 训练侧探针：冻结分析器 vs 可训练分析器 + 受控膨胀的精度阶梯。

回答 §3.2 明确许下的两个假设（原文：Section 6.5 ... separate the two）：

  (i)  学习诱导锐化 —— 若塌陷来自"训练把前端锐化/带上目标偏好"，则
       · 冻结首层（freeze_init=True，只关 conv1 梯度、不动前向）应当**不劣**于
         可训练首层；
       · 训练后核组的有效带宽应**可测地收窄**（ζ / Π 与初始核组相比）。
  (ii) 形态变化 —— 若塌陷来自"梳齿随转速整体缩放"，则对**源域**（已见转速）
       记录施加**同样大小**的精确膨胀（梳齿位置搬到 80 Hz）应当重现塌陷。

三条腿
------
L1 冻结臂精度：freeze_init=True，HUST T9 seed0，与交付配置同种子同数据序；
   与其配对的是同一脚本里跑的可训练臂（两者**初值逐位相同、数据序相同、
   只少一步 conv1 的梯度更新**，故 RNG 中性 —— 见 logs/_check_front_init.py）。
L2 核组审计：训练前后 conv1 的 ζ（过零率）/ Π（谱峰均比），口径与 §4.2 /
   analysis/make_fig2_design_space.py **完全一致**（公式逐行抄自该文件，避免
   import 触发其模块级脚本重画 fig2）；另附相对 L2 变化与逐核余弦。
L3 膨胀阶梯：把**未参与训练**的其它转速域的测试窗按 s = 80/f 精确压缩
   （梳齿搬到与真实目标同一位置），与真实 80 Hz 目标窗比较同一权重下的精度。
   阶梯 {D8 ×1.000, D8 ×1.0667, D7 ×1.000, D7 ×1.1429, D6 ×1.000, D6 ×1.2308,
   D9 真实}，其中 ×1.000 是"已见转速、未见窗口"的天花板，用于把总差距分解为
   **几何差距**（×1.000 vs 真实 80 Hz）与**纯膨胀代价**（×1.000 vs ×s）。

判读
----
· 若 L3 中 ×1.0667 的精度远高于真实 80 Hz 目标 -> 版本 (ii) 的"整体膨胀"
  不足以解释远目标塌陷（真实远目标难在别处：下游对齐/分类，或数据侧的非转速形态差）。
· 若 L1 中冻结臂不劣于可训练臂、且 L2 显示训练后谱形基本不动 -> 版本 (i)
  的"学习诱导锐化"在首层得不到支持。

用法
----
    AGM_FORCE_PY_SCAN=1 python logs/_s65_accuracy_probe.py
输出
----
    logs/_s65_accuracy.md     结果表（正文引用来源）
    logs/_s65_arm_*.log       两条臂的逐步训练日志（同源性自检用）
    logs/_s65_bank.npz        训练前/后 conv1 权重（复核用）
"""
import os
import sys
import time
import logging

# 交付运行的可复现性开关：纯 PyTorch 分块扫描（与主对比/各消融同口径）
os.environ['AGM_FORCE_PY_SCAN'] = '1'

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader, Dataset              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# run_stage.py 内部写 `from run_cell import ...`（阶段脚本本来就以 experiments/ 为
# 工作目录/导入根运行），故这里也要把 experiments/ 放进导入路径
sys.path.insert(0, os.path.join(_ROOT, 'experiments'))

import experiments.run_cell as RC                            # noqa: E402
import experiments.run_stage as RS                           # noqa: E402
import agmamba.method as _M                                  # noqa: E402
from utils.DatasetClass import InfiniteDataLoader            # noqa: E402

OUT_MD = os.path.join(_ROOT, 'logs', '_s65_accuracy.md')
OUT_NPZ = os.path.join(_ROOT, 'logs', '_s65_bank.npz')
REF_LOG = os.path.join(_ROOT, 'results_probe_align', 'HUST', 'AGMamba',
                       'aln_noall', 'T9',
                       'HUST_T9_AGMamba_fc_aln_noall_seed0.log')
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HUST 十个域的转速（Hz），index = 域号
SPEED = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
TARGET_F = 80.0          # T9 的远目标转速
# 膨胀阶梯：(域号, 膨胀因子 s)  s = 80 / f_domain
LADDER = [(8, 1.0), (8, TARGET_F / SPEED[8]),
          (7, 1.0), (7, TARGET_F / SPEED[7]),
          (6, 1.0), (6, TARGET_F / SPEED[6])]

# --------------------------------------------------------------- 度量（口径同 §4.2）
NFFT = 16384             # Π 用（§4.2 / make_fig2_design_space.py 同值）
BAND = 0.45


def zeta_rate(psi):
    """过零率 ζ（跨样本归一）。psi: (C,K) 或 (C,1,K)。公式抄自 fig2 脚本。"""
    a = np.asarray(psi)
    if a.ndim == 3:
        a = a[:, 0, :]
    return float(((a[:, :-1] * a[:, 1:]) < 0).sum(axis=-1).mean()) / (a.shape[-1] - 1)


def peak_to_mean(psi, nfft=NFFT, band=BAND):
    """谱峰均比 Π（功率口径，nfft=16384，带 u ≤ 0.45）。公式抄自 fig2 脚本。"""
    a = np.asarray(psi)
    if a.ndim == 3:
        a = a[:, 0, :]
    sp = np.abs(np.fft.rfft(a, n=nfft, axis=-1)) ** 2
    u = np.fft.rfftfreq(nfft)
    m = (u > 1e-9) & (u <= band)
    sp = sp[:, m]
    return float((sp.max(axis=-1) / np.clip(sp.mean(axis=-1), 1e-30,
                                            None)).mean())


def bank_stats(w):
    """w: (C,1,K) -> ζ、Π、相对 L2 变化（对参照）、逐核余弦均值。"""
    a = np.asarray(w, dtype=np.float64)
    if a.ndim == 3:
        a = a[:, 0, :]
    return zeta_rate(a), peak_to_mean(a), a


# --------------------------------------------------------------- 受控膨胀
class CompressView(Dataset):
    """把 (1,1024) 窗按 s 做**时间压缩**（s>1 ⇒ 梳齿频率 ×s）。

    口径：y[i] = x(i·s)。索引 i·s 超出窗尾时用**镜像填充**补足（缺口仅占窗
    尾 s−1 比例，HUST 的冲击瞬态在窗首，窗尾为噪声；s ≤ 1.23 时缺口 ≤ 19%）。
    s = 1.0 为恒等（用于验证机制本身不引入失真）。
    """

    def __init__(self, ds, s):
        self.ds = ds
        self.s = float(s)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.s == 1.0:
            return x, y
        a = np.asarray(x, dtype=np.float64)
        L = a.shape[-1]
        need = int(np.ceil((L - 1) * (self.s - 1.0))) + 2
        a = np.concatenate([a, a[..., -need:][..., ::-1]], axis=-1)
        pos = np.arange(L, dtype=np.float64) * self.s
        j0 = np.floor(pos).astype(np.int64)
        frac = pos - j0
        y2 = a[..., j0] * (1.0 - frac) + a[..., j0 + 1] * frac
        return torch.from_numpy(y2.astype(np.float32)), y


def load_target(dom):
    """取**域 dom 的测试（tail）划分**：把 dom 设为唯一目标域。

    源域列表任取其余 9 个（不影响目标划分的确定性），标签空间与其他任务相同。
    """
    task = [[d for d in range(10) if d != dom], [dom]]
    _, tar, _ = RC.load_data('HUST', task)
    return tar


# --------------------------------------------------------------- 训练一条臂
def eval_sets(model, sets):
    """用同一份权重逐个评估（返回精度 %，顺序同 sets）。"""
    model.eval()
    outs = []
    for _, ds, _, _ in sets:
        yp, yt = [], []
        for x, y in DataLoader(ds, batch_size=RC.BATCH_SIZE, shuffle=False,
                               num_workers=0):
            pred = model.predict(x.to(DEVICE))
            yp.extend(pred.detach().cpu().numpy().tolist())
            yt.extend(y.numpy().tolist())
        outs.append(100.0 * float(np.mean(np.asarray(yp) == np.asarray(yt))))
    model.train()
    return outs


def train_arm(tag, freeze, train=True, task='T9', with_ladder=True,
              seed=SEED):
    RC.setup_seed(seed)
    tgt_dom = int(RC.TASKS['HUST'][task][1][0])
    srcs, tar, num_cls = RC.load_data('HUST', RC.TASKS['HUST'][task])
    srcs = [RC.ResampleDataset(s) for s in srcs]
    tar = RC.ResampleDataset(tar)

    iters_per_epoch = max(1, (len(srcs[0]) + RC.BATCH_SIZE - 1) // RC.BATCH_SIZE)
    steps = RC.EPOCHS * iters_per_epoch
    checkpoint_freq = iters_per_epoch

    _M.configs = RS._delivered_configs()          # 交付配置（= pub 臂同源）
    _M.configs.freeze_init = bool(freeze)
    cfg = RC.make_config(_M, 'AGMamba', num_cls, steps, checkpoint_freq,
                         DEVICE, n_src=len(srcs))
    cfg.n_domains = len(srcs)
    model = getattr(_M, RC.METHOD_MODELS['AGMamba'])(cfg)
    model.to(DEVICE)

    print('[%s] steps=%d ckpt=%d freeze_init=%s front_init=%s init_mode=%s '
          'conv1.requires_grad=%s' % (
              tag, steps, checkpoint_freq, freeze, cfg.front_init,
              model.model.front.init_mode,
              model.model.front.conv1.weight.requires_grad), flush=True)
    w_init = model.model.front.conv1.weight.detach().cpu().clone()

    # ---- 受控膨胀评估集（仅 T9：阶梯以 80 Hz 为共同梳齿位置）
    sets = [('D%d(real)' % tgt_dom, tar, tgt_dom, 1.0)]
    if with_ladder:
        for dom, s in LADDER:
            ds = RC.ResampleDataset(load_target(dom))
            name = 'D%d' % dom + ('' if s == 1.0 else 'x%.4f' % s)
            sets.append((name, CompressView(ds, s), dom, s))
    names = [v[0] for v in sets]
    print('[%s/%s] eval sets: %s' % (tag, task, names), flush=True)

    logpath = os.path.join(_ROOT, 'logs', '_s65_arm_%s%s.log'
                           % (tag, '' if task == 'T9' else '_' + task))
    if not train:
        # 初始（未训练）对照：权重 = 交付初始化，不经任何更新
        lad = eval_sets(model, sets)
        return dict(tag=tag, freeze=freeze, model=model, logpath=None,
                    w_init=w_init, w_final=w_init.clone(), acc=None,
                    names=names, lad=np.array(lad))

    # ---- 逐步日志（与交付路径同格式；test_loaders 只含真实目标 ⇒ 轨迹逐位同源）
    logger = logging.getLogger('s65_' + tag)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(logpath, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    train_loaders = [InfiniteDataLoader(s, RC.BATCH_SIZE) for s in srcs]
    train_iter = zip(*train_loaders)
    test_loader = DataLoader(tar, batch_size=RC.BATCH_SIZE, shuffle=False,
                             num_workers=0)
    t0 = time.time()
    result = model.train_model(train_iter, [test_loader], logger)
    print('[%s] trained in %.0fs' % (tag, time.time() - t0), flush=True)

    acc = np.asarray(result['acces'], dtype=float) * 100.0   # (n_ckpt, 1)
    lad = np.array(eval_sets(model, sets))                   # 最终权重下的阶梯
    return dict(tag=tag, freeze=freeze, model=model, logpath=logpath,
                w_init=w_init,
                w_final=model.model.front.conv1.weight.detach().cpu().clone(),
                acc=acc, names=names, lad=lad)


def last5(acc):
    return float(np.mean(acc[-5:])), float(acc[-5:].min()), float(acc[-5:].max())


def ref_points(path):
    """参照日志的逐步 tgt_acc（%，仅取真实目标那一列）。"""
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            m = ln.strip()
            if m.startswith('tgt_acc:'):
                pts.append(float(m.split(':')[1]) * 100.0)
    return pts


def main():
    t_all = time.time()
    lines = []
    lines.append('# §6.5 训练侧探针（冻结分析器 + 受控膨胀精度阶梯）')
    lines.append('')
    lines.append('生成脚本：`logs/_s65_accuracy_probe.py` ｜ HUST T9 ｜ seed %d '
                 '｜ 交付配置（480 步 / 余弦 / gaborwide 512 alias2.0）｜ '
                 '`AGM_FORCE_PY_SCAN=1`' % SEED)
    lines.append('')

    arms = [train_arm('init', False, train=False),
            train_arm('trained', False),
            train_arm('frozen', True)]
    A = {a['tag']: a for a in arms}
    REAL = [n for n in A['trained']['names'] if n.endswith('(real)')][0]

    # ---- 初值一致性：两条训练臂的 conv1 初值必须逐位相同
    d0 = float((A['trained']['w_init'] - A['frozen']['w_init']).abs().max())
    lines.append('## 0. 起飞自检')
    lines.append('')
    lines.append('- 两条训练臂 conv1 初值 max|Δ| = %.9f -> %s'
                 % (d0, '一致（只差梯度更新）' if d0 == 0 else '不一致✗'))
    lines.append('- 初始臂核组与训练臂初值 max|Δ| = %.9f'
                 % float((A['init']['w_init'] - A['trained']['w_init'])
                         .abs().max()))
    lines.append('- 冻结臂 conv1 是否真冻结：`requires_grad=%s`'
                 % A['frozen']['model'].model.front.conv1.weight.requires_grad)
    lines.append('')

    # ---- 同源性自检：trained 臂的逐步 tgt_acc 与交付参照日志逐点比对
    rp = ref_points(REF_LOG)
    mine = A['trained']['acc'][:, 0]
    if len(rp) == len(mine):
        dd = float(np.max(np.abs(np.array(rp) - mine)))
        lines.append('- 同源自检：trained 臂 %d 个 checkpoint 与参照 '
                     '`aln_noall`(≡ pub) seed0 逐点 max|Δ| = %.6f -> %s'
                     % (len(rp), dd, '同源' if dd < 1e-6 else '不同源✗'))
    else:
        lines.append('- 同源自检：参考日志点数 %d != 本次 %d，无法逐点比对'
                     % (len(rp), len(mine)))
    lines.append('')

    # ---- L1 冻结臂
    lines.append('## L1 冻结分析器 vs 可训练分析器（真实 80 Hz 目标）')
    lines.append('')
    lines.append('| 臂 | 训练期 best | 训练期 last5 | 末权重 post-hoc |')
    lines.append('| --- | --- | --- | --- |')
    for tag in ('init', 'trained', 'frozen'):
        a = A[tag]
        j = a['names'].index(REAL)
        if a['acc'] is None:
            lines.append('| %s | — | — | %.2f |' % (tag, a['lad'][j]))
        else:
            lines.append('| %s | %.2f | %.2f | %.2f |'
                         % (tag, float(a['acc'][:, 0].max()),
                            float(np.mean(a['acc'][-5:, 0])), a['lad'][j]))
    lines.append('')
    lines.append('- 冻结 − 可训练（同口径 last5）= %+.2f（§5.3 复现下限 mean|Δ|=2.55 / '
                 'max 6.00 为保守上界）'
                 % (float(np.mean(A['frozen']['acc'][-5:, 0]))
                    - float(np.mean(A['trained']['acc'][-5:, 0]))))
    lines.append('')

    # ---- L2 核组审计
    zi, pi, ai = bank_stats(A['trained']['w_init'].numpy())
    zt, pt, at = bank_stats(A['trained']['w_final'].numpy())
    num = float(np.linalg.norm(at - ai))
    den = float(np.linalg.norm(ai))
    cos = np.sum(ai * at, axis=-1) / np.clip(
        np.linalg.norm(ai, axis=-1) * np.linalg.norm(at, axis=-1), 1e-30, None)
    lines.append('## L2 核组审计：训练前后（T9 seed%d 的可训练臂）' % SEED)
    lines.append('')
    lines.append('| 量 | 初始核组 | 训练后 | 变化 |')
    lines.append('| --- | --- | --- | --- |')
    lines.append('| ζ 过零率 | %.4f | %.4f | %+.4f |' % (zi, zt, zt - zi))
    lines.append('| Π 谱峰均比 | %.3f | %.3f | %+.3f |' % (pi, pt, pt - pi))
    lines.append('| ‖ΔW‖/‖W‖ | — | %.4f | — |' % (num / den))
    lines.append('| 逐核 cos(W_init, W_trained) 均值 | — | %.4f | — |'
                 % float(np.mean(cos)))
    lines.append('')
    lines.append('- 冻结臂核组 = 初始核组（逐位）：max|Δ| = %.9f'
                 % float((A['frozen']['w_final']
                          - A['frozen']['w_init']).abs().max()))
    lines.append('')

    # ---- L3 膨胀阶梯
    nm = A['trained']['names']
    lines.append('## L3 受控膨胀阶梯（末权重 post-hoc，单种子）')
    lines.append('')
    lines.append('| 评估集 | 梳齿位置 | init | trained | frozen |')
    lines.append('| --- | --- | --- | --- | --- |')
    for j, n in enumerate(nm):
        if n.endswith('(real)'):
            comb = '%.0f Hz（真实记录）' % SPEED[int(n[1:n.index('(')])]
        else:
            dom = int(n[1:n.index('x')]) if 'x' in n else int(n[1:])
            s = 1.0 if 'x' not in n else float(n[n.index('x') + 1:])
            comb = ('%.1f Hz（未膨胀）' % SPEED[dom] if s == 1.0
                    else '%.1f Hz ×%.4f → 80 Hz' % (SPEED[dom], s))
        lines.append('| %s | %s | %.2f | %.2f | %.2f |'
                     % (n, comb, A['init']['lad'][j], A['trained']['lad'][j],
                        A['frozen']['lad'][j]))
    lines.append('')

    lines.append('可训练臂上的分解（正数 = 变差）：')
    lines.append('')
    for dom in (8, 7, 6):
        s = TARGET_F / SPEED[dom]
        rawn = 'D%d' % dom
        tagn = 'D%dx%.4f' % (dom, s)
        for tag in ('trained', 'frozen'):
            v = A[tag]['lad']
            real = v[nm.index(REAL)]
            lines.append('- [%s] D%d（%.0f Hz）：未膨胀 %.2f -> 膨胀到 80 Hz %.2f '
                         '（纯膨胀代价 %+.2f）｜真实 80 Hz %.2f ｜几何差距 %+.2f'
                         % (tag, dom, SPEED[dom], v[nm.index(rawn)],
                            v[nm.index(tagn)], v[nm.index(tagn)] - v[nm.index(rawn)],
                            real, real - v[nm.index(tagn)]))
    lines.append('')

    # ---- 产物
    np.savez(OUT_NPZ,
             w_init=A['trained']['w_init'].numpy(),
             w_trained=A['trained']['w_final'].numpy(),
             w_frozen=A['frozen']['w_final'].numpy(),
             acc_trained=A['trained']['acc'],
             acc_frozen=A['frozen']['acc'],
             names=np.array(nm),
             ladder_acc=np.stack([A[t]['lad'] for t in
                                  ('init', 'trained', 'frozen')]),
             ladder_names=np.array(['init', 'trained', 'frozen']),
             ladder=np.array([(d, s) for d, s in LADDER], dtype=float))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines).encode('ascii', 'replace').decode(), flush=True)
    print('[done] %.0fs -> %s' % (time.time() - t_all, OUT_MD), flush=True)


def main_extra():
    """补跑外推剖面上的"冻结 vs 可训练"对照（T0 = 20 Hz、T4 = 40 Hz，seed 0）。

    目的：假设 (i) 预测"冻结分析器不塔陷或塔陷更弱"——单看远目标只能得到一个点，
    看**整个外推剖面**才能判断塔陷是否真的被“防住”。每格的预算、配置、种子
    与 T9 臂完全一致（同一 train_arm 路径）。
    """
    lines = ['# §6.5 冻结分析器：外推剖面（T0/T4，seed %d）' % SEED, '']
    lines.append('| 目标 | 臂 | best | last5 | 末权重 post-hoc |')
    lines.append('| --- | --- | --- | --- | --- |')
    res = {}
    for task in ('T0', 'T4'):
        for frozen in (False, True):
            tag = '%s_%s' % (task, 'frozen' if frozen else 'trained')
            a = train_arm(tag, frozen, task=task, with_ladder=False)
            j = 0
            lines.append('| %s | %s | %.2f | %.2f | %.2f |'
                         % (task, 'frozen' if frozen else 'trained',
                            float(a['acc'][:, 0].max()),
                            float(np.mean(a['acc'][-5:, 0])), a['lad'][j]))
            res[(task, frozen)] = (float(a['acc'][:, 0].max()),
                                   float(np.mean(a['acc'][-5:, 0])),
                                   a['lad'][j])
            print('RESULT %s %s' % (tag, res[(task, frozen)]), flush=True)
    lines.append('')
    for task in ('T0', 'T4'):
        t, f = res[(task, False)], res[(task, True)]
        lines.append('- %s：冻结 − 可训练（末权重 post-hoc）= %+.2f'
                     % (task, f[2] - t[2]))
    with open(os.path.join(_ROOT, 'logs', '_s65_frozen_targets.md'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    (main_extra if 'extra' in sys.argv else main)()
