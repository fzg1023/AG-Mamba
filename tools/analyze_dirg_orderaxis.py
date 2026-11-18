"""DIRG 转速轴失效机理诊断：① 圈数归一 ② 带通阶梯 ③ 谱白化去共振

只读 `datasets/dirg_data/dirg_x1024_ch0.npz`，**不训练、不占 GPU**。
口径与 `tools/analyze_dirg_bandladder.py` 对齐（LDA 代理、128x200Hz 带、7 类）。

代理读数：same7 / cross7 / cross3(部位) / cross2(有故障 vs 健康)，单位 %。
随机线 7 类 = 14.29。

用法: python tools/analyze_dirg_orderaxis.py            (固定载荷 L1000)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

NPZ = os.path.join(_ROOT, 'datasets', 'dirg_data', 'dirg_x1024_ch0.npz')
SPEEDS = [100, 200, 300, 400, 500]
CLASSES = ['0A', '1A', '2A', '3A', '4A', '5A', '6A']
GROUP_OF = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}   # 0 healthy / 1 inner / 2 roller
FS = 51200.0
NWIN = 100          # 每类每速取多少个 1024 点窗
LOAD_TAG = 'L1000'


# ----------------------------------------------------------------- 特征
def band_edges(edge_hz):
    return np.arange(0.0, FS / 2 + 1e-9, edge_hz)


def feat_db(win, edges):
    """逐窗 z-score -> 功率谱 -> 10log10 -> 按 edges 分带平均。"""
    x = np.asarray(win, dtype=np.float64).reshape(-1)
    s = x.std()
    if s < 1e-12:
        return None
    x = (x - x.mean()) / s
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / FS)
    idx = np.clip(np.digitize(f, edges) - 1, 0, len(edges) - 2)
    out = np.full(len(edges) - 1, -200.0)
    for k in range(len(edges) - 1):
        m = idx == k
        if m.any():
            out[k] = 10.0 * np.log10(P[m].mean() + 1e-30)
    return out


# ----------------------------------------------------------------- 代理
def proxy(X, y, spid, seeds=3):
    """返回 (same7, cross7, cross3, cross2)。"""
    y = np.asarray(y)
    spid = np.asarray(spid)
    g = np.array([GROUP_OF[a] for a in y])
    b = (y > 0).astype(int)
    res = []
    for lab in (y, g, b):
        ncls = len(np.unique(lab))
        same, cross = [], []
        for s in np.unique(spid):
            m = spid == s
            # same-domain: 5-fold CV（用 LDA 自带即可）
            acc = []
            for seed in range(seeds):
                rng = np.random.RandomState(seed)
                order = rng.permutation(m.sum())
                Xs, ys = X[m][order], lab[m][order]
                k = 5
                folds = np.array_split(np.arange(len(ys)), k)
                hit = tot = 0
                for i in range(k):
                    te = folds[i]
                    tr = np.concatenate([folds[j] for j in range(k) if j != i])
                    clf = LDA().fit(Xs[tr], ys[tr])
                    hit += (clf.predict(Xs[te]) == ys[te]).sum()
                    tot += len(te)
                acc.append(hit / tot)
            same.append(float(np.mean(acc)))
            # cross-domain: 其余转速训练 -> 该转速
            tr = ~m
            clf = LDA().fit(X[tr], lab[tr])
            cross.append(float((clf.predict(X[m]) == lab[m]).mean()))
        res.append((float(np.mean(same)), float(np.mean(cross))))
    return res[0] + res[1] + res[2]


# ----------------------------------------------------------------- 数据
def load_records(nwin=NWIN):
    z = np.load(NPZ)
    recs, ys, sp = {}, [], []
    for s in SPEEDS:
        for ci, c in enumerate(CLASSES):
            key = '%d_%s_%s' % (s, c, LOAD_TAG)
            if key not in z.files:
                raise SystemExit('缺 %s' % key)
            recs[(s, ci)] = np.asarray(z[key], dtype=np.float64)   # (500,1024)
    for s in SPEEDS:
        for ci in range(len(CLASSES)):
            ys.append(ci)
            sp.append(s)
    return recs, ys, sp


def windows_raw(recs, nwin=NWIN):
    X, y, sp = [], [], []
    for (s, ci), arr in sorted(recs.items()):
        for w in arr[:nwin]:
            d = feat_db(w, band_edges(200.0))
            if d is not None:
                X.append(d); y.append(ci); sp.append(s)
    return np.array(X), np.array(y), np.array(sp)


# ----------------------------------------------------------------- 变换
def bandpass_record(record, lo, hi):
    """在整段连续记录上做零相位 rFFT 掩码带通（升余弦过渡）。"""
    x = np.asarray(record, dtype=np.float64).reshape(-1)
    n = x.size
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    m = np.zeros_like(f)
    if lo > 0:
        t = (f > lo * 0.85) & (f < lo)
        m[t] = 0.5 * (1 - np.cos(np.pi * (f[t] - lo * 0.85) / (lo * 0.15)))
    else:
        m[f < hi * 0.85] = 1.0
        t = (f >= hi * 0.85) & (f < hi)
        m[t] = 0.5 * (1 + np.cos(np.pi * (f[t] - hi * 0.85) / (hi * 0.15)))
        m[f >= hi] = 0.0
        y = np.fft.irfft(np.fft.rfft(x) * m, n=n)
        return y
    m[(f >= lo) & (f < hi * 0.85)] = 1.0
    t2 = (f >= hi * 0.85) & (f < hi)
    m[t2] = 0.5 * (1 + np.cos(np.pi * (f[t2] - hi * 0.85) / (hi * 0.15)))
    m[f >= hi] = 0.0
    y = np.fft.irfft(np.fft.rfft(x) * m, n=n)
    return y


def order_norm_record(record, speed, revs=4.0, out=1024, nmax=NWIN):
    """圈数归一：每窗固定 revs 圈 -> 重采样到 out 点。"""
    x = np.asarray(record, dtype=np.float64).reshape(-1)
    npts = int(round(revs * FS / float(speed)))
    if npts < 64:
        return None
    nb = min(nmax, len(x) // npts)
    if nb < 5:
        return None
    src = np.arange(npts)
    dst = np.linspace(0, npts - 1, out)
    ws = []
    for b in range(nb):
        seg = x[b * npts:(b + 1) * npts]
        ws.append(np.interp(dst, src, seg))
    return np.array(ws)


def ref_envelope(recs):
    """跨全部 (速度,类) 的中位对数幅度谱 -> 固定共振 + 1/f 参考包络。"""
    per = []
    for arr in recs.values():
        P = np.mean([np.abs(np.fft.rfft(a)) ** 2 for a in arr[:min(50, len(arr))]], axis=0)
        per.append(P)
    P = np.median(np.array(per), axis=0)
    return P


def whiten_windows(arr, refP, nwin=NWIN):
    """逐窗：X' = X * sqrt(refP / ownP)（频域增益）-> 时域。"""
    own = np.mean([np.abs(np.fft.rfft(a)) ** 2 for a in arr[:min(50, len(arr))]], axis=0)
    own = np.maximum(own, 1e-12)
    gain = 1.0 / np.sqrt(np.maximum(refP, 1e-12))   # remove common carrier, keep class-relative shape
    out = []
    for a in arr[:nwin]:
        X = np.fft.rfft(a) * gain
        out.append(np.fft.irfft(X, n=a.size))
    return np.array(out)


def feats_from_windows(ws):
    X, y, sp = [], [], []
    for (s, ci), wlist in ws.items():
        if wlist is None:
            continue
        for w in wlist:
            d = feat_db(w, band_edges(200.0))
            if d is not None:
                X.append(d); y.append(ci); sp.append(s)
    return np.array(X), np.array(y), np.array(sp)


# ----------------------------------------------------------------- main
def main():
    revs_win = {s: s * 1024.0 / FS for s in SPEEDS}
    print('=' * 78)
    print('DIRG 转速轴诊断  载荷=%s  每类每速 %d 窗  特征=128x200Hz 带  随机线7类=14.29'
          % (LOAD_TAG, NWIN))
    print('窗内圈数（1024 点 @51200）: ' +
          '  '.join('%dHz=%.1f圈' % (s, revs_win[s]) for s in SPEEDS))
    print('=' * 78)

    recs, _, _ = load_records()
    refP = ref_envelope(recs)

    def show(name, X, y, sp):
        r = proxy(X, y, sp)
        print('%-34s | same7 %6.2f | cross7 %6.2f | cross3 %6.2f | cross2 %6.2f'
              % (name, r[0], r[1], r[3], r[5]))
        return r

    print('\n【基线】原始 1024 点窗')
    X, y, sp = windows_raw(recs)
    base = show('  raw', X, y, sp)

    print('\n【① 圈数归一】每窗固定 4 圈 -> 重采样到 1024 点')
    ws = {}
    for k, arr in recs.items():
        w = order_norm_record(arr, k[0], revs=4.0)
        ws[k] = w
    X1, y1, sp1 = feats_from_windows(ws)
    m1 = show('  order-norm (4 rev)', X1, y1, sp1)

    print('\n【② 带通阶梯】单带独立 LODO（整段记录零相位带通）')
    bands = [(0, 1600), (1600, 3200), (3200, 6400), (6400, 12800), (12800, 25600)]
    bres = {}
    for lo, hi in bands:
        Xb, yb, spb = windows_raw({k: bandpass_record(v, lo, hi).reshape(-1, 1024)
                                   for k, v in recs.items()})
        r = show('  band %5.1f-%-6.1f kHz' % (lo / 1000, hi / 1000), Xb, yb, spb)
        bres[(lo, hi)] = r

    print('\n【③ 谱白化去共振】逐窗频域增益 sqrt(跨记录中位包络/自身包络)')
    ws3 = {k: whiten_windows(v, refP) for k, v in recs.items()}
    X3, y3, sp3 = feats_from_windows(ws3)
    r3 = show('  whiten', X3, y3, sp3)

    print('\n【③+① 白化 + 圈数归一】')
    ws4 = {}
    for k, v in recs.items():
        w2 = whiten_windows(v, refP)
        cont = w2.reshape(-1)
        ws4[k] = order_norm_record(cont, k[0], revs=4.0)
    X4, y4, sp4 = feats_from_windows(ws4)
    r4 = show('  whiten + order-norm', X4, y4, sp4)

    print('\n' + '=' * 78)
    print('读数汇总（cross7 是"跨转速 7 类"主指标；base=%.2f）' % base[1])
    print('  ① 圈数归一        : %.2f  (%+.2f)' % (m1[1], m1[1] - base[1]))
    for lo, hi in bands:
        print('  ② 带通 %5.1f-%-5.1f kHz: %.2f  (%+.2f)'
              % (lo / 1000, hi / 1000, bres[(lo, hi)][1], bres[(lo, hi)][1] - base[1]))
    print('  ③ 白化            : %.2f  (%+.2f)' % (r3[1], r3[1] - base[1]))
    print('  ③+① 白化+圈数归一 : %.2f  (%+.2f)' % (r4[1], r4[1] - base[1]))
    print('=' * 78)


if __name__ == '__main__':
    main()
