"""生态位聚类评估。

本数据集无病理学家标注的空间域 ground truth，故不报 ARI/NMI（需真实标签）。
改用无监督空间指标：空间连贯性、CHAOS、Moran's I、Silhouette。
"""
import argparse, glob, json, os, time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

P = argparse.ArgumentParser()
P.add_argument('--tag', default='main')
P.add_argument('--k', type=int, default=10, help='niche 数')
P.add_argument('--knn', type=int, default=15, help='空间邻居数')
P.add_argument('--seed', type=int, default=0)
P.add_argument('--sil_n', type=int, default=20000)
a = P.parse_args()

RUN = f'/root/autodl-tmp/spahet/runs/{a.tag}'
NT  = ['epi', 'stroma', 'myeloids', 'tcells', 'plasmas']
rng = np.random.default_rng(a.seed)


def load(tag):
    """拼接所有样本的 embedding / 坐标 / 类型。"""
    Z, POS, CT, SMP = [], [], [], []
    for f in sorted(glob.glob(f'/root/autodl-tmp/spahet/runs/{tag}/emb_*.npz')):
        d = np.load(f, allow_pickle=True)
        s = os.path.basename(f)[4:-4]
        for t in NT:
            if f'z_{t}' not in d: continue
            Z.append(d[f'z_{t}']); POS.append(d[f'pos_{t}'])
            CT.append(d[f'ct_{t}']); SMP += [s] * len(d[f'z_{t}'])
    return (np.vstack(Z), np.vstack(POS), np.concatenate(CT), np.array(SMP))


def spatial_coherence(lbl, pos, smp, knn):
    """niche 标签与空间近邻的一致率（每样本内计算）。"""
    out = []
    for s in np.unique(smp):
        m = smp == s
        tree = cKDTree(pos[m]); l = lbl[m]
        _, idx = tree.query(pos[m], k=knn + 1)
        out.append((l[idx[:, 1:]] == l[:, None]).mean())
    return float(np.mean(out))


def chaos(lbl, pos, smp):
    """每个 niche 内部到最近同标签点的平均距离（越小越空间紧致）。"""
    vals = []
    for s in np.unique(smp):
        m = smp == s
        for c in np.unique(lbl[m]):
            p = pos[m][lbl[m] == c]
            if len(p) < 5: continue
            d, _ = cKDTree(p).query(p, k=2)
            vals.append(d[:, 1].mean())
    return float(np.mean(vals))


def morans_I(V, pos, smp, knn, chunk=20000):
    """embedding 各维度的平均 Moran's I（分块避免高维时中间张量爆内存）。"""
    out = []
    for s in np.unique(smp):
        m = smp == s
        p, v = pos[m], V[m]
        _, idx = cKDTree(p).query(p, k=knn + 1)
        idx = idx[:, 1:]
        vc  = v - v.mean(0)
        num = np.zeros(v.shape[1], dtype=np.float64)
        for i in range(0, len(vc), chunk):
            sl = slice(i, min(i + chunk, len(vc)))
            num += (vc[sl, None, :] * vc[idx[sl]]).sum(1).sum(0)
        den = (vc ** 2).sum(0)
        out.append(np.mean(num / (knn * den + 1e-12)))
    return float(np.mean(out))


def evaluate(Z, pos, smp, k, name):
    Zs  = StandardScaler().fit_transform(Z)
    lbl = KMeans(k, n_init=10, random_state=a.seed).fit_predict(Zs)
    sub = rng.choice(len(Zs), min(a.sil_n, len(Zs)), replace=False)
    r = {'method': name, 'k': k,
         'spatial_coherence': spatial_coherence(lbl, pos, smp, a.knn),
         'chaos_um':          chaos(lbl, pos, smp),
         'morans_I':          morans_I(Z, pos, smp, a.knn),
         'silhouette':        float(silhouette_score(Zs[sub], lbl[sub])),
         'n_cells':           int(len(Zs))}
    return r, lbl


if __name__ == '__main__':
    t0 = time.time()
    Z, pos, ct, smp = load(a.tag)
    print(f'[load] Z={Z.shape} 样本={len(np.unique(smp))} 用时 {time.time()-t0:.0f}s', flush=True)
    res, lbl = evaluate(Z, pos, smp, a.k, f'SpaHetGCL(tag={a.tag})')
    print(json.dumps(res, indent=1))
    np.savez_compressed(f'{RUN}/niche.npz', label=lbl, pos=pos, celltype=ct, sample=smp)
    json.dump(res, open(f'{RUN}/eval.json', 'w'), indent=1)

    # niche 组成（用于 Task3 解释）
    print(f'\n--- 各 niche 的 celltype 富集 (top5) ---')
    for c in range(a.k):
        m = lbl == c
        u, n = np.unique(ct[m], return_counts=True)
        top = ', '.join(f'{x}:{y*100/m.sum():.0f}%' for x, y in
                        sorted(zip(u, n), key=lambda z: -z[1])[:5])
        print(f'  niche{c:2d} (n={m.sum():6d}, {m.mean()*100:4.1f}%): {top}')
    print(f'\n用时 {time.time()-t0:.0f}s')
    print('EVAL_DONE_MARKER')
