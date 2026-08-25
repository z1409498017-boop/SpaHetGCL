"""Baseline 对比（自包含，不 import evaluate.py 以避免其模块级 argparse 副作用）。

method:
  leiden    表达 PCA + Leiden 社区检测（无空间信息）
  pca       表达 PCA(128) + KMeans
  smooth    空间 kNN 平滑表达 + KMeans（经典空间平滑思路）
  ctcomp    邻域细胞类型组成向量 + KMeans（经典 niche 方法思路，CellCharter 类似核心）
指标与 SpaHetGCL 评估完全一致：spatial_coherence / chaos_um / morans_I / silhouette。
"""
import argparse, json, os, time
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

P = argparse.ArgumentParser()
P.add_argument('method', choices=['leiden', 'pca', 'smooth', 'ctcomp'])
P.add_argument('--k', type=int, default=10)
P.add_argument('--knn', type=int, default=15)
P.add_argument('--seed', type=int, default=0)
P.add_argument('--sil_n', type=int, default=20000)
a = P.parse_args()

H5  = '/root/autodl-tmp/spahet/data/processed/GSE234713.h5ad'
OUT = f'/root/autodl-tmp/spahet/runs/baseline_{a.method}'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(a.seed)


def spatial_coherence(lbl, pos, smp, knn):
    out = []
    for s in np.unique(smp):
        m = smp == s
        _, idx = cKDTree(pos[m]).query(pos[m], k=knn + 1)
        l = lbl[m]
        out.append((l[idx[:, 1:]] == l[:, None]).mean())
    return float(np.mean(out))


def chaos(lbl, pos, smp):
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
    out = []
    for s in np.unique(smp):
        m = smp == s
        p, v = pos[m], V[m]
        _, idx = cKDTree(p).query(p, k=knn + 1)
        idx = idx[:, 1:]
        vc  = v - v.mean(0)
        num = np.zeros(v.shape[1], np.float64)
        for i in range(0, len(vc), chunk):
            sl = slice(i, min(i + chunk, len(vc)))
            num += (vc[sl, None, :] * vc[idx[sl]]).sum(1).sum(0)
        den = (vc ** 2).sum(0)
        out.append(np.mean(num / (knn * den + 1e-12)))
    return float(np.mean(out))


t0 = time.time()
adata = ad.read_h5ad(H5)
X   = adata.X.toarray().astype(np.float32) if hasattr(adata.X, 'toarray') else np.asarray(adata.X, np.float32)
pos = adata.obsm['spatial'].astype(np.float64)
smp = adata.obs['sample'].to_numpy()
print(f'[load] X={X.shape} 用时 {time.time()-t0:.0f}s', flush=True)

# ---------------- 表征 + 聚类 ----------------
if a.method == 'leiden':
    import scanpy as sc
    Z = PCA(50, random_state=a.seed).fit_transform(X)      # 表达已归一化，不再 log1p
    ah = ad.AnnData(X=np.zeros((len(Z), 1), np.float32))
    ah.obsm['X_pca'] = Z
    sc.pp.neighbors(ah, n_neighbors=a.knn, use_rep='X_pca', random_state=a.seed)
    # 粗调 resolution 使簇数接近 k，保证与其它方法可比
    best = None
    for res in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        sc.tl.leiden(ah, resolution=res, key_added='l', random_state=a.seed,
                     flavor='igraph', n_iterations=2, directed=False)
        nc = ah.obs['l'].nunique()
        print(f'  res={res}: {nc} clusters', flush=True)
        if best is None or abs(nc - a.k) < abs(best[1] - a.k):
            best = (res, nc, ah.obs['l'].astype(int).to_numpy())
        if nc >= a.k: break
    res, nclu, lbl = best
    note = f'leiden resolution={res}'
else:
    if a.method == 'pca':
        Z = PCA(128, random_state=a.seed).fit_transform(X)
        note = 'PCA(128)+KMeans'
    elif a.method == 'smooth':
        Z = np.zeros_like(X)
        for s in np.unique(smp):
            m = smp == s
            _, idx = cKDTree(pos[m]).query(pos[m], k=a.knn + 1)
            Z[m] = X[m][idx[:, 1:]].mean(1)               # 邻域平均表达
        note = f'spatial kNN(k={a.knn}) smoothing + KMeans'
    else:  # ctcomp：邻域细胞类型组成
        ct  = adata.obs['celltype'].astype(str).to_numpy()
        code, uct = pd.factorize(ct)
        Z = np.zeros((len(ct), len(uct)), np.float32)
        for s in np.unique(smp):
            m  = np.where(smp == s)[0]
            _, idx = cKDTree(pos[m]).query(pos[m], k=a.knn + 1)
            nb = code[m][idx[:, 1:]]                       # 邻居类型编码
            oh = np.zeros((len(m), len(uct)), np.float32)
            for j in range(nb.shape[1]):
                oh[np.arange(len(m)), nb[:, j]] += 1
            Z[m] = oh / a.knn
        note = f'neighborhood celltype composition(k={a.knn}, {len(uct)} types) + KMeans'
    Zs  = StandardScaler().fit_transform(Z)
    lbl = KMeans(a.k, n_init=10, random_state=a.seed).fit_predict(Zs)
    nclu = a.k

# ---------------- 指标 ----------------
Zs  = StandardScaler().fit_transform(Z)
sub = rng.choice(len(Zs), min(a.sil_n, len(Zs)), replace=False)
res_d = {'method': f'baseline_{a.method}', 'note': note, 'k': int(nclu),
         'spatial_coherence': spatial_coherence(lbl, pos, smp, a.knn),
         'chaos_um':          chaos(lbl, pos, smp),
         'morans_I':          morans_I(Z, pos, smp, a.knn),
         'silhouette':        float(silhouette_score(Zs[sub], lbl[sub])),
         'n_cells':           int(len(Zs)),
         'repr_dim':          int(Z.shape[1]),
         'wall_sec':          round(time.time() - t0, 1)}
print(json.dumps(res_d, indent=1))
json.dump(res_d, open(f'{OUT}/eval.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/niche.npz', label=lbl, pos=pos, sample=smp)
print(f'用时 {time.time()-t0:.0f}s')
print('BASELINE_DONE_MARKER')
