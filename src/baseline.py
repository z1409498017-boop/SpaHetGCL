"""Baseline 方法：Leiden / 表达+平滑 / PCA+KMeans。"""
import argparse, json, os, sys, time
import numpy as np, anndata as ad, scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.insert(0, '/root/autodl-tmp/spahet/src')
from evaluate import evaluate

P = argparse.ArgumentParser()
P.add_argument('method', choices=['leiden', 'smooth', 'pca'])
P.add_argument('--k', type=int, default=10)
P.add_argument('--knn', type=int, default=15)
P.add_argument('--seed', type=int, default=0)
P.add_argument('--sil_n', type=int, default=20000)
a = P.parse_args()

OUT = f'/root/autodl-tmp/spahet/runs/baseline_{a.method}'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
adata = ad.read_h5ad('/root/autodl-tmp/spahet/data/processed/GSE234713.h5ad')
print(f'[load] {adata.shape} 用时 {time.time()-t0:.0f}s', flush=True)

if a.method == 'leiden':
    # 纯表达图 + Leiden 社区检测
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X', random_state=a.seed, key_added='expr')
    sc.tl.leiden(adata, resolution=1.0, neighbors_key='expr', key_added='leiden', random_state=a.seed)
    # 调 resolution 让 cluster 数接近 k（粗调）
    for res in [0.5, 0.8, 1.2, 1.5, 2.0]:
        sc.tl.leiden(adata, resolution=res, neighbors_key='expr', key_added='tmp', random_state=a.seed)
        if len(adata.obs['tmp'].unique()) >= a.k: break
    Z = adata.obsm['X_expr'].copy()
    
elif a.method == 'smooth':
    # 空间平滑：坐标 kNN 邻居平均表达
    from scipy.spatial import cKDTree
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    Xs = np.zeros_like(X)
    for s in adata.obs['sample'].unique():
        m  = (adata.obs['sample'] == s).values
        p  = adata.obsm['spatial'][m]
        Xm = X[m]
        _, idx = cKDTree(p).query(p, k=a.knn + 1)
        Xs[m] = Xm[idx[:, 1:]].mean(1)
    Z = Xs
    
else:  # pca
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    Z = PCA(128, random_state=a.seed).fit_transform(X)

pos = adata.obsm['spatial']
smp = adata.obs['sample'].to_numpy()
res, lbl = evaluate(Z, pos, smp, a.k, f'baseline_{a.method}')
print(json.dumps(res, indent=1))
json.dump(res, open(f'{OUT}/eval.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/niche.npz', label=lbl, pos=pos, sample=smp)
print(f'用时 {time.time()-t0:.0f}s')
print('BASELINE_DONE_MARKER')
