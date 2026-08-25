#!/usr/bin/env python3
"""Baseline: Leiden 聚类（纯表达，无空间信息）"""
import scanpy as sc, numpy as np, json, time

adata = sc.read_h5ad("/root/autodl-tmp/adata.h5ad")
print(f"[data] {adata.n_obs} cells, {adata.n_vars} genes")

t0 = time.time()
# Leiden on raw normalized counts (X already log1p normalized)
sc.pp.highly_variable_genes(adata, n_top_genes=500, subset=False, flavor='seurat_v3', layer=None)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
dt = time.time() - t0

# 计算空间连贯性
from scipy.spatial import cKDTree
coords = adata.obsm["spatial"]
labels = adata.obs["leiden"].astype(int).values
spatial_coherence = []
for sample in adata.obs["sample"].unique():
    mask = (adata.obs["sample"] == sample).values
    if mask.sum() < 10: continue
    tree = cKDTree(coords[mask])
    nn_idx = tree.query(coords[mask], k=6)[1][:, 1:]
    neighbor_labels = labels[mask][nn_idx]
    same_frac = (neighbor_labels == labels[mask][:, None]).mean()
    spatial_coherence.append(same_frac)

result = {
    "method": "Leiden",
    "n_clusters": int(adata.obs["leiden"].nunique()),
    "spatial_coherence": float(np.mean(spatial_coherence)),
    "time_sec": round(dt, 1)
}
print(json.dumps(result, indent=2))
with open("/root/autodl-tmp/baseline_leiden.json", "w") as f:
    json.dump(result, f, indent=2)
print("BASELINE_LEIDEN_DONE")
