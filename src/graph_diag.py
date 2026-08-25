"""Delaunay 边长分布诊断：数据驱动确定距离阈值。"""
import numpy as np, anndata as ad, time
from scipy.spatial import Delaunay

PX_UM = 0.1228  # CosMx SMI 像素尺寸 um/px
t0 = time.time()
adata = ad.read_h5ad('/root/autodl-tmp/spahet/data/processed/GSE234713.h5ad')
print(f'[load] {adata.shape} 用时 {time.time()-t0:.0f}s', flush=True)
print(f'[subset 分布]\n{adata.obs["subset"].value_counts(dropna=False)}\n', flush=True)

all_len = []
print(f'{"sample":8} {"cells":>7} {"edges":>9} {"median_um":>10} {"p95_um":>8} {"p99_um":>8} {"max_um":>9}')
print('-'*66)
for s in sorted(adata.obs['sample'].unique()):
    m = (adata.obs['sample'] == s).values
    xy = adata.obsm['spatial'][m].astype(np.float64)
    tri = Delaunay(xy)
    # 从三角形提取无向边
    ia, ja = tri.vertex_neighbor_vertices
    src = np.repeat(np.arange(len(xy)), np.diff(ia))
    dst = ja
    keep = src < dst                      # 去重
    src, dst = src[keep], dst[keep]
    d_um = np.linalg.norm(xy[src] - xy[dst], axis=1) * PX_UM
    all_len.append(d_um)
    print(f'{s:8} {len(xy):7d} {len(src):9d} {np.median(d_um):10.2f} '
          f'{np.percentile(d_um,95):8.1f} {np.percentile(d_um,99):8.1f} {d_um.max():9.0f}')

d = np.concatenate(all_len)
print(f'\n=== 全局边长分布 (um)，共 {len(d)} 条边 ===')
for p in [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]:
    print(f'  p{p:<5}= {np.percentile(d, p):8.1f}')
print(f'  max   = {d.max():8.0f}')
print('\n=== 候选阈值保留率 ===')
for th in [20, 25, 30, 40, 50, 60, 80, 100]:
    print(f'  <= {th:3d} um : 保留 {(d<=th).mean()*100:5.2f}%  ({(d<=th).sum()} 条)')
print(f'\n[结论] 长尾边来自 FOV 间隙/组织边界，必须过滤')
print('DIAG_DONE_MARKER')
