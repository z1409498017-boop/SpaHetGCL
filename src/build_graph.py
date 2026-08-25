"""按样本构建空间异质图 -> PyG HeteroData。

节点类型 = subset (5 类)；关系 = (src_type, bin, dst_type)，bin ∈ {near<=10um, far<=30um}。
阈值由 graph_diag.py 的边长分布确定：中位数 8.5um，30um 保留 96.9%。
"""
import os, time, pickle
import numpy as np, anndata as ad, torch
from scipy.spatial import Delaunay
from torch_geometric.data import HeteroData

PX_UM  = 0.1228          # CosMx SMI 像素尺寸
D_MAX  = 30.0            # 邻接上限 (um)
D_NEAR = 10.0            # 近/远分界 (um)
NT     = ['epi', 'stroma', 'myeloids', 'tcells', 'plasmas']
MORPH  = ['Area', 'AspectRatio', 'Mean.PanCK', 'Mean.CD45', 'Mean.CD3', 'Mean.DAPI']
OUT    = '/root/autodl-tmp/spahet/data/graphs'
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
adata = ad.read_h5ad('/root/autodl-tmp/spahet/data/processed/GSE234713.h5ad')
genes = adata.var_names.to_list()
print(f'[load] {adata.shape} 用时 {time.time()-t0:.0f}s', flush=True)

# 形态特征全局标准化（表达已由原作者归一化，此处不再动）
Mo = adata.obs[MORPH].to_numpy(np.float32)
Mo = np.log1p(np.clip(Mo, 0, None))
Mo = (Mo - Mo.mean(0)) / (Mo.std(0) + 1e-8)

stats, rel_tot = [], {}
for s in sorted(adata.obs['sample'].unique()):
    m   = (adata.obs['sample'] == s).values
    sub = adata[m]
    xy  = sub.obsm['spatial'].astype(np.float64)
    Xs  = sub.X.toarray().astype(np.float32) if hasattr(sub.X, 'toarray') else np.asarray(sub.X, np.float32)
    Mos = Mo[m]
    nts = sub.obs['subset'].to_numpy()
    cts = sub.obs['celltype'].astype(str).to_numpy()

    # --- Delaunay -> 去重 -> 距离过滤 ---
    tri = Delaunay(xy)
    ia, ja = tri.vertex_neighbor_vertices
    src = np.repeat(np.arange(len(xy)), np.diff(ia)); dst = ja
    k = src < dst; src, dst = src[k], dst[k]
    d = np.linalg.norm(xy[src] - xy[dst], axis=1) * PX_UM
    k = d <= D_MAX; src, dst, d = src[k], dst[k], d[k]

    # --- 全局 idx -> (类型, 类型内局部 idx) ---
    loc = np.full(len(xy), -1, np.int64); tid = np.full(len(xy), -1, np.int64)
    data = HeteroData()
    for ti, t in enumerate(NT):
        mt = np.where(nts == t)[0]
        loc[mt] = np.arange(len(mt)); tid[mt] = ti
        data[t].x        = torch.from_numpy(np.hstack([Xs[mt], Mos[mt]]))
        data[t].pos      = torch.from_numpy((xy[mt] * PX_UM).astype(np.float32))
        data[t].celltype = cts[mt]
        data[t].gidx     = torch.from_numpy(mt)          # 回溯到样本内全局序号
    assert (tid >= 0).all(), '存在未分配类型的细胞'

    # --- 关系分桶：(src_type, bin, dst_type)，双向 ---
    nbin = (d <= D_NEAR).astype(np.int64)               # 0=near, 1=far
    bn   = ['near', 'far']
    nrel = 0
    for a, b in [(src, dst), (dst, src)]:               # 两个方向都建
        for ti, t1 in enumerate(NT):
            for tj, t2 in enumerate(NT):
                for bi in (0, 1):
                    k2 = (tid[a] == ti) & (tid[b] == tj) & (nbin == (1 - bi))
                    if not k2.any(): continue
                    rel = (t1, f'{bn[bi]}_{t1[:3]}_{t2[:3]}', t2)
                    ei  = torch.from_numpy(np.vstack([loc[a[k2]], loc[b[k2]]]))
                    ew  = torch.from_numpy(d[k2].astype(np.float32))
                    if rel in data.edge_types:          # 合并同名关系
                        data[rel].edge_index = torch.cat([data[rel].edge_index, ei], 1)
                        data[rel].edge_attr  = torch.cat([data[rel].edge_attr, ew])
                    else:
                        data[rel].edge_index, data[rel].edge_attr = ei, ew
                        nrel += 1
    for r in data.edge_types:
        rel_tot[r] = rel_tot.get(r, 0) + data[r].edge_index.shape[1]

    data.sample_id = s
    data.condition = s.split('_')[0]
    torch.save(data, f'{OUT}/{s}.pt')
    ne = sum(data[r].edge_index.shape[1] for r in data.edge_types)
    stats.append((s, len(xy), ne, nrel))
    print(f'  {s}: nodes={len(xy):6d} edges={ne:7d} relations={nrel:3d} '
          f'near={100*(d<=D_NEAR).mean():.1f}%', flush=True)

print(f'\n{"="*62}\n每样本汇总')
print(f'{"sample":8}{"nodes":>8}{"edges":>9}{"rels":>6}')
for s, n, e, r in stats: print(f'{s:8}{n:8d}{e:9d}{r:6d}')
print(f'{"TOTAL":8}{sum(x[1] for x in stats):8d}{sum(x[2] for x in stats):9d}')
print(f'\n关系种类总数 = {len(rel_tot)}')
print(f'\n--- 边数最多的 12 种关系 ---')
for r, c in sorted(rel_tot.items(), key=lambda x: -x[1])[:12]:
    print(f'  {str(r):58} {c:8d}')
print(f'\n--- 髓系<->基质 关系（Task3 关注）---')
for r, c in sorted(rel_tot.items(), key=lambda x: -x[1]):
    if {'myeloids', 'stroma'} == {r[0], r[2]}: print(f'  {str(r):58} {c:8d}')
print(f'\n特征维度 = {len(genes)} 基因 + {len(MORPH)} 形态 = {len(genes)+len(MORPH)}')
print(f'输出目录 {OUT}  总用时 {time.time()-t0:.0f}s')
print('GRAPH_DONE_MARKER')
