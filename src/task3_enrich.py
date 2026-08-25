"""Task3：巨噬细胞-成纤维细胞空间邻接富集（解释性验证，不做机制推断）。

统计口径：观测邻接边数 vs 标签置换零分布（保持图结构、样本内置换细胞类型标签）。
"""
import argparse, glob, json, os, time
import numpy as np, torch

P = argparse.ArgumentParser()
P.add_argument('--nperm', type=int, default=1000)
P.add_argument('--seed', type=int, default=0)
P.add_argument('--tag', default='main')
a = P.parse_args()

GD  = '/root/autodl-tmp/spahet/data/graphs'
RUN = f'/root/autodl-tmp/spahet/runs/{a.tag}'
MAC = ['M0', 'M1', 'M2', 'Macrophage NRG1', 'Inflammatory monocytes']
FIB = ['Fibroblasts', 'Inflammatory fibroblasts', 'Myofibroblasts', 'FRCs']
rng = np.random.default_rng(a.seed)
t0  = time.time()

rows = []
for f in sorted(glob.glob(f'{GD}/*.pt')):
    g = torch.load(f, weights_only=False)
    s, cond = g.sample_id, g.condition

    # 汇总为样本内统一编号：细类型数组 + 无向 near 边表
    off, ct_all, N = {}, [], 0
    for t in g.node_types:
        off[t] = N; N += g[t].x.size(0); ct_all.append(np.asarray(g[t].celltype, dtype=object))
    ct = np.concatenate(ct_all)

    E = []
    for r in g.edge_types:
        if not r[1].startswith('near_'): continue
        ei = g[r].edge_index.numpy()
        E.append(np.vstack([ei[0] + off[r[0]], ei[1] + off[r[2]]]))
    E = np.hstack(E)
    u, v = E[0], E[1]
    k = u < v                                    # 双向 -> 无向去重
    u, v = u[k], v[k]

    is_m = np.isin(ct, MAC); is_f = np.isin(ct, FIB)
    def count(mm, ff):
        return int(((mm[u] & ff[v]) | (ff[u] & mm[v])).sum())
    obs = count(is_m, is_f)

    # 置换零分布：样本内打乱细胞类型标签，保持图结构与各类型细胞数
    null = np.empty(a.nperm, np.int64)
    for i in range(a.nperm):
        perm = rng.permutation(N)
        null[i] = count(is_m[perm], is_f[perm])
    mu, sd = null.mean(), null.std() + 1e-9
    p_emp = (null >= obs).sum() / a.nperm        # 单侧经验 p
    rows.append({'sample': s, 'condition': cond, 'n_cells': N, 'n_near_edges': int(len(u)),
                 'n_mac': int(is_m.sum()), 'n_fib': int(is_f.sum()),
                 'observed': obs, 'null_mean': float(mu), 'null_sd': float(sd),
                 'fold_enrichment': float(obs / mu) if mu > 0 else None,
                 'z_score': float((obs - mu) / sd),
                 'p_perm': float(max(p_emp, 1.0 / a.nperm)),
                 'p_is_bound': bool(p_emp == 0)})
    print(f'  {s} ({cond}): obs={obs:6d} null={mu:9.1f}±{sd:6.1f} '
          f'fold={obs/mu:5.3f} z={(obs-mu)/sd:8.2f} p{"<" if p_emp==0 else "="}'
          f'{max(p_emp,1/a.nperm):.4f}', flush=True)

os.makedirs(RUN, exist_ok=True)
json.dump({'per_sample': rows, 'nperm': a.nperm,
           'macrophage_types': MAC, 'fibroblast_types': FIB},
          open(f'{RUN}/task3_enrich.json', 'w'), indent=1)

print(f'\n{"="*70}\n=== 按分组汇总 (fold enrichment) ===')
for c in ['HC', 'UC', 'CD']:
    v = [r['fold_enrichment'] for r in rows if r['condition'] == c]
    z = [r['z_score'] for r in rows if r['condition'] == c]
    if v: print(f'  {c}: fold = {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})   '
                f'z = {np.mean(z):.1f}')
print(f'\n所有样本 fold 均值 = {np.mean([r["fold_enrichment"] for r in rows]):.3f}')
print(f'置换检验次数 = {a.nperm}（p 下界 = {1/a.nperm}）')
print(f'用时 {time.time()-t0:.0f}s')
print('TASK3_DONE_MARKER')
