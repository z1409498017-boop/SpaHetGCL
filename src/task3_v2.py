"""Task3 v2：巨噬-成纤维空间邻接（两种零模型 + HC/UC/CD 差分）。

零模型对比：
  global  全样本均匀置换标签 —— 破坏所有空间结构；细胞类型各自聚集时必然 fold<1
  block   200um 网格块内置换 —— 保留局部组成与大尺度结构，回答"接触是否超出局部预期"
差分检验：以 block fold 为指标比较 HC / UC / CD（n=3/组，仅记录趋势）。
"""
import argparse, glob, json, os, time
import numpy as np, torch
from scipy.stats import ttest_ind, mannwhitneyu

P = argparse.ArgumentParser()
P.add_argument('--nperm', type=int, default=1000)
P.add_argument('--block', type=float, default=200.0, help='置换分块边长 (um)')
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
print(f'零模型: global(均匀置换) vs block({a.block:.0f}um 块内置换), nperm={a.nperm}\n')
print(f'{"sample":8}{"obs":>7}{"fold_glob":>11}{"fold_blk":>10}{"z_blk":>9}{"p_blk":>9}{"blocks":>8}')
print('-' * 64)

for f in sorted(glob.glob(f'{GD}/*.pt')):
    g = torch.load(f, weights_only=False)
    s, cond = g.sample_id, g.condition

    # 统一编号：细类型 + 坐标(um)
    off, ct_l, pos_l, N = {}, [], [], 0
    for t in g.node_types:
        off[t] = N; N += g[t].x.size(0)
        ct_l.append(np.asarray(g[t].celltype, dtype=object))
        pos_l.append(g[t].pos.numpy())
    ct  = np.concatenate(ct_l)
    pos = np.vstack(pos_l)

    # near 边 -> 无向去重
    E = []
    for r in g.edge_types:
        if not r[1].startswith('near_'): continue
        ei = g[r].edge_index.numpy()
        E.append(np.vstack([ei[0] + off[r[0]], ei[1] + off[r[2]]]))
    E = np.hstack(E)
    u, v = E[0], E[1]
    k = u < v
    u, v = u[k], v[k]

    is_m = np.isin(ct, MAC)
    is_f = np.isin(ct, FIB)
    cnt  = lambda mm, ff: int(((mm[u] & ff[v]) | (ff[u] & mm[v])).sum())
    obs  = cnt(is_m, is_f)

    # --- 零模型1：全局均匀置换 ---
    null_g = np.empty(a.nperm, np.int64)
    for i in range(a.nperm):
        p = rng.permutation(N)
        null_g[i] = cnt(is_m[p], is_f[p])

    # --- 零模型2：空间分块块内置换（向量化）---
    bx  = np.floor(pos[:, 0] / a.block).astype(np.int64)
    by  = np.floor(pos[:, 1] / a.block).astype(np.int64)
    _, blk = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    base   = np.argsort(blk, kind='stable')          # 按块分组的位置
    nblk   = int(blk.max()) + 1

    null_b = np.empty(a.nperm, np.int64)
    pm = np.empty(N, bool); pf = np.empty(N, bool)
    for i in range(a.nperm):
        shuf = np.lexsort((rng.random(N), blk))      # 块内随机序
        pm[base] = is_m[shuf]                        # 标签只在块内流动
        pf[base] = is_f[shuf]
        null_b[i] = cnt(pm, pf)

    mg, sg = null_g.mean(), null_g.std() + 1e-9
    mb, sb = null_b.mean(), null_b.std() + 1e-9
    pb = (null_b >= obs).sum() / a.nperm
    rows.append({'sample': s, 'condition': cond, 'n_cells': N,
                 'n_near_edges': int(len(u)), 'n_mac': int(is_m.sum()), 'n_fib': int(is_f.sum()),
                 'observed': obs, 'n_blocks': nblk,
                 'null_global_mean': float(mg), 'fold_global': float(obs / mg),
                 'null_block_mean': float(mb), 'null_block_sd': float(sb),
                 'fold_block': float(obs / mb), 'z_block': float((obs - mb) / sb),
                 'p_block': float(max(pb, 1.0 / a.nperm)), 'p_block_is_bound': bool(pb == 0)})
    print(f'{s:8}{obs:7d}{obs/mg:11.3f}{obs/mb:10.3f}{(obs-mb)/sb:9.2f}'
          f'{max(pb,1/a.nperm):9.4f}{nblk:8d}', flush=True)

os.makedirs(RUN, exist_ok=True)
json.dump({'per_sample': rows, 'nperm': a.nperm, 'block_um': a.block,
           'macrophage_types': MAC, 'fibroblast_types': FIB},
          open(f'{RUN}/task3_v2.json', 'w'), indent=1)

print(f'\n{"="*64}\n=== 分组汇总 ===')
G = {c: [r for r in rows if r['condition'] == c] for c in ['HC', 'UC', 'CD']}
for c, rs in G.items():
    fg = [r['fold_global'] for r in rs]; fb = [r['fold_block'] for r in rs]
    print(f'  {c} (n={len(rs)}): fold_global = {np.mean(fg):.3f}±{np.std(fg):.3f}   '
          f'fold_block = {np.mean(fb):.3f}±{np.std(fb):.3f}')

print(f'\n=== 差分检验（fold_block）===')
tests = {}
for c1, c2 in [('HC', 'UC'), ('HC', 'CD'), ('UC', 'CD')]:
    x = [r['fold_block'] for r in G[c1]]; y = [r['fold_block'] for r in G[c2]]
    t, p = ttest_ind(x, y)
    try:    _, pu = mannwhitneyu(x, y, alternative='two-sided')
    except Exception: pu = float('nan')
    tests[f'{c1}_vs_{c2}'] = {'t': float(t), 'p_ttest': float(p), 'p_mwu': float(pu),
                              'mean_diff': float(np.mean(y) - np.mean(x))}
    print(f'  {c1} vs {c2}: delta={np.mean(y)-np.mean(x):+.3f}  t={t:+.2f}  '
          f'p_t={p:.3f}  p_mwu={pu:.3f}')
print(f'\n  注：n=3/组，统计效力极低，仅记录趋势，不作显著性结论。')

json.dump(tests, open(f'{RUN}/task3_v2_tests.json', 'w'), indent=1)
print(f'\n用时 {time.time()-t0:.0f}s')
print('TASK3V2_DONE_MARKER')
