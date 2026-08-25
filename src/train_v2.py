"""SpaHetGCL v2：邻居对比学习（neighbor-positive contrastive）。

动机（由消融数据推出）：
  标准 InfoNCE 是实例判别 —— 把每个细胞视为独立类别互相推开，
  与「同一生态位的细胞应当相似」直接冲突。这解释了为何加入对比分支后
  silhouette 从 0.426(仅重建) 掉到 0.120(main)/0.155(仅对比)。
修法：
  正样本由「同细胞的另一增强视图」改为「空间近邻」，让目标函数直接编码生态位假设。
  --nbr_mode nbr   仅邻居正样本
  --nbr_mode both  邻居 + 同细胞双视图（各 0.5 权重）
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn.functional as F
from model import SpaHetGCL, mask_features, drop_edges, recon_loss, nt_xent

P = argparse.ArgumentParser()
P.add_argument('--epochs', type=int, default=100)
P.add_argument('--hidden', type=int, default=128)
P.add_argument('--layers', type=int, default=3)
P.add_argument('--heads', type=int, default=4)
P.add_argument('--lr', type=float, default=1e-3)
P.add_argument('--mask_ratio', type=float, default=0.3)
P.add_argument('--edge_drop', type=float, default=0.2)
P.add_argument('--tau', type=float, default=0.5)
P.add_argument('--n_sample', type=int, default=4096)
P.add_argument('--seed', type=int, default=0)
P.add_argument('--tag', type=str, default='v2_nbr')
P.add_argument('--nbr_mode', choices=['nbr', 'both'], default='nbr')
P.add_argument('--use_recon', action='store_true', help='默认关闭（消融证明重建分支有害）')
a = P.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
GD  = '/root/autodl-tmp/spahet/data/graphs'
OUT = f'/root/autodl-tmp/spahet/runs/{a.tag}'
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
graphs = []
for f in sorted(glob.glob(f'{GD}/*.pt')):
    graphs.append(torch.load(f, weights_only=False))
meta   = graphs[0].metadata()
NT     = meta[0]
in_dim = graphs[0][NT[0]].x.size(1)

# 预计算：统一索引空间下的 near 边（与 torch.cat([h[t] for t in NT]) 的顺序一致）
near_ei = []
for g in graphs:
    offs, c = {}, 0
    for t in NT:
        offs[t] = c; c += g[t].x.size(0)
    E = [np.vstack([g[r].edge_index[0].numpy() + offs[r[0]],
                    g[r].edge_index[1].numpy() + offs[r[2]]])
         for r in g.edge_types if r[1].startswith('near_')]
    near_ei.append(torch.from_numpy(np.hstack(E)).long())
    print(f'  {g.sample_id}: {c} nodes, {near_ei[-1].size(1)} near-edges', flush=True)
print(f'[data] {len(graphs)} 图, in_dim={in_dim}, 用时 {time.time()-t0:.0f}s', flush=True)

model = SpaHetGCL(in_dim, meta, a.hidden, a.heads, a.layers).to(dev)
opt   = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
nparam = sum(p.numel() for p in model.parameters())
print(f'[model] 参数量 = {nparam/1e6:.2f}M  nbr_mode={a.nbr_mode}  '
      f'use_recon={a.use_recon}  tau={a.tau}', flush=True)
gen = torch.Generator(device=dev); gen.manual_seed(a.seed)


def nbr_infonce(z, ei, tau, n_sample, gen):
    """邻居正样本 InfoNCE：随机采 near 边，(src,dst) 为正对，批内其余 dst 为负。

    注：同一节点可能被多次采到，形成少量假负样本；
    n_sample=4096 相对边数 ~15-45万，碰撞概率低，此处不做去重。
    """
    m = ei.size(1)
    k = torch.randint(0, m, (min(n_sample, m),), device=ei.device, generator=gen)
    s, d = ei[0, k], ei[1, k]
    zs = F.normalize(z[s], dim=1)
    zd = F.normalize(z[d], dim=1)
    sim = zs @ zd.t() / tau
    lbl = torch.arange(zs.size(0), device=z.device)
    return 0.5 * (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.t(), lbl))


hist = []
for ep in range(1, a.epochs + 1):
    model.train(); agg = {'loss': 0., 'rec': 0., 'nbr': 0., 'ins': 0.}
    for gi in np.random.permutation(len(graphs)):
        g  = graphs[gi]
        ei = near_ei[gi].to(dev, non_blocking=True)
        x_d   = {t: g[t].x.to(dev, non_blocking=True) for t in g.node_types}
        pos_d = {t: g[t].pos.to(dev, non_blocking=True) for t in g.node_types}
        e_d   = {r: g[r].edge_index.to(dev, non_blocking=True) for r in g.edge_types}
        opt.zero_grad(set_to_none=True)

        xm, mk = mask_features(x_d, a.mask_ratio, gen)
        h1 = model.encode(xm, pos_d, e_d)
        z1 = model.head(torch.cat([h1[t] for t in NT], 0))

        L_rec = recon_loss({t: model.dec[t](h1[t]) for t in h1}, x_d, mk) \
                if a.use_recon else torch.zeros((), device=dev)
        L_nbr = nbr_infonce(z1, ei, a.tau, a.n_sample, gen)

        L_ins = torch.zeros((), device=dev)
        if a.nbr_mode == 'both':
            xm2, _ = mask_features(x_d, a.mask_ratio, gen)
            h2 = model.encode(xm2, pos_d, drop_edges(e_d, a.edge_drop, gen))
            z2 = model.head(torch.cat([h2[t] for t in NT], 0))
            L_ins = nt_xent(z1, z2, a.tau, a.n_sample, gen)
            loss  = L_rec + 0.5 * L_nbr + 0.5 * L_ins
        else:
            loss  = L_rec + L_nbr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        agg['loss'] += loss.item(); agg['rec'] += float(L_rec)
        agg['nbr']  += float(L_nbr); agg['ins'] += float(L_ins)
    sched.step()
    n = len(graphs)
    r = {k: v / n for k, v in agg.items()}; r['epoch'] = ep
    r['lr'] = opt.param_groups[0]['lr']; hist.append(r)
    if ep % 5 == 0 or ep == 1:
        print(f'  ep{ep:4d} loss={r["loss"]:.4f} nbr={r["nbr"]:.4f} '
              f'ins={r["ins"]:.4f} rec={r["rec"]:.4f} lr={r["lr"]:.2e} '
              f'gpu={torch.cuda.max_memory_allocated()/2**30:.1f}G '
              f'[{time.time()-t0:.0f}s]', flush=True)

torch.save({'model': model.state_dict(), 'args': vars(a), 'in_dim': in_dim}, f'{OUT}/ckpt.pt')

model.eval()
with torch.no_grad():
    for g in graphs:
        h = model.encode({t: g[t].x.to(dev) for t in g.node_types},
                         {t: g[t].pos.to(dev) for t in g.node_types},
                         {r: g[r].edge_index.to(dev) for r in g.edge_types})
        np.savez_compressed(f'{OUT}/emb_{g.sample_id}.npz',
            **{f'z_{t}': h[t].cpu().numpy() for t in g.node_types},
            **{f'gidx_{t}': g[t].gidx.numpy() for t in g.node_types},
            **{f'ct_{t}': g[t].celltype.astype(str) for t in g.node_types},
            **{f'pos_{t}': g[t].pos.numpy() for t in g.node_types})

json.dump({'history': hist, 'args': vars(a), 'nparam': nparam,
           'wall_sec': time.time() - t0}, open(f'{OUT}/history.json', 'w'), indent=1)
print(f'\n最终 loss={hist[-1]["loss"]:.4f} (nbr={hist[-1]["nbr"]:.4f}, '
      f'ins={hist[-1]["ins"]:.4f}, rec={hist[-1]["rec"]:.4f})')
print(f'峰值显存 = {torch.cuda.max_memory_allocated()/2**30:.2f} GB')
print(f'总用时 {time.time()-t0:.0f}s -> {OUT}')
print('TRAINV2_DONE_MARKER')
