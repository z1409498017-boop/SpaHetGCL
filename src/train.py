"""SpaHetGCL 自监督预训练：掩码重建 + 双视图对比。"""
import argparse, glob, json, os, time
import numpy as np, torch
from model import SpaHetGCL, mask_features, drop_edges, recon_loss, nt_xent

P = argparse.ArgumentParser()
P.add_argument('--epochs', type=int, default=100)
P.add_argument('--hidden', type=int, default=128)
P.add_argument('--layers', type=int, default=3)
P.add_argument('--heads', type=int, default=4)
P.add_argument('--lr', type=float, default=1e-3)
P.add_argument('--mask_ratio', type=float, default=0.3)
P.add_argument('--edge_drop', type=float, default=0.2)
P.add_argument('--w_contrast', type=float, default=1.0)
P.add_argument('--tau', type=float, default=0.5)
P.add_argument('--seed', type=int, default=0)
P.add_argument('--tag', type=str, default='main')
P.add_argument('--no_contrast', action='store_true')
P.add_argument('--no_recon', action='store_true')
P.add_argument('--no_spatial_pe', action='store_true')
a = P.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
GD  = '/root/autodl-tmp/spahet/data/graphs'
OUT = f'/root/autodl-tmp/spahet/runs/{a.tag}'
os.makedirs(OUT, exist_ok=True)

if a.no_spatial_pe:                     # 消融：关掉空间位置编码
    import model as M
    M.spatial_pe = lambda pos, dim, scale=100.0: torch.zeros(pos.size(0), dim, device=pos.device)

t0 = time.time()
graphs = []
for f in sorted(glob.glob(f'{GD}/*.pt')):
    g = torch.load(f, weights_only=False)          # HeteroData 需关闭 weights_only
    graphs.append(g)
    print(f'  loaded {g.sample_id}: {sum(g[t].x.size(0) for t in g.node_types)} nodes, '
          f'{len(g.edge_types)} rels', flush=True)
meta   = graphs[0].metadata()
in_dim = graphs[0][meta[0][0]].x.size(1)
print(f'[data] {len(graphs)} 图, in_dim={in_dim}, {len(meta[0])} 节点类型, '
      f'{len(meta[1])} 关系, 用时 {time.time()-t0:.0f}s', flush=True)

model = SpaHetGCL(in_dim, meta, a.hidden, a.heads, a.layers).to(dev)
opt   = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
nparam = sum(p.numel() for p in model.parameters())
print(f'[model] 参数量 = {nparam/1e6:.2f}M', flush=True)
gen = torch.Generator(device=dev); gen.manual_seed(a.seed)

hist = []
for ep in range(1, a.epochs + 1):
    model.train(); agg = {'loss': 0., 'rec': 0., 'con': 0.}
    order = np.random.permutation(len(graphs))
    for gi in order:
        g = graphs[gi]
        x_d   = {t: g[t].x.to(dev, non_blocking=True) for t in g.node_types}
        pos_d = {t: g[t].pos.to(dev, non_blocking=True) for t in g.node_types}
        e_d   = {r: g[r].edge_index.to(dev, non_blocking=True) for r in g.edge_types}
        opt.zero_grad(set_to_none=True)

        # --- 视图1：掩码重建 ---
        xm, mk = mask_features(x_d, a.mask_ratio, gen)
        h1 = model.encode(xm, pos_d, e_d)
        L_rec = torch.zeros((), device=dev)
        if not a.no_recon:
            L_rec = recon_loss({t: model.dec[t](h1[t]) for t in h1}, x_d, mk)

        # --- 视图2：不同掩码 + 边dropout，做对比 ---
        L_con = torch.zeros((), device=dev)
        if not a.no_contrast:
            xm2, _ = mask_features(x_d, a.mask_ratio, gen)
            h2 = model.encode(xm2, pos_d, drop_edges(e_d, a.edge_drop, gen))
            z1 = model.head(torch.cat([h1[t] for t in meta[0]], 0))
            z2 = model.head(torch.cat([h2[t] for t in meta[0]], 0))
            L_con = nt_xent(z1, z2, a.tau, gen=gen)

        loss = L_rec + a.w_contrast * L_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        agg['loss'] += loss.item(); agg['rec'] += float(L_rec); agg['con'] += float(L_con)
    sched.step()
    n = len(graphs)
    rec = {k: v / n for k, v in agg.items()}; rec['epoch'] = ep
    rec['lr'] = opt.param_groups[0]['lr']; hist.append(rec)
    if ep % 5 == 0 or ep == 1:
        print(f'  ep{ep:4d} loss={rec["loss"]:.4f} recon={rec["rec"]:.4f} '
              f'contrast={rec["con"]:.4f} lr={rec["lr"]:.2e} '
              f'gpu={torch.cuda.max_memory_allocated()/2**30:.1f}G '
              f'[{time.time()-t0:.0f}s]', flush=True)

torch.save({'model': model.state_dict(), 'args': vars(a), 'in_dim': in_dim}, f'{OUT}/ckpt.pt')

# --- 导出 embedding（无增强、无掩码）---
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
print(f'\n最终 loss={hist[-1]["loss"]:.4f} (recon={hist[-1]["rec"]:.4f}, contrast={hist[-1]["con"]:.4f})')
print(f'峰值显存 = {torch.cuda.max_memory_allocated()/2**30:.2f} GB')
print(f'总用时 {time.time()-t0:.0f}s -> {OUT}')
print('TRAIN_DONE_MARKER')
