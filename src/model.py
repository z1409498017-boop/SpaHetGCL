"""SpaHetGCL: 空间异质图对比表征学习。

自监督双任务：
  L_recon    掩码特征重建 (masked node feature modeling)
  L_contrast 双视图节点级对比 (feature mask + edge dropout 增强)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HGTConv


def spatial_pe(pos, dim, scale=100.0):
    """2D 正弦空间位置编码；pos 单位 um，scale 为特征长度尺度。"""
    d4 = dim // 4
    freq = torch.exp(torch.arange(d4, device=pos.device, dtype=torch.float32)
                     * (-np.log(10000.0) / max(d4, 1)))
    out = []
    for i in range(2):
        p = pos[:, i:i + 1] / scale
        out += [torch.sin(p * freq), torch.cos(p * freq)]
    pe = torch.cat(out, 1)
    if pe.size(1) < dim:                                  # 维度不整除时补零
        pe = F.pad(pe, (0, dim - pe.size(1)))
    return pe


class SpaHetGCL(nn.Module):
    def __init__(self, in_dim, metadata, hidden=128, heads=4, layers=3, proj_dim=64):
        super().__init__()
        node_types = metadata[0]
        self.hidden = hidden
        self.proj  = nn.ModuleDict({t: nn.Linear(in_dim, hidden) for t in node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.norms = nn.ModuleList([
            nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types}) for _ in range(layers)])
        self.dec   = nn.ModuleDict({t: nn.Linear(hidden, in_dim) for t in node_types})
        self.head  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))

    def encode(self, x_dict, pos_dict, eidx_dict):
        h = {t: self.proj[t](x) + spatial_pe(pos_dict[t], self.hidden) for t, x in x_dict.items()}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, eidx_dict)
            h = {t: F.elu(norm[t](out[t])) if out.get(t) is not None else h[t] for t in h}
        return h

    def forward(self, x_dict, pos_dict, eidx_dict):
        return self.encode(x_dict, pos_dict, eidx_dict)


def mask_features(x_dict, ratio, gen):
    """按元素掩码，返回掩码后特征与布尔掩码。"""
    xm, masks = {}, {}
    for t, x in x_dict.items():
        m = torch.rand(x.shape, device=x.device, generator=gen) < ratio
        xm[t] = x.masked_fill(m, 0.0)
        masks[t] = m
    return xm, masks


def drop_edges(eidx_dict, ratio, gen):
    out = {}
    for r, ei in eidx_dict.items():
        if ei.size(1) == 0:
            out[r] = ei; continue
        keep = torch.rand(ei.size(1), device=ei.device, generator=gen) >= ratio
        out[r] = ei[:, keep] if keep.any() else ei[:, :1]
        continue
    return out


def recon_loss(dec_dict, x_dict, mask_dict):
    """仅在被掩码位置计算 MSE。"""
    num = tot = 0.0
    for t, m in mask_dict.items():
        if m.sum() == 0: continue
        num = num + F.mse_loss(dec_dict[t][m], x_dict[t][m], reduction='sum')
        tot = tot + m.sum()
    return num / tot.clamp(min=1)


def nt_xent(z1, z2, tau=0.5, n_sample=4096, gen=None):
    """跨节点类型统一采样的 InfoNCE。"""
    n = z1.size(0)
    if n > n_sample:
        idx = torch.randperm(n, device=z1.device, generator=gen)[:n_sample]
        z1, z2 = z1[idx], z2[idx]
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim = z1 @ z2.t() / tau
    lbl = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.t(), lbl))
