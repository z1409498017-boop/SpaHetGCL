#!/usr/bin/env python3
"""Task3 修正版：HC/UC/CD 差分比较 + 条件置换"""
import json, numpy as np, time
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind

t0 = time.time()
with open("/root/autodl-tmp/hetero_graphs.json") as f:
    graphs = json.load(f)

print("=== 髓系-基质 near 边密度（观测值）===")
results = {}
for g in graphs:
    sid, ct_dict = g["sample"], g["celltype_count"]
    n_mye = sum(ct_dict.get(t, 0) for t in ["myeloid"])
    n_str = sum(ct_dict.get(t, 0) for t in ["stromal"])
    edges = [e for e in g["edges"] if e[2] in ["myeloid-stromal_near", "stromal-myeloid_near"]]
    obs = len(edges)
    expected = 2 * n_mye * n_str / g["num_nodes"]  # 无向图期望
    fold = obs / expected if expected > 0 else 0
    cond = g["sample"].split("_")[0]  # HC / UC / CD
    results[sid] = {"obs": obs, "exp": expected, "fold": fold, "condition": cond,
                    "n_mye": n_mye, "n_str": n_str}
    print(f"  {sid:8} ({cond}): obs={obs:5} exp={expected:7.1f} fold={fold:.3f}")

print("\n=== 按条件分组 ===")
groups = {"HC": [], "UC": [], "CD": []}
for sid, r in results.items():
    groups[r["condition"]].append(r["fold"])

for cond in ["HC", "UC", "CD"]:
    vals = groups[cond]
    print(f"  {cond}: n={len(vals)}, fold = {np.mean(vals):.3f} ± {np.std(vals):.3f}")

# HC vs UC
if len(groups["HC"]) >= 2 and len(groups["UC"]) >= 2:
    t_stat, p_val = ttest_ind(groups["HC"], groups["UC"])
    print(f"\nHC vs UC: t={t_stat:.2f}, p={p_val:.4f}")

# HC vs CD
if len(groups["HC"]) >= 2 and len(groups["CD"]) >= 2:
    t_stat, p_val = ttest_ind(groups["HC"], groups["CD"])
    print(f"HC vs CD: t={t_stat:.2f}, p={p_val:.4f}")

out = {"by_sample": results, "by_group": groups, "time_sec": round(time.time()-t0, 1)}
with open("/root/autodl-tmp/task3_fixed.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n用时 {out['time_sec']}s")
print("TASK3_FIXED_DONE")
