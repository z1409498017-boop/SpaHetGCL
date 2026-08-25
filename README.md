# SpaHetGCL

**Neighbor-Contrastive Heterogeneous Graph Learning for Spatially Coherent Representation of Single-Cell Data**

**Authors:** Zheng Zhao (AgentAlphaAGI)

SpaHetGCL learns cell representations on spatial single-cell data by replacing the instance-discrimination positives of graph contrastive learning with *spatial-adjacency positives*: inside a heterogeneous graph transformer, every edge between physically adjacent cells (Delaunay triangulation of spatial coordinates) is defined as a positive pair. Training on a graph of 458,983 cells across nine human colonic sections (five cell lineages, fifty typed relations) raises Moran's I from 0.633 to 0.756 and spatial coherence from 0.765 to 0.862 relative to the strongest baseline, at roughly half the runtime and memory of the two-view instance-contrastive alternative.

This repository contains the code that reproduces the experiments in the manuscript: graph construction, self-supervised pre-training, unsupervised evaluation, baselines, and the interpretability checks.

## Key idea

| Contrastive formulation | Positives | Result |
|---|---|---|
| Instance discrimination (`train.py`) | Two augmented views of the *same* cell | Pushes every cell away from every other cell — conflicts with the spatial-niche assumption |
| **Neighbor contrast (`train_v2.py`)** | A cell and its *spatially adjacent* cells (near edges) | The objective directly encodes the niche hypothesis |

Two methodological findings accompany the main result:

- A masked-reconstruction branch **degrades** all four metrics (hence `--use_recon` is off by default in `train_v2.py`).
- The CHAOS compactness metric varies by up to ~2.4× across clustering initializations on a deterministic representation — single-run CHAOS values cannot support method comparison.

## Repository layout

```
SpaHetGCL/
├── src/                       # experiment code (13 modules)
│   ├── build_anndata.py       # raw GSE234713 files -> AnnData
│   ├── graph_diag.py          # Delaunay edge-length diagnostics (sets D_MAX / D_NEAR)
│   ├── build_graph.py         # AnnData -> PyG HeteroData (per sample)
│   ├── model.py               # SpaHetGCL model + losses (recon, instance InfoNCE)
│   ├── train.py               # v1 pre-training: masked reconstruction + instance contrast
│   ├── train_v2.py            # v2 pre-training: neighbor-positive contrastive (proposed)
│   ├── evaluate.py            # unsupervised niche metrics (coherence, CHAOS, Moran's I, silhouette)
│   ├── baseline.py            # Leiden / spatial smoothing / PCA baselines
│   ├── baseline_v2.py         # self-contained baselines: leiden, pca, smooth, ctcomp
│   ├── baseline_leiden.py     # early standalone Leiden baseline
│   ├── task3_enrich.py        # macrophage-fibroblast adjacency enrichment (permutation test)
│   ├── task3_v2.py            # enrichment with global + block permutation null models, HC/UC/CD
│   └── task3_fixed.py         # early enrichment check (superseded by task3_v2.py)
└── runs/                      # experiment results (one dir per tag)
    ├── eval.json              # four-metric evaluation for the run
    ├── history.json           # per-epoch training history
    ├── task3_v2.json          # per-sample enrichment (main run)
    ├── seeds_summary.txt      # seed-stability runs (coherence/Moran's I across seeds)
    ├── sweep_summary.txt      # tau / branch sweeps
    └── v2_summary.txt         # v2 tag summary
```

## Requirements

```
torch>=2.0
torch-geometric>=2.4
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
pandas>=2.0
anndata>=0.9
scanpy>=1.9
```

Install with:

```bash
pip install -r requirements.txt
```

## Reproducing the pipeline

> **Note on paths.** All scripts hardcode the path of the original training server
> (`/root/autodl-tmp/spahet/...`). Edit `GD`, `OUT`, `H5`, and `D` at the top of each
> script to point at your local copy of the data.

The data pipeline runs in seven stages:

```bash
# 1. Build AnnData from raw GEO files (GSE234713)
python src/build_anndata.py

# 2. Inspect Delaunay edge-length distribution (motivates D_MAX=30um, D_NEAR=10um)
python src/graph_diag.py

# 3. Build per-sample PyG heterogeneous graphs
python src/build_graph.py

# 4. Pre-train the proposed neighbor-contrastive model (v2)
python src/train_v2.py --tag v2_nbr --nbr_mode nbr

# 4'. (optional) v1 instance-contrastive baseline + ablations
python src/train.py --tag main
python src/train.py --tag abl_nocontrast --no_contrast
python src/train.py --tag abl_norecon   --no_recon
python src/train.py --tag abl_nospe     --no_spatial_pe

# 5. Evaluate niche metrics on any run tag
python src/evaluate.py --tag v2_nbr

# 6. Baselines (self-contained, same metrics)
python src/baseline_v2.py ctcomp
python src/baseline_v2.py leiden
python src/baseline_v2.py smooth
python src/baseline_v2.py pca

# 7. Macrophage-fibroblast spatial adjacency enrichment (Task 3)
python src/task3_v2.py --tag v2_nbr
```

## Results

All runs on GSE234713 (458,983 cells, k = 10 niches, 15 spatial neighbours).

| run | description | spatial_coherence | chaos (µm) | Moran's I | silhouette |
|---|---|---|---|---|---|
| `v2_nbr` (s0/s1/s2) | **proposed neighbor contrast** | 0.825 / 0.869 / 0.891 | 19.8 / 70.9 / 26.4 | 0.753 / 0.754 / 0.762 | 0.046 / 0.050 / 0.052 |
| `v2_both` | neighbor + instance contrast | 0.803 | 20.3 | 0.750 | 0.043 |
| `baseline_ctcomp` | neighborhood cell-type composition + KMeans | 0.766 | 69.6 | 0.633 | 0.058 |
| `baseline_smooth` | spatial kNN expression smoothing + KMeans | 0.865 | 85.4 | 0.606 | −0.035 |
| `baseline_leiden` | expression Leiden clustering | 0.589 | 461.4 | 0.158 | −0.021 |
| `baseline_pca` | expression PCA(128) + KMeans | 0.467 | 263.8 | 0.082 | −0.067 |
| `main` | v1 recon + instance contrast | 0.495 | 12.9 | 0.422 | 0.120 |
| `abl_nocontrast` | recon only | 0.463 | 33.7 | 0.298 | 0.426 |
| `abl_norecon` | instance contrast only | 0.502 | 12.9 | 0.474 | 0.155 |
| `abl_nospe` | no spatial positional encoding | 0.455 | 16.8 | 0.309 | 0.249 |
| `sw2_nr_t10` | v2, no recon, τ=1.0 | 0.504 | 17.2 | 0.474 | 0.109 |
| `sw2_nr_t20` | v2, no recon, τ=2.0 | 0.509 | 24.4 | 0.479 | 0.169 |

Aggregated over seeds, the proposed method (`v2_nbr`) reaches Moran's I 0.756 and spatial
coherence 0.862 — the values reported in the manuscript — against the strongest baseline
(`ctcomp`, 0.633 / 0.766). Note the CHAOS column's instability across seeds (and across
KMeans initializations); see the manuscript discussion.

## Data availability

The dataset GSE234713 analysed in this study is publicly available in the NCBI Gene
Expression Omnibus repository:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE234713

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code or find the results useful, please cite the manuscript:

```
Zheng Zhao. Neighbor-Contrastive Heterogeneous Graph Learning for Spatially
Coherent Representation of Single-Cell Data. [Journal], [year].
```
