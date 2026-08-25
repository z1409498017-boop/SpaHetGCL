import gzip, glob, os, re, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

t0 = time.time()
D   = '/root/autodl-tmp/spahet/data/GSE234713'
OUT = '/root/autodl-tmp/spahet/data/processed'
os.makedirs(OUT, exist_ok=True)
MAT = f'{D}/GSE234713_CosMx_normalized_matrix.txt.gz'

# ---------- 1. 解析矩阵表头 ----------
ncomment = 0
with gzip.open(MAT, 'rt') as f:
    for line in f:
        if line.startswith('"#'):
            ncomment += 1
        else:
            header = line.rstrip('\n'); break
cols  = [c.strip('"') for c in header.split('\t')]
genes = cols[3:]
print(f'[matrix] 注释行={ncomment} 总列={len(cols)} 前3列={cols[:3]} 基因数={len(genes)}', flush=True)

dtypes = {g: np.float32 for g in genes}
for c in cols[:3]: dtypes[c] = str
mat = pd.read_csv(MAT, sep='\t', skiprows=ncomment, dtype=dtypes)
print(f'[matrix] 读入 {mat.shape}  用时 {time.time()-t0:.0f}s', flush=True)
assert list(mat.columns) == cols, '列名不匹配'
mat['key'] = (mat['patient'].str.replace(' ', '_', regex=False) + '_'
              + mat['cell_id'] + '_' + mat['fov'])

# ---------- 2. annotation ----------
ann = pd.read_csv(f'{D}/GSE234713_CosMx_annotation.csv.gz')
ann.columns = ['key', 'subset', 'celltype']
print(f'[ann] {ann.shape}', flush=True)

# ---------- 3. metadata（坐标 + 形态） ----------
KEEP = ['Area','AspectRatio','CenterX_global_px','CenterY_global_px','Width','Height',
        'Mean.PanCK','Mean.CD45','Mean.CD3','Mean.DAPI']
metas = []
for f in sorted(glob.glob(f'{D}/GSM*_metadata_file.csv.gz')):
    s = re.match(r'GSM\d+_(.+)_metadata_file\.csv\.gz', os.path.basename(f)).group(1)
    m = pd.read_csv(f, dtype={'fov': str, 'cell_ID': str})
    m['sample'] = s
    m['key'] = s + '_' + m['cell_ID'] + '_' + m['fov']
    metas.append(m[['key','sample'] + [c for c in KEEP if c in m.columns]])
meta = pd.concat(metas, ignore_index=True)
print(f'[meta] {meta.shape} 样本数={meta["sample"].nunique()}', flush=True)

# ---------- 4. 对齐 ----------
for nm, df in [('mat', mat), ('ann', ann), ('meta', meta)]:
    assert df['key'].is_unique, f'{nm} key 不唯一'
mat, ann, meta = mat.set_index('key'), ann.set_index('key'), meta.set_index('key')
common = mat.index.intersection(ann.index).intersection(meta.index)
common = pd.Index(sorted(common))
print(f'[join] 三者交集 = {len(common)}', flush=True)

# ---------- 5. 组装 AnnData ----------
X = sp.csr_matrix(mat.loc[common, genes].to_numpy(dtype=np.float32))
obs = pd.DataFrame(index=common.astype(str))
obs['sample']    = meta.loc[common, 'sample'].values
obs['condition'] = obs['sample'].str.split('_').str[0]
obs['celltype']  = ann.loc[common, 'celltype'].values
obs['subset']    = ann.loc[common, 'subset'].values
for c in KEEP:
    if c in meta.columns: obs[c] = meta.loc[common, c].values.astype(np.float32)
obs['n_expressed'] = np.asarray((X > 0).sum(1)).ravel()
obs['total_norm']  = np.asarray(X.sum(1)).ravel()

adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name='gene')))
adata.obsm['spatial'] = np.c_[obs['CenterX_global_px'].values,
                              obs['CenterY_global_px'].values].astype(np.float32)
adata.uns['source'] = 'GSE234713 CosMx, normalized matrix (pre-normalized by authors)'
adata.write_h5ad(f'{OUT}/GSE234713.h5ad', compression='gzip')

# ---------- 6. 诊断 ----------
print('\n' + '='*60)
print(f'shape = {adata.shape}   稀疏度(非零占比) = {X.nnz/np.prod(X.shape):.4f}')
print(f'表达值范围 = [{X.data.min():.4f}, {X.data.max():.4f}]')
print(f'\n--- 每样本细胞数 ---\n{obs["sample"].value_counts().sort_index()}')
print(f'\n--- 分组 ---\n{obs["condition"].value_counts()}')
print(f'\n--- 粗类型 (subset, 拟作节点类型) ---\n{obs["subset"].value_counts(dropna=False)}')
print(f'\n--- 细类型数 = {obs["celltype"].nunique()} ; 含 NA = {obs["celltype"].isna().sum()}')
print(f'\n--- 巨噬/成纤维相关类型 ---')
mask = obs['celltype'].astype(str).str.contains('acrophage|^M0|^M1|^M2|ibroblast|onocyte|FRC', regex=True, na=False)
print(obs.loc[mask, 'celltype'].value_counts())
print(f'\n--- 每样本坐标范围(px) ---')
for s, g in obs.groupby('sample'):
    print(f'  {s}: x[{g["CenterX_global_px"].min():.0f},{g["CenterX_global_px"].max():.0f}] '
          f'y[{g["CenterY_global_px"].min():.0f},{g["CenterY_global_px"].max():.0f}] n={len(g)}')
print(f'\n文件: {OUT}/GSE234713.h5ad  ({os.path.getsize(OUT+"/GSE234713.h5ad")/2**20:.0f} MB)')
print(f'总用时 {time.time()-t0:.0f}s')
print('BUILD_DONE_MARKER')
