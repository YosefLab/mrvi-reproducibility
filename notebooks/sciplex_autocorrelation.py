# %%
import argparse
import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import plotnine as p9
import faiss
from tqdm import tqdm

import scvi_v2
import scvi
import scanpy as sc
import scipy

# %%
method_name = "scviv2"
cell_line = "A549"

base_dir_path = "/home/justin/ghrepos/scvi-v2-reproducibility"
adata_path = os.path.join(
    base_dir_path,
    f"results/sciplex_pipeline/data/sciplex_{cell_line}_significant_all_phases.preprocessed.h5ad",
)
model_path = os.path.join(
    base_dir_path,
    f"results/sciplex_pipeline/models/sciplex_{cell_line}_significant_all_phases.{method_name}",
)

# %%
adata = sc.read_h5ad(adata_path)
model = scvi_v2.MrVI.load(model_path, adata=adata)

cell_dists = model.get_local_sample_distances(
    adata,
    keep_cell=True,
)
# %%
# Reduce to distance from vehicle for each product. Final matrix cell x prod_dist_from_vehicle
dists_from_vehicle = (
    cell_dists.cell.sel(sample_x="Vehicle_0").drop_sel(sample_y="Vehicle_0").copy()
)
del cell_dists
# %%
dists_from_vehicle_df = dists_from_vehicle.drop_vars("sample_x").to_pandas()
del dists_from_vehicle

# %%
latent_u = model.get_latent_representation(adata, give_z=False)
# %%
def faiss_hnsw_nn(X: np.ndarray, k: int):
    """Gpu HNSW nearest neighbor search using faiss."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    # index = faiss.IndexFlatL2(X.shape[1])
    M = 32
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(X)
    distances, indices = index.search(X, k)
    # distances are squared
    return indices, np.sqrt(distances)


# %%
nn_indices, nn_dists = faiss_hnsw_nn(latent_u, k=16)
# %%
# Remove self
nn_indices = nn_indices[:, 1:]
nn_dists = nn_dists[:, 1:]
# %%
# TODO: vmap this
def compute_autocorrelation(features_df, nn_indices, nn_dists, variant="binary"):
    N = features_df.shape[0]
    K = nn_indices.shape[1]
    autocorrelations = np.zeros(features_df.shape[1])
    for i in tqdm(range(features_df.shape[1])):
        if variant == "binary":
            W = np.zeros((N, N))
            np.add.at(W, (np.repeat(np.arange(N), K), nn_indices.ravel()), 1)
        elif variant == "gaussian_kernel":
            W = np.full((N, N), float("inf"))
            np.put(
                W, (np.repeat(np.arange(N), K), nn_indices.ravel()), nn_dists.ravel()
            )
            W = np.exp(-((W / nn_dists[:, K // 3][:, None]) ** 2))
            W = W / (np.sum(W, axis=1, keepdims=True) + 1e-6)
        else:
            raise ValueError(f"Unknown variant {variant}")
        x = features_df.values[:, i : i + 1]
        autocorrelation = (W * (x @ x.T)).sum()
        autocorrelations[i] = autocorrelation
    return pd.Series(autocorrelations, index=features_df.columns, name="autocorrelation")

# %%
autocorr_df = compute_autocorrelation(dists_from_vehicle_df, nn_indices, nn_dists, variant="binary")
# %%
# Get top two and bottom two autocorrelated products
top_k = autocorr_df.nlargest(5)
bot_k = autocorr_df.nsmallest(5)
print(top_k)
print(bot_k)

# %%
# Color MDE by distance to vehicle for these 4 products
U_mde = scvi.model.utils.mde(latent_u)

# %%
prods = top_k.index.tolist() + bot_k.index.tolist()
autocorr_vals  = top_k.values.tolist() + bot_k.values.tolist()

adata.obsm["U_mde"] = U_mde

for prod, autocorr in zip(prods, autocorr_vals):
    adata.obs[f"{prod}_dist_to_vehicle"] = dists_from_vehicle_df[prod]
    sc.pl.embedding(adata, "U_mde", color=f"{prod}_dist_to_vehicle", show=False)
    plt.title(f"{prod} Distance to Vehicle, Autocorrelation: {autocorr}")
    plt.savefig(os.path.join(base_dir_path, f"notebooks/figures/{prod}_dist_to_vehicle.png"))
    plt.clf()
# %%
