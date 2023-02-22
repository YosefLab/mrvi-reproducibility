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

import scvi_v2
import scanpy as sc
import scipy

# %%
comp_pca_dists = xr.load_dataarray("/home/justin/ghrepos/scvi-v2-reproducibility/results/sciplex_pipeline/distance_matrices/sciplex_A549_significant_all_phases.composition_pca.distance_matrices.nc")

# %%
g_dists = sns.clustermap(
    comp_pca_dists.sel(
        phase_name="G1",
        sample_x=comp_pca_dists.sample_x,
        sample_y=comp_pca_dists.sample_y,
    ).to_pandas(),
    cmap="YlGnBu",
    yticklabels=True,
    xticklabels=True,
    vmin=0,
    vmax=0.1
)
g_dists.ax_heatmap.set_xticklabels(
    g_dists.ax_heatmap.get_xmajorticklabels(), fontsize=2
)
g_dists.ax_heatmap.set_yticklabels(
    g_dists.ax_heatmap.get_ymajorticklabels(), fontsize=2
)

# %%
comp_pca_dists.sel(
    phase_name="G1",
    sample_x=comp_pca_dists.sample_x,
    sample_y=comp_pca_dists.sample_y,
).max()

# %%
# mrvi_dists = xr.load_dataarray("/home/justin/ghrepos/scvi-v2-reproducibility/results/sciplex_pipeline/distance_matrices/sciplex_A549_significant_all_phases.scviv2.distance_matrices.nc")
mrvi_dists = xr.load_dataarray("/home/justin/ghrepos/scvi-v2-reproducibility/results/sciplex_pipeline/distance_matrices/sciplex_A549_significant_all_phases.scviv2.normalized_distance_matrices.nc")
mrvi_dists -= 1

# %%
mrvi_dists.sel(
    phase_name="G1",
    sample_x=mrvi_dists.sample_x,
    sample_y=mrvi_dists.sample_y,
).max()

# %%
g1_dists = mrvi_dists.sel(
    phase_name="G1",
    sample_x=mrvi_dists.sample_x,
    sample_y=mrvi_dists.sample_y,
)
g1_pca_dists = comp_pca_dists.sel(
    phase_name="G1",
    sample_x=mrvi_dists.sample_x,
    sample_y=mrvi_dists.sample_y,
)

g1_pca_dists *= g1_dists.mean() / g1_pca_dists.mean()

# %%
# Plot histogram of two distances overlayed with 0.5 opacity
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.distplot(g1_pca_dists, ax=ax[0], kde=False)
sns.distplot(g1_dists, ax=ax[1], kde=False)

# %%
# split histogram into two parts, one with distances between same prod, one with
# distances between different prod
all_products = set()
all_doses = set()
for sample_name in g1_dists.sample_x.data:
    product_name, dose = sample_name.split("_")
    if product_name != "Vehicle":
        all_products.add(product_name)
    if dose != "0":
        all_doses.add(dose)

non_diag_mask = (
    np.ones(shape=g1_dists.shape) - np.identity(g1_dists.shape[0])
).astype(bool)
in_prod_mask = np.zeros(shape=g1_dists.shape, dtype=bool)
for product_name in all_products:
    for dosex in all_doses:
        for dosey in all_doses:
            if dosex == dosey:
                continue
            dosex_idx = np.where(
                g1_dists.sample_x.data == f"{product_name}_{dosex}"
            )[0]
            if len(dosex_idx) == 0:
                continue
            dosey_idx = np.where(
                g1_dists.sample_y.data == f"{product_name}_{dosey}"
            )[0]
            if len(dosey_idx) == 0:
                continue
            in_prod_mask[dosex_idx[0], dosey_idx[0]] = True

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(g1_dists.data[non_diag_mask], ax=ax, kde=False, label="all")
sns.histplot(g1_dists.data[in_prod_mask], ax=ax, kde=False, label="in_prod")
plt.legend()
# Draw a vertical line at the means
ax.axvline(g1_dists.data[non_diag_mask].mean(), color="b", linestyle="--")
ax.axvline(g1_dists.data[in_prod_mask].mean(), color="k", linestyle="--")

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(g1_pca_dists.data[non_diag_mask], ax=ax, kde=False, label="all")
sns.histplot(g1_pca_dists.data[in_prod_mask], ax=ax, kde=False, label="in_prod")
plt.legend()
# Draw a vertical line at the means
ax.axvline(g1_pca_dists.data[non_diag_mask].mean(), color="b", linestyle="--")
ax.axvline(g1_pca_dists.data[in_prod_mask].mean(), color="k", linestyle="--")

# %%
print(g1_dists.data[in_prod_mask].mean() / g1_dists.data[non_diag_mask].mean())
print(g1_pca_dists.data[in_prod_mask].mean() / g1_pca_dists.data[non_diag_mask].mean())

# %%
adjusted_ranks = (scipy.stats.rankdata(g1_dists.data).reshape(g1_dists.shape) - g1_dists.shape[0])
print(adjusted_ranks[in_prod_mask].mean() / non_diag_mask.sum())

adjusted_pca_ranks = (scipy.stats.rankdata(g1_pca_dists.data).reshape(g1_pca_dists.shape) - g1_pca_dists.shape[0])
print(adjusted_pca_ranks[in_prod_mask].mean() / non_diag_mask.sum())

# %%


adata = sc.read_h5ad(adata_in)
model = scvi_v2.MrVI.load(model_in, adata=adata)

cell_dists = model.get_local_sample_distances(
    adata, keep_cell=True,
)