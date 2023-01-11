# %%
import os

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2

# %%
symsim_results_root = (
    "/home/justin/ghrepos/scvi-v2-reproducibility/results/symsim_pipeline"
)
symsim_adatas_root = f"{symsim_results_root}/data"

# Iterate through files in symsim_adatas_root
model_adatas = {}
for adata_file in sorted(os.listdir(symsim_adatas_root)):
    if adata_file.endswith(".h5ad") and not adata_file.endswith(".preprocessed.h5ad"):
        adata = sc.read_h5ad(f"{symsim_adatas_root}/{adata_file}")
        model_name = adata_file.split(".")[-2]
        model_adatas[model_name] = adata

# %%
scvi_adata = model_adatas["scviv2"]
# %%
sample_order = scvi_adata.uns[scvi_adata.uns["sample_order_key"]]
sample_meta = scvi_adata.uns["sample_metadata"]
local_sample_rep = scvi_adata.obsm[scvi_adata.uns["local_sample_rep_key"]]
local_sample_dists = scvi_adata.obsm[scvi_adata.uns["local_sample_dists_key"]]
# %%
# Plot histogram of raw distances colored by cell type
dist_cts = np.broadcast_to(
    np.expand_dims(scvi_adata.obs["celltype"].copy().to_numpy(), (1, 2)),
    local_sample_dists.shape,
).flatten()
dist_df = pd.DataFrame(local_sample_dists.flatten(), columns=["dist"])
dist_df["celltype"] = dist_cts

sns.histplot(data=dist_df, x="dist", hue="celltype")
plt.xlim(0, 2)

# %%
# Histogram for a single sample sample pair across all cells
i = 0
j = 31
single_dist_df = pd.DataFrame(
    local_sample_dists[:, i, j].flatten(), columns=["dist"]
)
single_dist_df["celltype"] = scvi_adata.obs["celltype"].copy().to_numpy()

sns.histplot(data=single_dist_df, x="dist", hue="celltype")
plt.xlim(0, 2)
plt.title(f"{sample_order[i]} vs {sample_order[j]}")

# %%
# Simply set scale of color bar by 0 to 3*standard deviation of all distances
# This already looks good
max_dist = 3*np.std(local_sample_dists.flatten())
for ct in scvi_adata.obs["celltype"].unique():
    ct_dists = local_sample_dists[scvi_adata.obs["celltype"] == ct]
    avged_ct_dists = np.mean(ct_dists, axis=0)
    sns.heatmap(avged_ct_dists, cmap="YlGnBu", xticklabels=False, yticklabels=False, vmin=0, vmax=max_dist)
    plt.show()
    plt.clf()


# %%
# Plot chi2 pdf (distribution of distances based on the u->v prior)
# dof = local_sample_rep.shape[2]
dof = 10
x = np.linspace(chi2.ppf(0.01, dof),
                chi2.ppf(0.99, dof), 100)
plt.plot(x, chi2.pdf(x, dof),
       'r-', lw=5, alpha=0.6, label='chi2 pdf')
# %%
def dist_to_sim(dist, dof=2):
    return 1 - chi2.cdf((dist**2)/2, dof)

# %%
# run dist_to_sim on every element of local_sample_dists then plot
local_sample_sims = np.apply_along_axis(dist_to_sim, 1, local_sample_dists)
# %%
# Plot histogram of chi based similarities. Ends up being everything close to 1
plt.hist(local_sample_sims.flatten(), bins=100)

# %%
# Plot chi based similarities per cell type
for ct in scvi_adata.obs["celltype"].unique():
    ct_sims = local_sample_sims[scvi_adata.obs["celltype"] == ct]
    avged_ct_sims = np.mean(ct_sims, axis=0)
    sns.heatmap(avged_ct_sims, cmap="YlGnBu_r", xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()
    plt.clf()

# %%
# Get standard dev of control cell type then use for z scores.
# Use 0 as the mean (think of as a half-normal distribution with loc=0).
# In practice we would use the std dev of v for the original sample.
control_ct_dists = local_sample_dists[scvi_adata.obs["celltype"] == "CT2:1"]
stdev_zero_mean = (control_ct_dists**2).mean()**0.5
# %%
# Take 95% percentile of all distances to prune outliers
ninetyfifth_percentile = np.quantile(local_sample_dists.flatten(), 0.95)
# %%
# Compute similarities as (95pct - dists) / 95pct.
# Clip values below 0.
local_adjusted_sims = np.clip(ninetyfifth_percentile - local_sample_dists, a_min=0, a_max=None) / ninetyfifth_percentile

# %%
# Plot heatmaps for these similarities per cell type.
# Looks good and scale is from 0 to 1.
for ct in scvi_adata.obs["celltype"].unique():
    ct_sims = local_adjusted_sims[scvi_adata.obs["celltype"] == ct]
    avged_ct_sims = np.mean(ct_sims, axis=0)
    sns.heatmap(avged_ct_sims, cmap="YlGnBu_r", xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()
    plt.clf()
# %%
# Compute normalized distances as a function of the standard dev for the control cell type.
# The final expression becomes clip(dists, max=95pct) / stddev.
# The color bar range is set to 0 to the max of all normalized distances.
for ct in scvi_adata.obs["celltype"].unique():
    ct_sims = local_adjusted_sims[scvi_adata.obs["celltype"] == ct]
    ct_sims = ct_sims * ninetyfifth_percentile / stdev_zero_mean
    v_max = np.max(ct_sims)
    ct_sims = v_max - ct_sims
    avged_ct_sims = np.mean(ct_sims, axis=0)
    sns.heatmap(avged_ct_sims, cmap="YlGnBu", xticklabels=False, yticklabels=False, vmin=0, vmax=v_max)
    plt.show()
    plt.clf()

# %%
