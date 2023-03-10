"""
Exploratory notebook computing the autocorrelation of distances to Vehicle in the sciplex dataset.
"""
# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import jax
import jax.numpy as jnp
import faiss

import scvi_v2
import scvi
import scanpy as sc
import seaborn as sns
import scipy
import xarray as xr

# %%
method_name = "scviv2"
cell_line = "A549"
data_variant = "significant_subsampled"

base_dir_path = "/home/justin/ghrepos/scvi-v2-reproducibility"
results_dir_path = os.path.join(base_dir_path, "results/sciplex_pipeline")
adata_path = os.path.join(
    results_dir_path,
    f"data/sciplex_{cell_line}_{data_variant}_all_phases.preprocessed.h5ad",
)
model_path = os.path.join(
    results_dir_path,
    f"models/sciplex_{cell_line}_{data_variant}_all_phases.{method_name}",
)


# %%
adata = sc.read_h5ad(adata_path)
model = scvi_v2.MrVI.load(model_path, adata=adata)

# %%
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
adata.obsm["U"] = latent_u
# %%
def faiss_hnsw_nn(X: np.ndarray, k: int):
    """Gpu HNSW nearest neighbor search using faiss."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    M = 32
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(X)
    distances, indices = index.search(X, k)
    # distances are squared
    return indices, np.sqrt(distances)


# %%
nn_indices, nn_dists = faiss_hnsw_nn(latent_u, k=51)
# %%
# Remove self
nn_indices = nn_indices[:, 1:]
nn_dists = nn_dists[:, 1:]
# %%
def compute_rank_autocorrelation(
    features_df, nn_indices, nn_dists, variant="binary", return_local=False
):
    features_df = dists_from_vehicle_df
    variant = "binary"
    N = features_df.shape[0]
    K = nn_indices.shape[1]

    if variant == "binary":
        W = np.zeros((N, N))
        np.add.at(W, (np.repeat(np.arange(N), K), nn_indices.ravel()), 1)
        W = W / (np.sum(W, axis=1, keepdims=True) + 1e-6)
    elif variant == "gaussian_kernel":
        W = np.full((N, N), float("inf"))
        np.put(W, (np.repeat(np.arange(N), K), nn_indices.ravel()), nn_dists.ravel())
        W = np.exp(-((W / nn_dists[:, K // 3][:, None]) ** 2))
        W = W / (np.sum(W, axis=1, keepdims=True) + 1e-6)
    else:
        raise ValueError(f"Unknown variant {variant}")

    feature_vals = jnp.array(features_df.values)
    feature_vals_ranks = scipy.stats.rankdata(feature_vals, axis=1) / N
    local_autocorrelations = jax.lax.map(
        lambda x: (W * jnp.outer(x, x)).sum(axis=1)
        - W.mean() * jnp.outer(x, x).sum(axis=1),
        feature_vals_ranks.T,
    )
    if return_local:
        return pd.DataFrame(
            jax.device_get(local_autocorrelations),
            index=features_df.columns,
            columns=features_df.index,
        )

    autocorrelations = local_autocorrelations.sum(axis=1)
    return pd.Series(
        jax.device_get(autocorrelations),
        index=features_df.columns,
        name="autocorrelation",
    )


# %%
autocorr_df = compute_rank_autocorrelation(
    dists_from_vehicle_df, nn_indices, nn_dists, variant="binary"
)
local_autocorr_df = compute_rank_autocorrelation(
    dists_from_vehicle_df, nn_indices, nn_dists, variant="binary", return_local=True
)

# %%
# Save results
autocorr_df.to_csv(
    os.path.join(
        results_dir_path,
        f"metrics/sciplex_{cell_line}_significant_all_phases.{method_name}.autocorr.csv",
    )
)
local_autocorr_df.to_csv(
    os.path.join(
        results_dir_path,
        f"metrics/sciplex_{cell_line}_significant_all_phases.{method_name}.local_autocorr.csv",
    )
)

# %%
# Load results
autocorr_df = pd.read_csv(
    os.path.join(
        results_dir_path,
        f"metrics/sciplex_{cell_line}_significant_all_phases.{method_name}.autocorr.csv",
    ),
    index_col=0,
)["autocorrelation"]
local_autocorr_df = pd.read_csv(
    os.path.join(
        results_dir_path,
        f"metrics/sciplex_{cell_line}_significant_all_phases.{method_name}.local_autocorr.csv",
    ),
    index_col=0,
    header=0,
)


# %%
# Get top k and bottom k autocorrelated products
top_k = autocorr_df.nlargest(5)
bot_k = autocorr_df.nsmallest(5)
print(top_k)
print(bot_k)

# %%
# Color MDE by distance to vehicle for these 4 products
U_mde = scvi.model.utils.mde(latent_u)
adata.obsm["U_mde"] = U_mde

# %%
prods = top_k.index.tolist() + bot_k.index.tolist()
autocorr_vals = top_k.values.tolist() + bot_k.values.tolist()

for prod, autocorr in zip(prods, autocorr_vals):
    adata.obs[f"{prod}_dist_to_vehicle"] = dists_from_vehicle_df[prod]
    sc.pl.embedding(adata, "U_mde", color=f"{prod}_dist_to_vehicle", show=False)
    plt.title(f"{prod} Distance to Vehicle, Autocorrelation: {autocorr}")
    plt.savefig(
        os.path.join(base_dir_path, f"notebooks/figures/{prod}_dist_to_vehicle.png"),
        bbox_inches="tight",
    )
    plt.clf()

    adata.obs[f"{prod}_local_rank_autocorr"] = local_autocorr_df.loc[prod]
    sc.pl.embedding(
        adata,
        "U_mde",
        color=f"{prod}_local_rank_autocorr",
        show=False,
        vmin=0,
        vmax=1e-4,
    )
    plt.title(f"{prod} Local Autocorrelation, Autocorrelation: {autocorr}")
    plt.savefig(
        os.path.join(
            base_dir_path, f"notebooks/figures/{prod}_local_rank_autocorr.png"
        ),
        bbox_inches="tight",
    )
    plt.clf()

# %%
# Color by phase
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.embedding(adata, "U_mde", color="phase", show=False, ax=ax)
plt.savefig(
    os.path.join(base_dir_path, f"notebooks/figures/{method_name}_phase_mde.png"),
    bbox_inches="tight",
)

# %%
# Plot select correlation drug plots over doses
# sel_prod = "TAK-901"
# sel_prod = "Tanespimycin (17-AAG)"
sel_prod = "Thalidomide"
doses = ["10", "100", "1000", "10000"]
fig, axes = plt.subplots(len(doses), 2, figsize=(10, len(doses) * 5))
plt.title(f"{sel_prod}")
for i, dose in enumerate(doses):
    ax_pair = axes[i, :]
    prod = sel_prod + "_" + dose
    adata.obs[f"{prod}_dist_to_vehicle"] = dists_from_vehicle_df[prod]
    sc.pl.embedding(
        adata,
        "U_mde",
        color=f"{prod}_dist_to_vehicle",
        show=False,
        ax=ax_pair[0],
        vmin=0,
        vmax=1,
    )

    adata.obs[f"{prod}_local_rank_autocorr"] = local_autocorr_df.loc[prod]
    sc.pl.embedding(
        adata, "U_mde", color=f"{prod}_local_rank_autocorr", show=False, ax=ax_pair[1]
    )
plt.savefig(
    os.path.join(base_dir_path, f"notebooks/figures/{sel_prod}_full_autocorr.png"),
)

# %%
# Plot smoothed abundances of select prod (num neighbors including self of that prod dose combo)
nn_indices, _ = faiss_hnsw_nn(latent_u, k=200)

fig, axes = plt.subplots(len(doses) // 2, 2, figsize=(10, len(doses) // 2 * 5))
for i, dose in enumerate(doses):
    ax = axes[i // 2, i % 2]
    prod = sel_prod + "_" + dose
    prod_idxs = adata.obs["product_dose"] == prod
    num_neighbors_prod = np.vectorize(lambda i: prod_idxs[nn_indices[i]].sum())(
        np.arange(nn_indices.shape[0])
    )
    adata.obs[f"{prod}_nn"] = num_neighbors_prod
    sc.pl.embedding(
        adata,
        "U_mde",
        color=f"{prod}_nn",
        show=False,
        ax=ax,
        vmin=0,
        vmax=3,
    )
    ax.set_title(
        f"{prod} 200 NN, total cells: {(adata.obs[f'product_dose'] == prod).sum()}"
    )
plt.savefig(
    os.path.join(base_dir_path, f"notebooks/figures/{sel_prod}_full_nn.png"),
)

# %%
# Threshold on distance to vehicle then cluster for DEG analysis
dist_threshold = 0.6
adata.layers["log1p"] = sc.pp.log1p(adata, copy=True).X
adata.uns["log1p"] = {"base": None}  # address bug in sc.tl.rank_genes_groups

fig, axes = plt.subplots(len(doses), 3, figsize=(20, len(doses) * 5))
for i, dose in enumerate(doses):
    ax_pair = axes[i, :]
    prod = sel_prod + "_" + dose
    adata.obs[f"{prod}_thresh"] = (
        (dists_from_vehicle_df[prod] >= dist_threshold).astype(int).astype("category")
    )
    sc.pl.embedding(
        adata,
        "U_mde",
        color=f"{prod}_thresh",
        show=False,
        ax=ax_pair[0],
        vmin=0,
        vmax=2,
    )

    thresh_idxs = adata.obs[f"{prod}_thresh"] == 1
    thresh_adata = adata[thresh_idxs]
    sc.pp.neighbors(thresh_adata, use_rep="U")
    sc.tl.leiden(thresh_adata, resolution=0.2, key_added=f"{prod}_leiden")
    adata.obs[f"{prod}_leiden"] = "not clustered"
    adata.obs.loc[thresh_idxs, f"{prod}_leiden"] = thresh_adata.obs[f"{prod}_leiden"]
    adata.obs[f"{prod}_leiden"] = adata.obs[f"{prod}_leiden"].astype("category")
    if "not clustered" in adata.obs[f"{prod}_leiden"].cat.categories:
        adata.obs[f"{prod}_leiden"] = adata.obs[
            f"{prod}_leiden"
        ].cat.reorder_categories(
            ["not clustered"] + list(adata.obs[f"{prod}_leiden"].cat.categories[:-1])
        )
    sc.pl.embedding(adata, "U_mde", color=f"{prod}_leiden", show=False, ax=ax_pair[1])

    vehicle_adata = adata[adata.obs["product_dose"] == "Vehicle_0"]
    # Filter out groups with < 10 cells
    for group in vehicle_adata.obs[f"{prod}_leiden"].cat.categories:
        if sum(vehicle_adata.obs[f"{prod}_leiden"] == group) < 10:
            adata.obs.loc[
                adata.obs[f"{prod}_leiden"] == group, f"{prod}_leiden"
            ] = "not clustered"
    adata.obs[f"{prod}_leiden"] = adata.obs[
        f"{prod}_leiden"
    ].cat.remove_unused_categories()
    sc.tl.rank_genes_groups(
        vehicle_adata,
        f"{prod}_leiden",
        layer="log1p",
        method="wilcoxon",
    )
    sc.pl.rank_genes_groups_dotplot(
        vehicle_adata,
        n_genes=5,
        groups=list(vehicle_adata.obs[f"{prod}_leiden"].cat.categories[1:])
        if "not clustered" in adata.obs[f"{prod}_leiden"].cat.categories
        else list(vehicle_adata.obs[f"{prod}_leiden"].cat.categories),
        values_to_plot="logfoldchanges",
        cmap="bwr",
        vmin=-4,
        vmax=4,
        min_logfoldchange=0.5,
        dendrogram=False,
        ax=ax_pair[2],
        show=False,
    )
fig.tight_layout()
plt.savefig(
    os.path.join(base_dir_path, f"notebooks/figures/{sel_prod}_full_thresh_deg.png"),
    bbox_inches="tight",
)

# %%
# Load composition scVI results
baseline_method_name = "composition_SCVI_leiden1_subleiden1"
baseline_dist_array = xr.load_dataarray(
    os.path.join(
        results_dir_path,
        f"distance_matrices/sciplex_{cell_line}_{data_variant}_all_phases.{baseline_method_name}.distance_matrices.nc",
    )
)
baseline_adata = sc.read(
    os.path.join(
        results_dir_path,
        f"data/sciplex_{cell_line}_{data_variant}_all_phases.{baseline_method_name}.h5ad",
    )
)
# %%
# For select product color by dist to matrix over mde by phase
fig, axes = plt.subplots(len(doses) // 2, 2, figsize=(10, len(doses) // 2 * 5))
for i, dose in enumerate(doses):
    ax = axes[i // 2, i % 2]
    prod = sel_prod + "_" + dose
    baseline_vehicle_dists = baseline_dist_array.sel(
        sample_x="Vehicle_0", sample_y=prod
    )
    cluster_dim_name = baseline_vehicle_dists.dims[0]
    dist_to_vehicle = np.zeros(adata.shape[0])
    for cluster in baseline_vehicle_dists[cluster_dim_name].unique():
        dist_to_vehicle[
            adata.obs[adata.uns["cluster_key"]] == cluster
        ] = baseline_vehicle_dists.loc[cluster].values
    adata.obs[f"{prod}_{baseline_method_name}_dist_to_vehicle"] = dist_to_vehicle
    sc.pl.embedding(
        adata,
        "U_mde",
        color=f"{prod}_{baseline_method_name}_dist_to_vehicle",
        show=False,
        ax=ax,
        vmin=0.3,
        vmax=0.5,
    )
plt.savefig(
    os.path.join(
        base_dir_path,
        f"notebooks/figures/{sel_prod}_{baseline_method_name}_full_dist.png",
    ),
)
# %%
# Check relationship between abundance of cells per sample and distribution of dist to vehicle
num_cells_per_sample = adata.obs["product_dose"].value_counts()
mean_dist_to_vehicle = dists_from_vehicle_df.mean(axis=0)
var_dist_to_vehicle = dists_from_vehicle_df.var(axis=0)

# %%
# Join these series then plot
stats_per_sample_df = pd.DataFrame(
    {
        "num_cells_per_sample": num_cells_per_sample.loc[mean_dist_to_vehicle.index],
        "mean_dist_to_vehicle": mean_dist_to_vehicle,
        "var_dist_to_vehicle": var_dist_to_vehicle,
    }
)
stats_per_sample_df["dose"] = stats_per_sample_df.index.map(lambda x: x.split("_")[-1])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(
    stats_per_sample_df,
    x="num_cells_per_sample",
    y="mean_dist_to_vehicle",
    hue="dose",
    ax=axes[0],
)
sns.scatterplot(
    stats_per_sample_df,
    x="num_cells_per_sample",
    y="var_dist_to_vehicle",
    hue="dose",
    ax=axes[1],
)
plt.savefig(
    os.path.join(
        base_dir_path, f"notebooks/figures/{method_name}_mean_var_by_cells.png"
    ),
)

# %%
# Violin plots separated by dose
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.violinplot(
    data=stats_per_sample_df,
    x="dose",
    y="mean_dist_to_vehicle",
    ax=axes[0],
)
sns.violinplot(
    data=stats_per_sample_df,
    x="dose",
    y="var_dist_to_vehicle",
    ax=axes[1],
)
plt.savefig(
    os.path.join(
        base_dir_path, f"notebooks/figures/{method_name}_mean_var_by_dose.png"
    ),
)


# %%
