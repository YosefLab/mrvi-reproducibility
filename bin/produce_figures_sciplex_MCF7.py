# %%
import argparse
import shutil
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import plotnine as p9
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
import gseapy as gp
from tqdm import tqdm

from utils import load_results, perform_gsea
from tree_utils import hierarchical_clustering
from plot_utils import INCH_TO_CM, SCIPLEX_PATHWAY_CMAP, BARPLOT_CMAP

import mrvi

# Change to False if you want to run this script directly
RUN_WITH_PARSER = False
plt.rcParams["svg.fonttype"] = "none"


# %%
def parser():
    """Parse paths to results files (used by nextflow)"""
    parser = argparse.ArgumentParser(description="Analyze results of symsim_new")
    parser.add_argument("--results_paths", "--list", nargs="+")
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


# %%
if RUN_WITH_PARSER:
    args = parser()
    results_paths = args.results_paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pd.Series(results_paths).to_csv(
        os.path.join(output_dir, "path_to_intermediary_files.txt"), index=False
    )
else:
    output_dir = os.path.join("../results/sciplex_pipeline/figures")
    results_paths = set(glob.glob("../results/sciplex_pipeline/*/*.csv"))


# %%
def save_figures(filename, dataset_name):
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    plt.savefig(os.path.join(dataset_dir, filename + ".svg"), bbox_inches="tight")
    plt.savefig(
        os.path.join(dataset_dir, filename + ".png"), bbox_inches="tight", dpi=300
    )


# %%
pathway_color_map = SCIPLEX_PATHWAY_CMAP

# %%
# Representations
dataset_name = "sciplex_MCF7_simple_filtered_all_phases"
basedir = Path(output_dir).parent.parent.absolute()
all_results_files = glob.glob(os.path.join(basedir, "**"), recursive=True)
rep_results_paths = [
    x
    for x in all_results_files
    if x.startswith(
        f"/home/justin/ghrepos/mrvi-reproducibility/bin/../results/sciplex_pipeline/data/{dataset_name}"
    )
    and x.endswith(".h5ad")
]
rep_results = load_results(rep_results_paths)

# %%
mde_reps = rep_results["representations"].query("representation_type == 'MDE'")
# %%
if mde_reps.size >= 1:
    unique_reps = mde_reps.representation_name.unique()
    for rep in unique_reps:
        for color_by in ["pathway_level_1", "phase"]:
            rep_plots = mde_reps.query(f"representation_name == '{rep}'").sample(frac=1)
            if color_by == "pathway_level_1":
                palette = pathway_color_map
            else:
                palette = None
            fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
            sns.scatterplot(
                rep_plots, x="x", y="y", hue=color_by, palette=palette, ax=ax, s=3
            )
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5
            )
            ax.set_xlabel("MDE1")
            ax.set_ylabel("MDE2")
            ax.set_title(rep)

            save_figures(
                f"{rep}_{color_by}",
                dataset_name,
            )
            plt.clf()

# %%
method_names = [
    "mrvi_attention_iso_z_2_u_2",
    "mrvi_attention_iso_z_5_u_2",
    "mrvi_attention_iso_z_10_u_2",
    "mrvi_attention_iso_z_30_u_2",
    "mrvi_attention_iso_z_50_u_2",
    "mrvi_attention_iso_z_5_u_5",
    "mrvi_attention_iso_z_10_u_5",
    "mrvi_attention_iso_z_30_u_5",
    "mrvi_attention_iso_z_50_u_5",
    "mrvi_attention_iso_z_10_u_10",
    "mrvi_attention_iso_z_30_u_10",
    "mrvi_attention_iso_z_50_u_10",
    "mrvi_attention_iso_z_30_u_30",
    "mrvi_attention_iso_z_50_u_30",
    "mrvi_attention_iso_z_50_u_50",
]
# %%
# Cross method comparison plots
all_results = load_results(results_paths)
sciplex_metrics_df = all_results["sciplex_metrics"]
sciplex_metrics_df = sciplex_metrics_df[
    sciplex_metrics_df["model_name"].isin(method_names)
]

for ds in sciplex_metrics_df["dataset_name"].unique():
    dataset_dir = os.path.join(output_dir, ds)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    plot_df = sciplex_metrics_df[
        (sciplex_metrics_df["dataset_name"] == ds)
        # & (sciplex_metrics_df["leiden_1.0"].isna())
        & (
            sciplex_metrics_df["distance_type"] == "distance_matrices"
        )  # Exclude normalized matrices
    ]
    for metric in [
        "gt_silhouette_score",
        "gt_correlation_score",
        "in_product_all_dist_avg_percentile",
        "in_product_top_2_dist_avg_percentile",
    ]:
        if plot_df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(4 * INCH_TO_CM, 6 * INCH_TO_CM))
        plot_df["Model"] = plot_df["model_name"].map(
            lambda x: "u={}, z={}".format(x.split("_")[-1], x.split("_")[-3])
        )
        sns.barplot(
            data=plot_df,
            y="Model",
            x=metric,
            order=plot_df.sort_values(metric, ascending=False)["Model"].values,
            color="blue",
            ax=ax,
        )
        min_lim = plot_df[metric].min() - 0.05
        max_lim = plot_df[metric].max() + 0.05
        ax.set_xlim(min_lim, max_lim)
        save_figures(metric, ds)
        plt.clf()


# %%
# ELBO validation comparison
elbo_validation_df = rep_results["elbo_validations"]
fig, ax = plt.subplots(figsize=(4 * INCH_TO_CM, 6 * INCH_TO_CM))
elbo_validation_df["Model"] = elbo_validation_df["model_name"].map(
    lambda x: "u={}, z={}".format(x.split("_")[-1], x.split("_")[-3])
)
elbo_validation_df = elbo_validation_df.drop_duplicates(
    subset=["Model", "elbo_validation"], keep="first"
)

sns.barplot(
    data=elbo_validation_df,
    y="Model",
    x="elbo_validation",
    order=elbo_validation_df.sort_values("elbo_validation", ascending=False)[
        "Model"
    ].values,
    color="blue",
    ax=ax,
)
min_lim = elbo_validation_df["elbo_validation"].min() - 20
max_lim = elbo_validation_df["elbo_validation"].max() + 20
ax.set_xlim(min_lim, max_lim)
save_figures(f"elbo_validation_comparison", dataset_name)
plt.clf()

# %%
# Per dataset plots
cell_lines = ["MCF7"]
method_names = [
    "mrvi_attention_iso_z_10_u_5",
    "mrvi_attention_iso_z_30_u_5",
    "mrvi_attention_iso_z_30_u_10",
]
use_normalized = False
for method_name in method_names:
    for cl in cell_lines:
        dataset_name = f"sciplex_{cl}_simple_filtered_all_phases"
        dists_path = f"{dataset_name}.{method_name}.distance_matrices.nc"
        if not RUN_WITH_PARSER:
            dists_path = os.path.join(
                "../results/sciplex_pipeline/distance_matrices", dists_path
            )
        dists = xr.open_dataarray(dists_path)
        cluster_dim_name = dists.dims[0]

        adata_path = f"{dataset_name}.{method_name}.final.h5ad"
        if not RUN_WITH_PARSER:
            adata_path = os.path.join("../results/sciplex_pipeline/data", adata_path)
        adata = sc.read(adata_path)

        sample_to_pathway = (
            adata.obs[["product_dose", "pathway_level_1"]]
            .drop_duplicates()
            .set_index("product_dose")["pathway_level_1"]
            .to_dict()
        )
        sample_to_color_df = (
            dists.sample_x.to_series().map(sample_to_pathway).map(pathway_color_map)
        )

        sample_to_dose = (
            adata.obs[["product_dose", "dose"]]
            .drop_duplicates()
            .set_index("product_dose")["dose"]
            .fillna(0.0)
            .map(lambda x: cm.get_cmap("viridis", 256)(np.log10(x) / 4))
        )

        color_cols = [
            sample_to_color_df,
            sample_to_dose,
        ]
        col_names = [
            "Pathway",
            "Dose",
        ]
        full_col_colors_df = pd.concat(
            color_cols,
            axis=1,
        )
        full_col_colors_df.columns = col_names

        # Pathway annotated clustermap filtered down to the same product doses
        for cluster in dists[cluster_dim_name].values:
            unlike_thresh = np.percentile(dists.sel(sample_x="Vehicle_0").values, 80)
            vehicle_unlike_samples = dists.coords["sample_y"][
                dists.loc[cluster].sel(sample_x="Vehicle_0") > unlike_thresh
            ].values.tolist() + ["Vehicle_0"]
            full_col_colors_df["Retained for analysis"] = full_col_colors_df.index.isin(
                vehicle_unlike_samples
            )
            full_col_colors_df["Retained for analysis"] = full_col_colors_df[
                "Retained for analysis"
            ].map({True: "red", False: "black"})

            # unnormalized version
            unnormalized_vmax = np.percentile(dists.values, 90)
            g_dists = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=dists.sample_x,
                    sample_y=dists.sample_y,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=unnormalized_vmax,
                figsize=(20, 20),
            )
            g_dists.ax_heatmap.set_xticklabels(
                g_dists.ax_heatmap.get_xmajorticklabels(), fontsize=2
            )
            g_dists.ax_heatmap.set_yticklabels(
                g_dists.ax_heatmap.get_ymajorticklabels(), fontsize=2
            )

            handles = [
                Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map
            ]
            product_legend = plt.legend(
                handles,
                pathway_color_map,
                title="Product Name",
                bbox_to_anchor=(1, 0.9),
                bbox_transform=plt.gcf().transFigure,
                loc="upper right",
            )
            plt.gca().add_artist(product_legend)
            save_figures(
                f"{cluster}.{method_name}.distance_matrices_heatmap", dataset_name
            )
            plt.clf()

            g = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=vehicle_unlike_samples,
                    sample_y=vehicle_unlike_samples,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=unnormalized_vmax,
            )
            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xmajorticklabels(), fontsize=2
            )
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_ymajorticklabels(), fontsize=2
            )

            handles = [
                Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map
            ]
            product_legend = plt.legend(
                handles,
                pathway_color_map,
                title="Product Name",
                bbox_to_anchor=(1, 0.9),
                bbox_transform=plt.gcf().transFigure,
                loc="upper right",
            )
            plt.gca().add_artist(product_legend)
            save_figures(
                f"{cluster}.{method_name}.vehicleunlike_{'normalized_' if use_normalized else ''}distance_matrices_heatmap",
                dataset_name,
            )
            plt.clf()


# %%
#######################################
# Final distance matrix and DE analysis
#######################################
cl = "MCF7"
method_name = "mrvi_attention_iso_z_30_u_10"

dataset_name = f"sciplex_{cl}_simple_filtered_all_phases"
dists_path = f"{dataset_name}.{method_name}.distance_matrices.nc"
if not RUN_WITH_PARSER:
    dists_path = os.path.join(
        "../results/sciplex_pipeline/distance_matrices", dists_path
    )
dists = xr.open_dataarray(dists_path)
cluster_dim_name = dists.dims[0]

adata_path = f"{dataset_name}.{method_name}.final.h5ad"
if not RUN_WITH_PARSER:
    adata_path = os.path.join("../results/sciplex_pipeline/data", adata_path)
adata = sc.read(adata_path)

sample_to_pathway = (
    adata.obs[["product_dose", "pathway_level_1"]]
    .drop_duplicates()
    .set_index("product_dose")["pathway_level_1"]
    .to_dict()
)
sample_to_color_df = (
    dists.sample_x.to_series().map(sample_to_pathway).map(pathway_color_map)
)

sample_to_dose = (
    adata.obs[["product_dose", "dose"]]
    .drop_duplicates()
    .set_index("product_dose")["dose"]
    .fillna(0.0)
    .map(lambda x: cm.get_cmap("viridis", 256)(np.log10(max(x, 1)) / 4))
)

color_cols = [
    sample_to_color_df,
    sample_to_dose,
]
col_names = [
    "Pathway",
    "Dose",
]
full_col_colors_df = pd.concat(
    color_cols,
    axis=1,
)
full_col_colors_df.columns = col_names

# sig_samples = adata.obs[
#     (adata.obs[f"{cl}_deg_product_dose"] == "True")
#     | (adata.obs["product_name"] == "Vehicle")
# ]["product_dose"].unique()
unlike_thresh = np.percentile(dists.sel(sample_x="Vehicle_0").values, 80)
sig_samples = dists.coords["sample_y"][
    dists.loc[1].sel(sample_x="Vehicle_0") > unlike_thresh
].values.tolist() + ["Vehicle_0"]
d1 = dists.loc[1].sel(
    sample_x=sig_samples,
    sample_y=sig_samples,
)
vmax = np.percentile(dists.values, 90)
Z = hierarchical_clustering(d1.values, method="ward", return_ete=False)
g = sns.clustermap(
    d1.to_pandas(),
    yticklabels=True,
    xticklabels=True,
    row_linkage=Z,
    col_linkage=Z,
    row_colors=full_col_colors_df,
    vmin=0,
    vmax=vmax,
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=2)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=2)

handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
product_legend = plt.legend(
    handles,
    pathway_color_map,
    title="Product Name",
    bbox_to_anchor=(1, 0.9),
    bbox_transform=plt.gcf().transFigure,
    loc="upper right",
)
plt.gca().add_artist(product_legend)
save_figures(
    f"{method_name}.distances_fig",
    dataset_name,
)
# plt.clf()
# %%
train_adata_path = f"{dataset_name}.preprocessed.h5ad"
if not RUN_WITH_PARSER:
    train_adata_path = os.path.join(
        "../results/sciplex_pipeline/data", train_adata_path
    )
train_adata = sc.read(train_adata_path)
# %%
model_path = f"{dataset_name}.{method_name}"
if not RUN_WITH_PARSER:
    model_path = os.path.join("../results/sciplex_pipeline/models", model_path)
# Register adata to get scvi sample assignment
model = mrvi.MrVI.load(model_path, adata=train_adata, accelerator="cpu")

# %%
import pymde

mde_kwargs = dict(
    embedding_dim=2,
    constraint=pymde.Standardized(),
    repulsive_fraction=1.5,
    device="cuda",
    n_neighbors=15,
)
for latent_key in ["z", "u"]:
    rep = f"X_{method_name}_{latent_key}_mde"
    latent_obsm_key = f"X_{method_name}_{latent_key}"
    latent = adata.obsm[latent_obsm_key]
    latent_mde = pymde.preserve_neighbors(latent, **mde_kwargs).embed().cpu().numpy()
    adata.obsm[rep] = latent_mde
    rep_plots = mde_reps.query(f"representation_name == '{rep}'")
    rep_plots.loc[(rep_plots["index"] == adata.obs_names), ["new_x", "new_y"]] = (
        adata.obsm[rep]
    )
    rep_plots = rep_plots.sample(frac=1)
    for color_by in ["phase", "pathway_level_1"]:
        if color_by == "phase":
            palette = None
        elif color_by == "pathway_level_1":
            palette = pathway_color_map
        fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
        sns.scatterplot(
            rep_plots, x="new_x", y="new_y", hue=color_by, palette=palette, ax=ax, s=3
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
        ax.set_xlabel("MDE1")
        ax.set_ylabel("MDE2")
        ax.set_title(rep)
        save_figures(
            f"{rep}_{color_by}.final",
            dataset_name,
        )
        plt.clf()

    # Compute UMAPs
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=latent_obsm_key)
    sc.tl.umap(adata)
    rep = f"X_{method_name}_{latent_key}_umap"
    umap_rep = "X_umap"
    adata.obs[f"{rep}_1"] = adata.obsm[umap_rep][:, 0]
    adata.obs[f"{rep}_2"] = adata.obsm[umap_rep][:, 1]
    for color_by in ["phase", "pathway_level_1"]:
        if color_by == "phase":
            palette = None
        elif color_by == "pathway_level_1":
            palette = pathway_color_map
        fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
        sns.scatterplot(
            adata.obs,
            x=f"{rep}_1",
            y=f"{rep}_2",
            hue=color_by,
            palette=palette,
            ax=ax,
            s=3,
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP with {color_by}")
        save_figures(
            f"{rep}_{color_by}.final",
            dataset_name,
        )
        plt.clf()

# %%
# Metric plots

# Compute random silhouette baseline and avg percentile baseline
gt_cluster_labels_df = None
gt_clusters_path = f"../data/l1000_signatures/{cl}_cluster_labels.csv"
gt_cluster_labels_df = pd.read_csv(gt_clusters_path, index_col=0)
# Assign them all at 10000 nM dose
new_sample_idx = [prod + "_10000" for prod in list(gt_cluster_labels_df.index)]
gt_cluster_labels_df.index = new_sample_idx
# Filter on samples in the distance matrix
gt_cluster_labels_df = gt_cluster_labels_df.loc[
    np.intersect1d(
        dists.coords["sample_x"].data,
        gt_cluster_labels_df.index.array,
    )
]

dist_inferred = (
    dists.loc[1]
    .sel(
        sample_x=gt_cluster_labels_df.index,
        sample_y=gt_cluster_labels_df.index,
    )
    .values
)
np.fill_diagonal(dist_inferred, 0)
# set seed
np.random.seed(45)
random_asws = []
for i in range(100):
    random_asw = silhouette_score(
        dist_inferred,
        gt_cluster_labels_df.sample(frac=1).values.ravel(),
        metric="precomputed",
    )
    random_asw = (random_asw + 1) / 2
    random_asws.append(random_asw)
mean_random_asw = np.mean(random_asws)
print(mean_random_asw)

# %%
import scipy

all_products = set()
all_doses = set()
for sample_name in dists.sample_x.data:
    product_name, dose = sample_name.split("_")
    if product_name != "Vehicle":
        all_products.add(product_name)
    if dose != "0":
        all_doses.add(dose)
in_product_all_dist_avg_percentile = []
top_two_doses = ["1000", "10000"]
cluster_dists = dists.loc[1]
cluster_dists_arr = cluster_dists.data
non_diag_mask = (
    np.ones(shape=cluster_dists_arr.shape) - np.identity(cluster_dists_arr.shape[0])
).astype(bool)
in_prod_mask = np.zeros(shape=cluster_dists_arr.shape, dtype=bool)
for product_name in all_products:
    for dosex in all_doses:
        for dosey in all_doses:
            if dosex == dosey:
                continue
            dosex_idx = np.where(
                cluster_dists.sample_x.data == f"{product_name}_{dosex}"
            )[0]
            if len(dosex_idx) == 0:
                continue
            dosey_idx = np.where(
                cluster_dists.sample_y.data == f"{product_name}_{dosey}"
            )[0]
            if len(dosey_idx) == 0:
                continue
            in_prod_mask[dosex_idx[0], dosey_idx[0]] = True
# %%
# shuffle the mask
np.random.seed(45)
random_in_prod_all_dist_avg_percentiles = []
for _ in range(100):
    shuffled_in_prod_mask = np.zeros(shape=in_prod_mask.shape, dtype=bool)
    shuffled_in_prod_mask[np.triu_indices(in_prod_mask.shape[0])] = (
        np.random.permutation(in_prod_mask[np.triu_indices(in_prod_mask.shape[0])])
    )
    shuffled_in_prod_mask = shuffled_in_prod_mask + shuffled_in_prod_mask.T

    adjusted_ranks = (
        scipy.stats.rankdata(cluster_dists_arr).reshape(cluster_dists_arr.shape)
        - cluster_dists_arr.shape[0]
    )
    shuffled_in_prod_all_dist_avg_percentile = (
        adjusted_ranks[shuffled_in_prod_mask].mean() / non_diag_mask.sum()
    )
    random_in_prod_all_dist_avg_percentiles.append(
        shuffled_in_prod_all_dist_avg_percentile
    )
print(np.mean(random_in_prod_all_dist_avg_percentiles))

# %%
all_results = load_results(results_paths)
sciplex_metrics_df = all_results["sciplex_metrics"]

plot_df = sciplex_metrics_df[
    (sciplex_metrics_df["dataset_name"] == dataset_name)
    & (sciplex_metrics_df["leiden_1.0"].isna())
    & (
        sciplex_metrics_df["distance_type"] == "distance_matrices"
    )  # Exclude normalized matrices
]
model_to_method_name_mapping = {
    method_name: "mrVI",
    "composition_SCVI_clusterkey_subleiden1": "CompositionSCVI",
    "composition_PCA_clusterkey_subleiden1": "CompositionPCA",
}
# select rows then color w cmap + baseline
plot_df = plot_df[plot_df["model_name"].isin(model_to_method_name_mapping.keys())]
plot_df["method_name"] = plot_df["model_name"].map(model_to_method_name_mapping)
for random_asw in random_asws:
    plot_df = pd.concat(
        (
            plot_df,
            pd.DataFrame(
                {"method_name": ["Random"], "gt_silhouette_score": [random_asw]}
            ),
        ),
        axis=0,
    )

metric = "gt_silhouette_score"

fig, ax = plt.subplots(figsize=(6 * INCH_TO_CM, 6 * INCH_TO_CM))
sns.barplot(
    data=plot_df,
    y="method_name",
    x=metric,
    order=["mrVI", "CompositionSCVI", "CompositionPCA", "Random"],
    palette=BARPLOT_CMAP,
    ax=ax,
)
min_lim = plot_df[metric].min() - 0.05
max_lim = plot_df[metric].max() + 0.05
ax.set_xlim(min_lim, max_lim)
ax.set_xticks([0.35, 0.4, 0.45, 0.5, 0.55])
save_figures(f"{metric}_final", dataset_name)

# %%
plot_df = plot_df[plot_df["model_name"].isin(model_to_method_name_mapping.keys())]
plot_df["method_name"] = plot_df["model_name"].map(model_to_method_name_mapping)
for shuffled_in_prod_all_dist_avg_percentile in random_in_prod_all_dist_avg_percentiles:
    plot_df = pd.concat(
        (
            plot_df,
            pd.DataFrame(
                {
                    "method_name": ["Random"],
                    "in_product_all_dist_avg_percentile": [
                        shuffled_in_prod_all_dist_avg_percentile
                    ],
                }
            ),
        ),
        axis=0,
    )

metric = "in_product_all_dist_avg_percentile"

fig, ax = plt.subplots(figsize=(6 * INCH_TO_CM, 6 * INCH_TO_CM))
sns.barplot(
    data=plot_df,
    y="method_name",
    x=metric,
    order=["mrVI", "CompositionSCVI", "CompositionPCA", "Random"],
    palette=BARPLOT_CMAP,
    ax=ax,
)
min_lim = plot_df[metric].min() - 0.03
max_lim = plot_df[metric].max() + 0.03
ax.set_xlim(min_lim, max_lim)
ax.set_xticks([0.3, 0.35, 0.4, 0.45, 0.5])
save_figures(f"{metric}_final", dataset_name)

# %%
# Silhouette analysis
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

# Define the range of clusters to test
n_clusters_range = range(2, 30)

silhouette_scores = []

for n_clusters in n_clusters_range:
    # Generate clusters
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Calculate silhouette score
    score = silhouette_score(d1.to_pandas(), clusters)
    silhouette_scores.append(score)

    print(f"Silhouette Score for {n_clusters} clusters: {score}")

# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores, marker="o")
plt.xticks(n_clusters_range)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.grid(True)
save_figures("silhouette_score", dataset_name)
# %%
# Sum of Squared Differences within each cluster
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import pairwise_distances
import numpy as np

# Define the range of clusters to test
n_clusters_range = range(2, 30)

ssd_scores = []

for n_clusters in n_clusters_range:
    # Generate clusters
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Calculate sum of squared differences within each cluster
    ssd = 0
    for cluster_id in np.unique(clusters):
        cluster_points = d1.to_pandas()[clusters == cluster_id]
        pairwise_dist = pairwise_distances(cluster_points)
        ssd += (
            np.sum(np.square(pairwise_dist)) / 2
        )  # Divide by 2 to correct for double counting
    ssd_scores.append(ssd)

    print(f"Sum of Squared Differences for {n_clusters} clusters: {ssd}")

# Plotting the SSD scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, ssd_scores, marker="o")
plt.xticks(n_clusters_range)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Differences")
plt.title("SSD by Number of Clusters")
plt.grid(True)
save_figures("ssd_comparison", dataset_name)

# %%
# DE analysis
plt.rcParams["axes.grid"] = False

n_clusters = 6

clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
donor_info_ = pd.DataFrame({"cluster_id": clusters}, index=d1.sample_x.values)
vehicle_cluster = f"Cluster {donor_info_.loc['Vehicle_0']['cluster_id']}"

# %%
# Viz clusters
cluster_color_map = {i: c for i, c in enumerate(sns.color_palette("tab10", 10))}
cluster_colors = donor_info_.cluster_id.map(cluster_color_map)
g = sns.clustermap(
    d1.to_pandas(),
    yticklabels=True,
    xticklabels=True,
    row_linkage=Z,
    col_linkage=Z,
    row_colors=cluster_colors,
    vmin=0,
    vmax=vmax,
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=5)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=5)

save_figures(f"{method_name}.distances_fig.clustered", dataset_name)

# %%
# MDE with low opacity vehicle cluster
# vehicle_sim_thresh = 0.4
vehicle_sim_thresh = unlike_thresh
vehicle_dists = (
    dists.where(dists.values < vehicle_sim_thresh)
    .sel(sample_x="Vehicle_0", _dummy_name=1)
    .to_pandas()
)
vehicle_sim_samples = vehicle_dists[~vehicle_dists.isna()].index.to_list()
sns.histplot(dists.sel(sample_x="Vehicle_0").to_numpy().flatten(), bins=50)
plt.axvline(vehicle_sim_thresh, color="red")

# %%
rep = f"X_{method_name}_u_mde"
rep_plots = mde_reps.query(f"representation_name == '{rep}'")
rep_plots.loc[(rep_plots["index"] == adata.obs_names), ["new_x", "new_y"]] = adata.obsm[
    rep
]
rep_plots = rep_plots.sample(frac=1)
color_by = "pathway_level_1"
palette = pathway_color_map
marker_size = 3
fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
full_opacity_rep_plots = rep_plots[~rep_plots["product_dose"].isin(vehicle_sim_samples)]
partial_opacity_rep_plots = rep_plots[
    rep_plots["product_dose"].isin(vehicle_sim_samples)
]
sns.scatterplot(
    partial_opacity_rep_plots,
    x="new_x",
    y="new_y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    alpha=0.1,
)
sns.scatterplot(
    full_opacity_rep_plots,
    x="new_x",
    y="new_y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    legend=False,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")

save_figures(
    f"{rep}_{color_by}_opacity_thresh_{vehicle_sim_thresh}",
    dataset_name,
)
rep = f"X_{method_name}_z_mde"
rep_plots = mde_reps.query(f"representation_name == '{rep}'")
rep_plots.loc[(rep_plots["index"] == adata.obs_names), ["new_x", "new_y"]] = adata.obsm[
    rep
]
rep_plots = rep_plots.sample(frac=1)
color_by = "pathway_level_1"
palette = pathway_color_map
marker_size = 3
fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
full_opacity_rep_plots = rep_plots[~rep_plots["product_dose"].isin(vehicle_sim_samples)]
partial_opacity_rep_plots = rep_plots[
    rep_plots["product_dose"].isin(vehicle_sim_samples)
]
sns.scatterplot(
    partial_opacity_rep_plots,
    x="new_x",
    y="new_y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    alpha=0.1,
)
sns.scatterplot(
    full_opacity_rep_plots,
    x="new_x",
    y="new_y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    legend=False,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")

save_figures(
    f"{rep}_{color_by}_opacity_thresh_{vehicle_sim_thresh}",
    dataset_name,
)

# %%
# Compute UMAPs for umap representation
for latent_key in ["u", "z"]:
    rep = f"X_{method_name}_{latent_key}_umap"
    color_by = "pathway_level_1"
    palette = pathway_color_map
    rep_plots = adata.obs[[f"{rep}_1", f"{rep}_2", "product_dose", color_by]]
    rep_plots = rep_plots.sample(frac=1)
    full_opacity_rep_plots = rep_plots[
        ~rep_plots["product_dose"].isin(vehicle_sim_samples)
    ]
    partial_opacity_rep_plots = rep_plots[
        rep_plots["product_dose"].isin(vehicle_sim_samples)
    ]
    fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
    sns.scatterplot(
        partial_opacity_rep_plots,
        x=f"{rep}_1",
        y=f"{rep}_2",
        hue=color_by,
        palette=palette,
        ax=ax,
        s=marker_size,
        alpha=0.1,
    )
    sns.scatterplot(
        full_opacity_rep_plots,
        x=f"{rep}_1",
        y=f"{rep}_2",
        hue=color_by,
        palette=palette,
        ax=ax,
        s=marker_size,
        legend=False,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    save_figures(
        f"{rep}_{color_by}",
        dataset_name,
    )


# %%
# Multivariate analysis DE
# (For this we create a column for each cluster since we require float values)
model_path = f"{dataset_name}.{method_name}"
if not RUN_WITH_PARSER:
    model_path = os.path.join("../results/sciplex_pipeline/models", model_path)
# Register adata to get scvi sample assignment
model = mrvi.MrVI.load(model_path, adata=train_adata, accelerator="cuda")

# %%
train_adata.obs["donor_cluster"] = (
    train_adata.obs["product_dose"].map(donor_info_["cluster_id"]).values.astype(int)
)
train_adata.obs.loc[train_adata.obs.product_dose == "Vehicle_0", "donor_cluster"] = -1
for cluster_i in range(1, n_clusters + 1):
    train_adata.obs[f"donor_cluster_{cluster_i}"] = (
        train_adata.obs["donor_cluster"] == cluster_i
    ).astype(int)
obs_df = train_adata.obs.copy()
obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
model.donor_info = obs_df.set_index("_scvi_sample").sort_index()
sub_train_adata = train_adata[train_adata.obs["donor_cluster"] != "nan"]
sub_train_adata.obs["_indices"] = np.arange(sub_train_adata.shape[0])

# %%
cluster_wise_multivar_res = {}
for cluster_i in range(1, n_clusters + 1):
    cluster_sub_train_adata = sub_train_adata[
        (sub_train_adata.obs["donor_cluster"] == cluster_i)
        | (sub_train_adata.obs["donor_cluster"] == -1)
    ].copy()
    # cluster_sub_train_adata = cluster_sub_train_adata[np.random.choice(cluster_sub_train_adata.shape[0], 50)]
    cluster_multivar_res = model.perform_multivariate_analysis(
        cluster_sub_train_adata,
        donor_keys=[f"donor_cluster_{cluster_i}"],
        batch_size=32,
        store_lfc=True,
    )
    cluster_wise_multivar_res[cluster_i] = cluster_multivar_res

# %%
# save cluster-wise results
import pickle

with open(
    os.path.join(
        output_dir, f"{dataset_name}.{method_name}.cluster_wise_multivar_res.pkl"
    ),
    "wb",
) as f:
    pickle.dump(cluster_wise_multivar_res, f)

# %%
import pickle

# read cluster-wise results
with open(
    os.path.join(
        output_dir, f"{dataset_name}.{method_name}.cluster_wise_multivar_res.pkl"
    ),
    "rb",
) as f:
    cluster_wise_multivar_res = pickle.load(f)

# %%
# GSEA for DE genes
# Load gene sets
gene_set_name = "MSigDB_Hallmark_2020"
lfc_thresh = 1
plt_vmax = 2

gene_sets = [gp.parser.download_library(gene_set_name, "human")]
# import json

# with open("../data/MSigDB_GTRD.json", "r") as f:
#     out = json.load(f)
#     gene_sets = {k: v["geneSymbols"] for k, v in out.items()}
# %%

enr_result_dict = {}
full_dfs = {}
de_dfs = {}
for cluster_i in cluster_wise_multivar_res:
    cluster_multivar_res = cluster_wise_multivar_res[cluster_i]
    betas_ = (
        cluster_multivar_res["lfc"]
        .transpose("cell_name", "covariate", "gene")
        .loc[{"covariate": f"donor_cluster_{cluster_i}"}]
        .values
    )
    betas_ = betas_ / np.log(2)  # change to log 2

    lfc_df = pd.DataFrame(
        {
            "LFC": betas_.mean(0),
            "LFC_std": betas_.std(0),
            "gene": sub_train_adata.var_names,
            "gene_index": np.arange(sub_train_adata.shape[1]),
        }
    ).assign(absLFC=lambda x: np.abs(x.LFC))
    full_dfs[cluster_i] = lfc_df

    cond = lfc_df.absLFC > lfc_thresh
    obs_de = lfc_df.loc[cond, :].reset_index(drop=True)
    obs_de.LFC.hist(bins=100)
    de_dfs[cluster_i] = obs_de

    de_genes = obs_de.gene.values
    de_genes = [gene for gene in de_genes if str(gene) != "nan"]
    if len(de_genes) > 0:
        try:
            enr_results, fig = perform_gsea(
                de_genes, gene_sets=gene_sets, plot=True, use_server=False
            )
            enr_result_dict[cluster_i] = enr_results
        except ValueError as e:
            print(e)

    # directional gsea
    up_cond = lfc_df.LFC > lfc_thresh
    down_cond = lfc_df.LFC < -lfc_thresh
    up_obs_de = lfc_df.loc[up_cond, :]
    down_obs_de = lfc_df.loc[down_cond, :]

    up_de_genes = up_obs_de.gene.values
    up_de_genes = [gene for gene in up_de_genes if str(gene) != "nan"]
    if len(up_de_genes) > 0:
        try:
            up_enr_results, up_fig = perform_gsea(
                up_de_genes,
                gene_sets=gene_sets,
                plot=True,
                use_server=False,
            )
            enr_result_dict[f"{cluster_i}_up"] = up_enr_results
        except ValueError as e:
            print(f"Up GSEA Error: {e}")

    down_de_genes = down_obs_de.gene.values
    down_de_genes = [gene for gene in down_de_genes if str(gene) != "nan"]
    if len(down_de_genes) > 0:
        try:
            down_enr_results, down_fig = perform_gsea(
                down_de_genes,
                gene_sets=gene_sets,
                plot=True,
                use_server=False,
            )
            enr_result_dict[f"{cluster_i}_down"] = down_enr_results
        except ValueError as e:
            print(f"Down GSEA Error: {e}")


# %%
enr_pval_df_records = []
for cluster_idx in range(1, n_clusters + 1):
    if cluster_idx not in enr_result_dict:
        enr_pval_df_records.append({"cluster_idx": cluster_idx})
    else:
        enr_cluster_results = enr_result_dict[cluster_idx]
        enr_pval_df_records.append(
            {
                "cluster_idx": cluster_idx,
                **enr_cluster_results.pivot(
                    index="Gene_set", columns="Term", values="Significance score"
                )
                .iloc[0]
                .to_dict(),
            }
        )
enr_pval_df = pd.DataFrame.from_records(enr_pval_df_records, index="cluster_idx")
enr_pval_df.fillna(0, inplace=True)

# %%
# Plot GSEA heatmap
filtered_enr_pval_df = enr_pval_df.loc[
    :, (enr_pval_df > -np.log10(0.05)).values.any(axis=0)
]
sns.clustermap(
    filtered_enr_pval_df.T,
    col_cluster=False,
    yticklabels=True,
    xticklabels=True,
    vmin=0,
    vmax=plt_vmax,
    cmap="Reds",
)
save_figures(
    f"multivar_gsea_heatmap.{gene_set_name}",
    dataset_name,
)
# %%
# Up direction GSEA
up_enr_pval_df_records = []
for cluster_idx in range(1, n_clusters + 1):
    up_cluster_idx = f"{cluster_idx}_up"
    if up_cluster_idx not in enr_result_dict:
        up_enr_pval_df_records.append({"cluster_idx": cluster_idx})
    else:
        up_enr_cluster_results = enr_result_dict[up_cluster_idx]
        up_enr_pval_df_records.append(
            {
                "cluster_idx": cluster_idx,
                **up_enr_cluster_results.pivot(
                    index="Gene_set", columns="Term", values="Significance score"
                )
                .iloc[0]
                .to_dict(),
            }
        )
up_enr_pval_df = pd.DataFrame.from_records(up_enr_pval_df_records, index="cluster_idx")
up_enr_pval_df.fillna(0, inplace=True)

# %%
# Plot up GSEA heatmap
filtered_up_enr_pval_df = up_enr_pval_df.loc[
    :, (up_enr_pval_df > -np.log10(0.05)).values.any(axis=0)
]
sns.clustermap(
    filtered_up_enr_pval_df.T,
    col_cluster=False,
    yticklabels=True,
    xticklabels=True,
    vmin=0,
    vmax=plt_vmax,
    cmap="Reds",
)
save_figures(
    f"multivar_gsea_heatmap.{gene_set_name}.up",
    dataset_name,
)

# %%
# Down direction GSEA
down_enr_pval_df_records = []
for cluster_idx in range(1, n_clusters + 1):
    down_cluster_idx = f"{cluster_idx}_down"
    if down_cluster_idx not in enr_result_dict:
        down_enr_pval_df_records.append({"cluster_idx": cluster_idx})
    else:
        down_enr_cluster_results = enr_result_dict[down_cluster_idx]
        down_enr_pval_df_records.append(
            {
                "cluster_idx": cluster_idx,
                **down_enr_cluster_results.pivot(
                    index="Gene_set", columns="Term", values="Significance score"
                )
                .iloc[0]
                .to_dict(),
            }
        )
down_enr_pval_df = pd.DataFrame.from_records(
    down_enr_pval_df_records, index="cluster_idx"
)
down_enr_pval_df.fillna(0, inplace=True)

# %%
# Plot down GSEA heatmap
filtered_down_enr_pval_df = down_enr_pval_df.loc[
    :, (down_enr_pval_df > -np.log10(0.05)).values.any(axis=0)
]
sns.clustermap(
    filtered_down_enr_pval_df.T,
    col_cluster=False,
    yticklabels=True,
    xticklabels=True,
    vmin=0,
    vmax=plt_vmax,
    cmap="Reds",
)
save_figures(
    f"multivar_gsea_heatmap.{gene_set_name}.down",
    dataset_name,
)

# %%
# Plot up and down together
from matplotlib.tri import Triangulation

# Ensure the columns of up_enr_pval_df and down_enr_pval_df are consistent
all_columns = set(up_enr_pval_df.columns).union(set(down_enr_pval_df.columns))
for col in all_columns:
    if col not in up_enr_pval_df:
        up_enr_pval_df[col] = 0
    if col not in down_enr_pval_df:
        down_enr_pval_df[col] = 0

# Reorder the columns to make sure they are consistent between the two dataframes
up_enr_pval_df = up_enr_pval_df.reindex(columns=sorted(all_columns))
down_enr_pval_df = down_enr_pval_df.reindex(columns=sorted(all_columns))

significant_columns = set(
    up_enr_pval_df.columns[(up_enr_pval_df > -np.log10(0.05)).any()]
).union(down_enr_pval_df.columns[(down_enr_pval_df > -np.log10(0.05)).any()])

up_enr_pval_df = up_enr_pval_df[significant_columns]
down_enr_pval_df = down_enr_pval_df[significant_columns]

# Perform hierarchical clustering on columns and reorder dataframes based on the clustering result
from scipy.cluster.hierarchy import linkage, leaves_list

# Combine both up and down dataframes for clustering
combined_df = up_enr_pval_df + down_enr_pval_df

# Perform hierarchical clustering on columns
linked_cols = linkage(combined_df.T, method="average", metric="euclidean")
col_order = leaves_list(linked_cols)

# Reorder columns based on hierarchical clustering
up_enr_pval_df = up_enr_pval_df.iloc[:, col_order]
down_enr_pval_df = down_enr_pval_df.iloc[:, col_order]


M = up_enr_pval_df.shape[0]
N = up_enr_pval_df.shape[1]
x = np.arange(M + 1)
y = np.arange(N + 1)
xs, ys = np.meshgrid(x, y)

triangles1 = [
    (i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1))
    for j in range(N)
    for i in range(M)
]
triangles2 = [
    (i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1))
    for j in range(N)
    for i in range(M)
]
triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)
img1 = plt.tripcolor(
    triang1,
    down_enr_pval_df.T.values.flatten(),
    cmap="Blues",
    vmax=plt_vmax,
    vmin=0,
)
img2 = plt.tripcolor(
    triang2,
    up_enr_pval_df.T.values.flatten(),
    cmap="Reds",
    vmax=plt_vmax,
    vmin=0,
)

plt.colorbar(img2, ticks=range(10), pad=-0.05)
plt.colorbar(img1, ticks=range(10))
plt.xlim(x[0] - 0.5, x[-1] - 0.5)
plt.ylim(y[0] - 0.5, y[-1] - 0.5)
plt.xticks(
    ticks=np.arange(up_enr_pval_df.shape[0]),
    labels=up_enr_pval_df.index.values,
    rotation=0,
)
plt.yticks(ticks=np.arange(len(up_enr_pval_df.columns)), labels=up_enr_pval_df.columns)
save_figures(
    f"multivar_gsea_heatmap.{gene_set_name}.up_down",
    dataset_name,
)

# %%
# Cluster wise barplots
top_de_genes_per_cluster = {}
for cluster_i, de_df in de_dfs.items():
    if de_df.shape[0] == 0:
        continue
    de_df.sort_values("absLFC", ascending=False, inplace=True)
    abr_de_df = de_df[:20]
    top_de_genes_per_cluster[cluster_i] = abr_de_df[:50].gene.values
    abr_de_df.sort_values("LFC", ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
    sns.barplot(
        x="gene",
        y="LFC",
        color="blue",
        data=abr_de_df,
        ax=ax,
    )
    ax.axhline(y=0, color="black")
    # rotate x labels
    ax.set_title(f"Cluster {cluster_i} Top LFC DE Genes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
    save_figures(
        f"cluster_{cluster_i}_top_lfc_de_genes.{gene_set_name}",
        dataset_name,
    )

# %%
# matrixplot of top de genes
from functools import reduce

top_de_genes = reduce(np.union1d, top_de_genes_per_cluster.values())

# get lfcs for each cluster for top de genes
top_de_lfcs_cols = []
for cluster_i, full_df in full_dfs.items():
    print(cluster_i)
    top_des_df = full_df[full_df["gene"].isin(top_de_genes)][["gene", "LFC"]].set_index(
        "gene"
    )
    top_de_lfcs_cols.append(top_des_df.loc[top_de_genes]["LFC"].values.reshape(-1, 1))

top_de_lfcs_df = pd.DataFrame(
    np.hstack(top_de_lfcs_cols), index=top_de_genes, columns=full_dfs.keys()
)
top_de_lfcs_df

# %%
sns.clustermap(
    top_de_lfcs_df,
    col_cluster=False,
    yticklabels=True,
    xticklabels=True,
    center=0,
    cmap="seismic",
)
save_figures(
    f"top_de_lfcs_clustermap.{gene_set_name}",
    dataset_name,
)

# %%
# cell-wise dist matrix analysis
dists_sub_adata = model.adata[np.random.choice(model.adata.shape[0], 20000)]
cell_dists = model.get_local_sample_distances(dists_sub_adata)

# %%
# PCA of upper triangular matrix of distance matrices
from sklearn.decomposition import PCA

cell_dists_array = cell_dists.cell.data
triu_i, triu_j = np.triu_indices(cell_dists_array.shape[1], k=1)
triu_cell_dists_array = cell_dists_array[:, triu_i, triu_j]

# %%
pca = PCA(n_components=20)
pca.fit(triu_cell_dists_array)

# %%
# Plot variance explained
plt.bar(range(1, 21), np.cumsum(pca.explained_variance_ratio_))
# Show all x tick marks
plt.xticks(range(1, 21))
plt.xlabel("Number of Principal Components")
plt.ylabel("Proportion of Variance Explained")
save_figures("triu_dist_pca_variance_explained", dataset_name)

# %%
pca_xy = pd.DataFrame(pca.transform(triu_cell_dists_array)[:, :2], columns=["x", "y"])
pca_xy.loc[:, "phase"] = (
    model.adata.obs.loc[cell_dists.cell_name.values, "phase"].astype(str).values
)
pca_xy

# %%
sns.scatterplot(pca_xy, x="x", y="y", hue="phase", legend=False, s=3)
plt.xlabel("PC1")
plt.ylabel("PC2")
save_figures("triu_dist_pca_phase", dataset_name)

# %%
sns.scatterplot(pca_xy, x="x", y="y", s=3)
plt.xlabel("PC1")
plt.ylabel("PC2")
save_figures("triu_dist_pca", dataset_name)

# %%
# umap of pca comps
from umap import UMAP

pca_all = pca.transform(triu_cell_dists_array)
umap = UMAP(n_components=2)
umap.fit(pca_all)
umap_xy = pd.DataFrame(umap.transform(pca_all), columns=["x", "y"])

# %%
sns.scatterplot(umap_xy, x="x", y="y")
save_figures("triu_dist_umap", dataset_name)

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_all)
    silhouette_avg = silhouette_score(pca_all, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# %%
# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different Cluster Numbers")
save_figures("triu_dist_silhouette_scores", dataset_name)

# %%
