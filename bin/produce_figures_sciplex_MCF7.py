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

from utils import load_results, perform_gsea
from tree_utils import hierarchical_clustering
from plot_utils import INCH_TO_CM, SCIPLEX_PATHWAY_CMAP, BARPLOT_CMAP

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
    and x.endswith(".final.h5ad")
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
            # rep_plots = mde_reps.query(
            #     f"representation_name == '{rep}' and MCF7_deg_product_dose == 'True'"
            # )
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
# Cross method comparison plots
all_results = load_results(results_paths)
sciplex_metrics_df = all_results["sciplex_metrics"]

for dataset_name in sciplex_metrics_df["dataset_name"].unique():
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    plot_df = sciplex_metrics_df[
        (sciplex_metrics_df["dataset_name"] == dataset_name)
        & (sciplex_metrics_df["leiden_1.0"].isna())
        & (
            sciplex_metrics_df["distance_type"] == "distance_matrices"
        )  # Exclude normalized matrices
    ]
    for metric in [
        "gt_silhouette_score",
        "in_product_all_dist_avg_percentile",
        "in_product_top_2_dist_avg_percentile",
    ]:
        if plot_df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(4 * INCH_TO_CM, 4 * INCH_TO_CM))
        sns.barplot(
            data=plot_df,
            y="model_name",
            x=metric,
            order=plot_df.sort_values(metric, ascending=False)["model_name"].values,
            color="blue",
            ax=ax,
        )
        min_lim = plot_df[metric].min() - 0.05
        max_lim = plot_df[metric].max() + 0.05
        ax.set_xlim(min_lim, max_lim)
        save_figures(metric, dataset_name)
        plt.clf()

# %%
# same metrics for normalized
for dataset_name in sciplex_metrics_df["dataset_name"].unique():
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    plot_df = sciplex_metrics_df[
        (sciplex_metrics_df["dataset_name"] == dataset_name)
        & (
            (
                sciplex_metrics_df["distance_type"] == "normalized_distance_matrices"
            )  # Only normalized matrices
            | (
                sciplex_metrics_df["model_name"].str.startswith("composition")
                & sciplex_metrics_df["leiden_1.0"].isna()
            )
        )
    ]
    for metric in [
        "gt_silhouette_score",
        "in_product_all_dist_avg_percentile",
        "in_product_top_2_dist_avg_percentile",
    ]:
        if plot_df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(4 * INCH_TO_CM, 4 * INCH_TO_CM))
        sns.barplot(
            data=plot_df,
            y="model_name",
            x=metric,
            order=plot_df.sort_values(metric, ascending=False)["model_name"].values,
            color="blue",
            ax=ax,
        )
        min_lim = plot_df[metric].min() - 0.05
        max_lim = plot_df[metric].max() + 0.05
        ax.set_xlim(min_lim, max_lim)
        save_figures(f"{metric}_normalized", dataset_name)
        plt.clf()


# %%
cell_lines = ["MCF7"]
method_names = [
    # "mrvi_attention_noprior",
    # "mrvi_attention_no_prior_mog",
    # "mrvi_z10",
    # "mrvi_z30",
    # "mrvi_z10_u5",
    "mrvi_z20_u5",
    "mrvi_z30_u5",
    # "mrvi_z10_u10",
    # "mrvi_z50_u5",
    # "mrvi_z100_u5",
]

# Per dataset plots
use_normalized = False
for method_name in method_names:
    for cl in cell_lines:
        dataset_name = f"sciplex_{cl}_simple_filtered_all_phases"
        normalized_dists_path = (
            f"{dataset_name}.{method_name}.normalized_distance_matrices.nc"
        )
        if not RUN_WITH_PARSER:
            normalized_dists_path = os.path.join(
                "../results/sciplex_pipeline/distance_matrices",
                normalized_dists_path,
            )
        normalized_dists = xr.open_dataarray(normalized_dists_path)
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
            normalized_dists.sample_x.to_series()
            .map(sample_to_pathway)
            .map(pathway_color_map)
        )

        if not RUN_WITH_PARSER:
            n_deg_dict = pd.read_csv(
                f"../notebooks/output/{cl}_flat_deg_dict.csv", index_col=0
            ).to_dict()["0"]
            sample_to_n_deg_df = normalized_dists.sample_x.to_series().map(n_deg_dict)
            sample_to_n_deg_df = sample_to_n_deg_df.map(
                lambda x: cm.get_cmap("viridis", 256)(x / np.max(sample_to_n_deg_df))
            )

        sample_to_sig_prod_dose = (
            adata.obs[["product_dose", f"{cl}_deg_product_dose"]]
            .drop_duplicates()
            .set_index("product_dose")[f"{cl}_deg_product_dose"]
            .fillna("False")
            .map({"True": "red", "False": "blue"})
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
            # sample_to_sig_prod_dose,
        ]
        col_names = [
            "Pathway",
            "Dose",
            # "Product-Dose passed DEG filter?",
        ]
        # if not RUN_WITH_PARSER:
        #     color_cols.append(sample_to_n_deg_df)
        #     col_names.append("n_degs")
        full_col_colors_df = pd.concat(
            color_cols,
            axis=1,
        )
        full_col_colors_df.columns = col_names

        # Pathway annotated clustermap filtered down to the same product doses
        for cluster in dists[cluster_dim_name].values:
            # unnormalized version
            unnormalized_vmax = np.percentile(dists.values, 90)
            g_dists = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=normalized_dists.sample_x,
                    sample_y=normalized_dists.sample_y,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=unnormalized_vmax,
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

            # normalized with same order
            normalized_vmax = np.percentile(normalized_dists.values, 90)
            g = sns.clustermap(
                normalized_dists.loc[cluster]
                .sel(
                    sample_x=normalized_dists.sample_x,
                    sample_y=normalized_dists.sample_y,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=normalized_vmax,
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
                f"{cluster}.{method_name}.normalized_distance_matrices_heatmap",
                dataset_name,
            )
            plt.clf()

            dists = normalized_dists if use_normalized else dists
            sig_samples = adata.obs[
                (adata.obs[f"{cl}_deg_product_dose"] == "True")
                | (adata.obs["product_name"] == "Vehicle")
            ]["product_dose"].unique()
            g = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=sig_samples,
                    sample_y=sig_samples,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=normalized_vmax if use_normalized else unnormalized_vmax,
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
                f"{cluster}.{method_name}.sig_{'normalized_' if use_normalized else ''}distance_matrices_heatmap",
                dataset_name,
            )
            plt.clf()

            top_samples = adata.obs[
                (adata.obs["dose"] == 10000.0)
                | (adata.obs["product_name"] == "Vehicle")
            ]["product_dose"].unique()
            g = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=top_samples,
                    sample_y=top_samples,
                )
                .to_pandas(),
                yticklabels=True,
                xticklabels=True,
                col_colors=full_col_colors_df,
                vmin=0,
                vmax=normalized_vmax if use_normalized else unnormalized_vmax,
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
                f"{cluster}.{method_name}.topdose_{'normalized_' if use_normalized else ''}distance_matrices_heatmap",
                dataset_name,
            )
            plt.clf()


# %%
#######################################
# Final distance matrix and DE analysis
#######################################
cl = "MCF7"
method_name = "mrvi_z30_u5"

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

sig_samples = adata.obs[
    (adata.obs[f"{cl}_deg_product_dose"] == "True")
    | (adata.obs["product_name"] == "Vehicle")
]["product_dose"].unique()
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
    order=["Random", "mrVI", "CompositionSCVI", "CompositionPCA"],
    palette=BARPLOT_CMAP,
    ax=ax,
)
min_lim = plot_df[metric].min() - 0.05
max_lim = plot_df[metric].max() + 0.05
ax.set_xlim(min_lim, max_lim)
ax.set_xticks([0.3, 0.35, 0.4, 0.45, 0.5])
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
# DE analysis
plt.rcParams["axes.grid"] = False

n_clusters = 11

clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
donor_info_ = pd.DataFrame({"cluster_id": clusters}, index=d1.sample_x.values)
vehicle_cluster = f"Cluster {donor_info_.loc['Vehicle_0']['cluster_id']}"

# %%
# Viz clusters
cluster_color_map = {i: c for i, c in enumerate(sns.color_palette("tab20", 20))}
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
train_adata_path = f"{dataset_name}.preprocessed.h5ad"
if not RUN_WITH_PARSER:
    train_adata_path = os.path.join(
        "../results/sciplex_pipeline/data", train_adata_path
    )
train_adata = sc.read(train_adata_path)
# train_adata_log = train_adata[train_adata.obs.product_dose.isin(sig_samples)].copy()

# %%
adata_path = f"{dataset_name}.{method_name}.final.h5ad"
if not RUN_WITH_PARSER:
    adata_path = os.path.join("../results/sciplex_pipeline/data", adata_path)
adata = sc.read(adata_path)
# train_adata_log.layers["counts"] = np.round(train_adata_log.X)
# sc.pp.normalize_total(train_adata_log)
# sc.pp.log1p(train_adata_log)
# train_adata_log.obs.loc[:, "donor_status"] = train_adata_log.obs.product_dose.map(
#     donor_info_.loc[:, "cluster_id"]
# ).values
# train_adata_log.obs.loc[:, "donor_status"] = "Cluster " + train_adata_log.obs.loc[
#     :, "donor_status"
# ].astype(str)

# # remove mt genes
# train_adata_log = train_adata_log[:, ~train_adata_log.var_names.str.startswith("MT-")]
# print(train_adata_log)

# method = "t-test"
# # Set vehicle as own donor status to use as reference
# train_adata_log.obs.loc[train_adata_log.obs.product_dose == "Vehicle_0", "donor_status"] = "Vehicle"
# sc.tl.rank_genes_groups(
#     train_adata_log,
#     "donor_status",
#     reference="Vehicle",
#     method=method,
#     n_genes=1000,
#     # rankby_abs=False,
# )
# sc.pl.rank_genes_groups_dotplot(
#     train_adata_log,
#     n_genes=5,
#     min_logfoldchange=1,
#     swap_axes=True,
#     save=f"{method_name}.clustered.svg",
# )
# # move file to correct place
# dataset_dir = os.path.join(output_dir, dataset_name)
# shutil.move(
#     f"figures/dotplot_{method_name}.clustered.svg",
#     os.path.join(dataset_dir, f"dotplot_{method_name}.clustered.svg"),
# )
# if not os.listdir(f"figures/"):
#     os.rmdir(f"figures/")

# # %%
# # GSEA for DE genes
# sc.tl.filter_rank_genes_groups(
#     train_adata_log,
#     min_fold_change=1,
#     min_in_group_fraction=0.25,
#     max_out_group_fraction=0.5,
# )
# # Load gene sets
# gene_set_names = [
#     # "MSigDB_Oncogenic_Signatures",
#     "MSigDB_Hallmark_2020",
#     # "WikiPathway_2021_Human",
#     # "KEGG_2021_Human",
#     # "Reactome_2022",
#     # "GO_Biological_Process_2023",
#     # "GO_Cellular_Component_2023",
#     # "GO_Molecular_Function_2023",
# ]
# gene_sets = [
#     gp.parser.download_library(gene_set_name, "human")
#     for gene_set_name in gene_set_names
# ]

# # %%
# enr_result_dict = {}
# for i in range(1, n_clusters + 1):
#     cluster_name = f"Cluster {i}"
#     de_genes = train_adata_log.uns["rank_genes_groups_filtered"]["names"][
#         cluster_name
#     ].tolist()
#     de_genes = [gene for gene in de_genes if str(gene) != "nan"]
#     try:
#         enr_results, fig = perform_gsea(de_genes, gene_sets=gene_sets, plot=True, use_server=False)
#         enr_result_dict[i] = enr_results
#         print(i)
#     except ValueError as e:
#         print(e)
#         continue
#     fig = fig + p9.theme(
#         figure_size=(6 * INCH_TO_CM, 6 * INCH_TO_CM),
#     )
#     dataset_dir = os.path.join(output_dir, dataset_name)
#     # fig.draw()
#     # fig.save(os.path.join(dataset_dir, f"gsea_cluster_{i}.svg"))

# # %%
# enr_pval_df_records = []
# for cluster_idx in range(1, n_clusters + 1):
#     if cluster_idx not in enr_result_dict:
#         enr_pval_df_records.append({"cluster_idx": cluster_idx})
#     else:
#         enr_cluster_results = enr_result_dict[cluster_idx]
#         enr_pval_df_records.append(
#             {"cluster_idx": cluster_idx, **enr_cluster_results.pivot(index="Gene_set", columns="Term", values="Significance score").iloc[0].to_dict()}
#         )
# enr_pval_df = pd.DataFrame.from_records(enr_pval_df_records, index="cluster_idx")
# enr_pval_df.fillna(0, inplace=True)

# # %%
# # Plot GSEA heatmap
# sns.clustermap(enr_pval_df.T, col_cluster=False, yticklabels=True, xticklabels=True, vmin=0, vmax=2, cmap="coolwarm")
# save_figures(
#     f"gsea_heatmap",
#     dataset_name,
# )

# %%
# MDE with low opacity vehicle cluster
vehicle_sim_thresh = 0.4
vehicle_dists = (
    dists.where(dists.values < vehicle_sim_thresh)
    .sel(sample_x="Vehicle_0", _dummy_name=1)
    .to_pandas()
)
vehicle_sim_samples = vehicle_dists[~vehicle_dists.isna()].index.to_list()
sns.histplot(dists.sel(sample_x="Vehicle_0").to_numpy().flatten(), bins=50)
plt.axvline(vehicle_sim_thresh, color="red")

# %%
rep = f"X_{method_name}_z_mde"
rep_plots = mde_reps.query(f"representation_name == '{rep}'").sample(frac=1)
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
    x="x",
    y="y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    alpha=0.1,
)
sns.scatterplot(
    full_opacity_rep_plots,
    x="x",
    y="y",
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
rep = f"X_{method_name}_u_mde"
rep_plots = mde_reps.query(f"representation_name == '{rep}'").sample(frac=1)
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
    x="x",
    y="y",
    hue=color_by,
    palette=palette,
    ax=ax,
    s=marker_size,
    alpha=0.1,
)
sns.scatterplot(
    full_opacity_rep_plots,
    x="x",
    y="y",
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
# z mdes colored by cluster membership
fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
plot_df = pd.DataFrame(adata.obsm[f"X_{method_name}_z_mde"], columns=["x", "y"])
plot_df["product_cluster"] = (
    adata.obs["product_dose"].map(donor_info_["cluster_id"]).values
)
plot_df = plot_df.loc[~plot_df["product_cluster"].isna()]
plot_df["product_cluster"] = plot_df["product_cluster"].astype(int).astype(str)
# shuffle
plot_df = plot_df.sample(frac=1).reset_index(drop=True)
sns.scatterplot(
    plot_df,
    x="x",
    y="y",
    hue="product_cluster",
    hue_order=[str(i) for i in range(1, n_clusters + 1)],
    palette={str(k): v for k, v in cluster_color_map.items()},
    ax=ax,
    s=marker_size,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")
ax.set_title(method_name)
save_figures(
    f"{method_name}.z_mdes_colored_by_cluster",
    dataset_name,
)

# %%
# u mdes colored by cluster membership
fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
plot_df = pd.DataFrame(adata.obsm[f"X_{method_name}_u_mde"], columns=["x", "y"])
plot_df["product_cluster"] = (
    adata.obs["product_dose"].map(donor_info_["cluster_id"]).values
)
plot_df = plot_df.loc[~plot_df["product_cluster"].isna()]
plot_df["product_cluster"] = plot_df["product_cluster"].astype(int).astype(str)
# shuffle
plot_df = plot_df.sample(frac=1).reset_index(drop=True)
sns.scatterplot(
    plot_df,
    x="x",
    y="y",
    hue="product_cluster",
    hue_order=[str(i) for i in range(1, n_clusters + 1)],
    palette={str(k): v for k, v in cluster_color_map.items()},
    ax=ax,
    s=marker_size,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")
ax.set_title(method_name)
save_figures(f"{method_name}.u_mdes_colored_by_cluster", dataset_name)

# %%
# admissibility check
import mrvi

model_path = f"{dataset_name}.{method_name}"
if not RUN_WITH_PARSER:
    model_path = os.path.join("../results/sciplex_pipeline/models", model_path)
model = mrvi.MrVI.load(model_path, adata=train_adata)
# %%
outlier_res = model.get_outlier_cell_sample_pairs(
    flavor="ball",
    subsample_size=5000,
    quantile_threshold=0.03,
)
outlier_res
# %%
outlier_res.to_netcdf(
    os.path.join(output_dir, f"{dataset_name}.{method_name}.outlier_res.nc")
)
# %%
adata.obs.loc[outlier_res.cell_name.values, "total_admissible"] = (
    outlier_res.is_admissible.sum(axis=1).values
)
plot_df = pd.DataFrame(adata.obsm[f"X_{method_name}_u_mde"], columns=["x", "y"])
plot_df["total_admissible"] = adata.obs.total_admissible.values

fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
sns.scatterplot(plot_df, x="x", y="y", hue="total_admissible", ax=ax, s=3)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")
ax.set_title(method_name)
save_figures(f"{method_name}.u_total_admissible", dataset_name)

# %%
fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
sns.histplot(plot_df, x="total_admissible", ax=ax)
save_figures(f"{method_name}.total_admissible_hist", dataset_name)

# %%
# Multivariate analysis DE
# (For this we create a column for each cluster since we require float values)
import mrvi

model_path = f"{dataset_name}.{method_name}"
if not RUN_WITH_PARSER:
    model_path = os.path.join("../results/sciplex_pipeline/models", model_path)
# Register adata to get scvi sample assignment
model = mrvi.MrVI.load(model_path, adata=train_adata)

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
# GSEA for DE genes
# Load gene sets
gene_set_names = [
    "MSigDB_Hallmark_2020",
]
gene_sets = [
    gp.parser.download_library(gene_set_name, "human")
    for gene_set_name in gene_set_names
]

enr_result_dict = {}
full_dfs = {}
de_dfs = {}
for cluster_i in cluster_wise_multivar_res:
    # cluster_i = 1
    cluster_multivar_res = cluster_wise_multivar_res[cluster_i]
    betas_ = (
        cluster_multivar_res["lfc"]
        .transpose("cell_name", "covariate", "gene")
        .loc[{"covariate": f"donor_cluster_{cluster_i}"}]
        .values
    )
    betas_ = betas_ / np.log(2)  # change to log 2
    # plt.hist(betas_.mean(0), bins=100)
    # plt.xlabel("LFC")
    # plt.show()

    lfc_df = pd.DataFrame(
        {
            "LFC": betas_.mean(0),
            "LFC_std": betas_.std(0),
            "gene": sub_train_adata.var_names,
            "gene_index": np.arange(sub_train_adata.shape[1]),
        }
    ).assign(absLFC=lambda x: np.abs(x.LFC))
    full_dfs[cluster_i] = lfc_df

    # thresh = np.quantile(lfc_df.absLFC, 0.95)
    # lfc_df.absLFC.hist(bins=100)
    # plt.axvline(thresh, color="red")
    # plt.xlabel("AbsLFC")
    # plt.show()
    # print((lfc_df.absLFC > thresh).sum())

    cond = lfc_df.absLFC > 1
    betas_de = betas_[:, cond]
    obs_de = lfc_df.loc[cond, :].reset_index(drop=True)
    obs_de.LFC.hist(bins=100)
    de_dfs[cluster_i] = obs_de

    de_genes = obs_de.gene.values
    de_genes = [gene for gene in de_genes if str(gene) != "nan"]
    try:
        enr_results, fig = perform_gsea(
            de_genes, gene_sets=gene_sets, plot=True, use_server=False
        )
        enr_result_dict[cluster_i] = enr_results
    except ValueError as e:
        print(e)
        continue
    # fig = fig + p9.theme(
    #     figure_size=(6 * INCH_TO_CM, 6 * INCH_TO_CM),
    # )
    # dataset_dir = os.path.join(output_dir, dataset_name)
    # fig.draw()
    # fig.save(os.path.join(dataset_dir, f"gsea_cluster_{i}.svg"))

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
sns.clustermap(
    enr_pval_df.T,
    col_cluster=False,
    yticklabels=True,
    xticklabels=True,
    vmin=0,
    vmax=2,
    cmap="Reds",
)
save_figures(
    f"multivar_gsea_heatmap",
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
        f"cluster_{cluster_i}_top_lfc_de_genes",
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
    "top_de_lfcs_clustermap",
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
