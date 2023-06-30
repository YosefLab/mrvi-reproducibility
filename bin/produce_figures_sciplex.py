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
from utils import load_results
from tree_utils import hierarchical_clustering
from scipy.cluster.hierarchy import fcluster
from plot_utils import INCH_TO_CM

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
pathway_color_map = {
    "Antioxidant": "#00FFFF",  # aquamarine
    "Apoptotic regulation": "#DAA520",  # goldenrod
    "Cell cycle regulation": "#008080",  # teal
    "DNA damage & DNA repair": "#808080",  # grey
    "Epigenetic regulation": "#000080",  # navy
    "Focal adhesion signaling": "#A52A2A",  # brown
    "HIF signaling": "#FFC0CB",  # pink
    "JAK/STAT signaling": "#008000",  # green
    "Metabolic regulation": "#FFD700",  # gold
    "Neuronal signaling": "#808000",  # olive
    "Nuclear receptor signaling": "#7FFF00",  # chartreuse
    "PKC signaling": "#DDA0DD",  # plum
    "Protein folding & Protein degradation": "#4B0082",  # indigo
    "TGF/BMP signaling": "#00FFFF",  # cyan
    "Tyrosine kinase signaling": "#ADD8E6",  # lightblue
    "Other": "#DA70D6",  # orchid
    "Vehicle": "#FF0000",  # red
}

# %%
# Representations
dataset_name = "sciplex_A549_simple_filtered_all_phases"
basedir = Path(output_dir).parent.parent.absolute()
all_results_files = glob.glob(os.path.join(basedir, "**"), recursive=True)
rep_results_paths = [
    x
    for x in all_results_files
    if x.startswith(
        f"/home/justin/ghrepos/scvi-v2-reproducibility/bin/../results/sciplex_pipeline/data/{dataset_name}"
    )
    and x.endswith(".final.h5ad")
]
rep_results = load_results(rep_results_paths)

# %%
mde_reps = rep_results["representations"].query("representation_type == 'MDE'")
if mde_reps.size >= 1:
    unique_reps = mde_reps.representation_name.unique()
    for rep in unique_reps:
        for color_by in ["pathway_level_1", "phase"]:
            rep_plots = mde_reps.query(f"representation_name == '{rep}'").sample(frac=1)
            # rep_plots = mde_reps.query(
            #     f"representation_name == '{rep}' and A549_deg_product_dose == 'True'"
            # )
            if color_by == "pathway_level_1":
                palette = pathway_color_map
            else:
                palette = None
            fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
            sns.scatterplot(
                rep_plots, x="x", y="y", hue=color_by, palette=palette, ax=ax, s=20
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
cell_lines = ["A549"]
method_names = [
    # "scviv2_attention_noprior",
    # "scviv2_attention_no_prior_mog",
    # "scviv2_z10",
    # "scviv2_z30",
    # "scviv2_z10_u5",
    "scviv2_z20_u5",
    "scviv2_z30_u5",
    # "scviv2_z10_u10",
    "scviv2_z20_u10",
    "scviv2_z30_u10",
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
                cmap="YlGnBu",
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
                cmap="YlGnBu",
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
                cmap="YlGnBu",
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
                cmap="YlGnBu",
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
# Final distance matrix and DE analysis
cl = "A549"
method_name = "scviv2_z20_u5"

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
    cmap="YlGnBu",
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
# DE analysis
plt.rcParams["axes.grid"] = False

n_clusters = 6

clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
donor_info_ = pd.DataFrame({"cluster_id": clusters}, index=d1.sample_x.values)

# %%
train_adata_path = f"{dataset_name}.preprocessed.h5ad"
if not RUN_WITH_PARSER:
    train_adata_path = os.path.join(
        "../results/sciplex_pipeline/data", train_adata_path
    )
train_adata = sc.read(train_adata_path)
train_adata_log = train_adata[train_adata.obs.product_dose.isin(sig_samples)].copy()

# %%
adata_path = f"{dataset_name}.{method_name}.final.h5ad"
if not RUN_WITH_PARSER:
    adata_path = os.path.join("../results/sciplex_pipeline/data", adata_path)
adata = sc.read(adata_path)
sc.pp.normalize_total(train_adata_log)
sc.pp.log1p(train_adata_log)
train_adata_log.obs.loc[:, "donor_status"] = train_adata_log.obs.product_dose.map(
    donor_info_.loc[:, "cluster_id"]
).values
train_adata_log.obs.loc[:, "donor_status"] = "Cluster " + train_adata_log.obs.loc[
    :, "donor_status"
].astype(str)

# remove mt genes
train_adata_log = train_adata_log[:, ~train_adata_log.var_names.str.startswith("MT-")]
print(train_adata_log)

method = "t-test"
sc.tl.rank_genes_groups(
    train_adata_log,
    "donor_status",
    method=method,
    n_genes=1000,
    # rankby_abs=False,
)
sc.pl.rank_genes_groups_dotplot(
    train_adata_log,
    n_genes=5,
    min_logfoldchange=0.5,
    swap_axes=True,
    save=f"{method_name}.clustered.svg",
)
# move file to correct place
dataset_dir = os.path.join(output_dir, dataset_name)
shutil.move(
    f"figures/dotplot_{method_name}.clustered.svg",
    os.path.join(dataset_dir, f"dotplot_{method_name}.clustered.svg"),
)
if not os.listdir(f"figures/"):
    os.rmdir(f"figures/")

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
    ax=ax,
    s=20,
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
    ax=ax,
    s=20,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
ax.set_xlabel("MDE1")
ax.set_ylabel("MDE2")
ax.set_title(method_name)
save_figures(f"{method_name}.u_mdes_colored_by_cluster", dataset_name)

# %%
# admissibility check
import scvi_v2

model_path = f"{dataset_name}.{method_name}"
if not RUN_WITH_PARSER:
    model_path = os.path.join("../results/sciplex_pipeline/models", model_path)
model = scvi_v2.MrVI.load(model_path, adata=train_adata)
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
adata.obs.loc[
    outlier_res.cell_name.values, "total_admissible"
] = outlier_res.is_admissible.sum(axis=1).values
plot_df = pd.DataFrame(adata.obsm[f"X_{method_name}_u_mde"], columns=["x", "y"])
plot_df["total_admissible"] = adata.obs.total_admissible.values

fig, ax = plt.subplots(figsize=(15 * INCH_TO_CM, 15 * INCH_TO_CM))
sns.scatterplot(plot_df, x="x", y="y", hue="total_admissible", ax=ax, s=20)
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
train_adata.obs["donor_cluster"] = train_adata.obs["product_dose"].map(donor_info_["cluster_id"]).values.astype(int)
for cluster_i in range(1, n_clusters + 1):
    train_adata.obs[f"donor_cluster_{cluster_i}"] = (
        train_adata.obs["donor_cluster"] == cluster_i
    ).astype(int)
obs_df = train_adata.obs.copy()
obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
model.donor_info = obs_df.set_index("_scvi_sample").sort_index()
sub_train_adata = train_adata[train_adata.obs["donor_cluster"] != "nan"]
# sub_train_adata = sub_train_adata[np.random.choice(sub_train_adata.shape[0], 1000, replace=False)].copy()
sub_train_adata.obs["_indices"] = np.arange(sub_train_adata.shape[0])
multivar_res = model.perform_multivariate_analysis(
    sub_train_adata,
    donor_keys = [f"donor_cluster_{cluster_i}" for cluster_i in range(1, n_clusters + 1)],
    store_lfc = True,
)
multivar_res
# %%
gene_properties = (sub_train_adata.X != 0).mean(axis=0).A1
gene_properties = pd.DataFrame(gene_properties, index=sub_train_adata.var_names, columns=["sparsity"])
top_genes = (multivar_res.lfc.mean("cell_name")
    .to_dataframe()
    .reset_index()
    .assign(
        abs_lfc=lambda x: np.abs(x.lfc),
    )
    .merge(gene_properties, left_on="gene", right_index=True, how="left")
    .sort_values("abs_lfc", ascending=False)
    .query("abs_lfc > 0.1")
)
top_genes
# %%
fig, ax = plt.subplots(figsize=(25 * INCH_TO_CM, 15 * INCH_TO_CM))
sns.scatterplot(top_genes, x="lfc", y="sparsity", hue="covariate", ax=ax)
plt.ylim(0, 0.9)
plt.axvline(0, color="grey", linestyle="--")
# annotate each point with gene name
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=7, rotation=45)

genes_to_label = top_genes.query("sparsity > 0.3")
label_point(genes_to_label.lfc, genes_to_label.sparsity, genes_to_label.gene, ax)  

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=1.5)
save_figures(f"{method_name}.top_genes", dataset_name)

# %%
genes_to_label

# %%
# model.explore_stratifications(dists.sel(
#     sample_x=sig_samples,
#     sample_y=sig_samples,
# ), show_figures=True)
# %%
