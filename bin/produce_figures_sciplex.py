# %%
import argparse
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import plotnine as p9
from matplotlib.patches import Patch
from utils import load_results, INCH_TO_CM

# Change to False if you want to run this script directly
RUN_WITH_PARSER = True
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
    plt.savefig(os.path.join(dataset_dir, filename + ".svg"))


# %%
pathway_color_map = {
    "Antioxidant": "aquamarine",
    "Apoptotic regulation": "goldenrod",
    "Cell cycle regulation": "azure",
    "DNA damage & DNA repair": "grey",
    "Epigenetic regulation": "navy",
    "Focal adhesion signaling": "brown",
    "HIF signaling": "darkgreen",
    "JAK/STAT signaling": "green",
    "Metabolic regulation": "gold",
    "Neuronal signaling": "olive",
    "Nuclear receptor signaling": "chartreuse",
    "PKC signaling": "plum",
    "Protein folding & Protein degradation": "indigo",
    "TGF/BMP signaling": "cyan",
    "Tyrosine kinase signaling": "lightblue",
    "Other": "orchid",
    "Vehicle": "red",
}

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
        & (
            sciplex_metrics_df["distance_type"] == "distance_matrices"
        )  # Exclude normalized matrices
    ]
    for metric in sciplex_metrics_df.columns:
        if metric in [
            "model_name",
            "dataset_name",
            "distance_type",
            "phase",
            "phase_name",
            "leiden_1.0",
        ]:
            continue
        if plot_df[metric].isna().any():
            continue
        plot_df.loc[plot_df["phase"].isna(), "phase"] = plot_df.loc[
            plot_df["phase"].isna(), "phase_name"
        ]
        plot_df.loc[plot_df["phase"].isna(), "phase"] = "leiden"
        fig = (
            p9.ggplot(
                plot_df,
                p9.aes(x="model_name", y=metric, color="phase"),
            )
            + p9.geom_boxplot(width=0.15)
            + p9.theme_classic()
            + p9.coord_flip()
            + p9.theme(
                legend_position="top",
                figure_size=(4 * INCH_TO_CM, 6 * INCH_TO_CM),
            )
            + p9.labs(x="model_name", y=metric)
        )
        fig.save(os.path.join(dataset_dir, f"{metric}.svg"))

# %%
cell_lines = ["A549", "MCF7", "K562"]
method_names = ["scviv2", "scviv2_nonlinear"]

# Per dataset plots
for method_name in method_names:
    for cl in cell_lines:
        dataset_name = f"sciplex_{cl}_simple_filtered_all_phases"
        normalized_dists_path = (
            f"{dataset_name}.{method_name}.normalized_distance_matrices.nc"
        )
        normalized_dists = xr.open_dataarray(normalized_dists_path)
        dists_path = f"{dataset_name}.{method_name}.distance_matrices.nc"
        dists = xr.open_dataarray(dists_path)
        cluster_dim_name = dists.dims[0]

        adata_path = f"{dataset_name}.{method_name}.final.h5ad"
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

        full_col_colors_df = pd.concat(
            [
                sample_to_color_df,
                sample_to_n_deg_df,
                sample_to_sig_prod_dose,
            ],
            axis=1,
        )
        full_col_colors_df.columns = [
            "pathway",
            "n_degs",
            "sig_prod_dose",
        ]

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

            sig_samples = adata.obs[
                (adata.obs[f"{cl}_deg_product_dose"] == "True")
                | (adata.obs["product_name"] == "Vehicle")
            ]["product_dose"].unique()
            g = sns.clustermap(
                normalized_dists.loc[cluster]
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
                f"{cluster}.{method_name}.sig_normalized_distance_matrices_heatmap",
                dataset_name,
            )
            plt.clf()

# %%
baseline_method_names = [
    "composition_PCA_clusterkey_subleiden1",
    "composition_SCVI_clusterkey_subleiden1",
]

# Per baseline dataset plots
for method_name in baseline_method_names:
    for cl in cell_lines:
        dataset_name = f"sciplex_{cl}_simple_filtered_all_phases"

        dists_path = f"{dataset_name}.{method_name}.distance_matrices.nc"
        dists = xr.open_dataarray(dists_path)
        cluster_dim_name = dists.dims[0]

        adata_path = f"{dataset_name}.{method_name}.final.h5ad"
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

        # Pathway annotated clustermap filtered down to the same product doses
        for cluster in dists[cluster_dim_name].values:
            vmax = np.percentile(dists.values, 90)
            g_dists = sns.clustermap(
                dists.loc[cluster].to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                vmin=0,
                vmax=vmax,
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

            sig_samples = adata.obs[
                (adata.obs[f"{cl}_deg_product_dose"] == "True")
                | (adata.obs["product_name"] == "Vehicle")
            ]["product_dose"].unique()
            g_dists = sns.clustermap(
                dists.loc[cluster]
                .sel(
                    sample_x=sig_samples,
                    sample_y=sig_samples,
                )
                .to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                vmin=0,
                vmax=vmax,
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
                f"{cluster}.{method_name}.sig_distance_matrices_heatmap", dataset_name
            )
            plt.clf()

# %%
