# %%
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import plotnine as p9
from matplotlib.patches import Patch
from utils import load_results, INCH_TO_CM

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
    results_paths = (
        pd.read_csv(os.path.join(output_dir, "path_to_intermediary_files.txt"))
        .squeeze()
        .values.flatten()
    )
# %%
def save_figures(filename):
    plt.savefig(os.path.join(output_dir, filename + ".svg"))
    plt.savefig(
        os.path.join(output_dir, filename + ".png"), dpi=300, bbox_inches="tight"
    )


def check_if_in_results(filepath):
    for result_path in results_paths:
        if os.path.samefile(result_path, filepath):
            return True
    raise ValueError(f"{filepath} not found in results_paths")


# %%
pathway_color_map = {
    "Antioxidant": "aquamarine",
    "Apoptotic regulation": "goldenrod",
    "Cell cycle regulation": "azure",
    "DNA damage & DNA repair": "grey",
    "Epigenetic regulation": "navy",
    "Focal adhesion signaling": "brown",
    "HIF signaling": "darkgreen",
    "JAK/STAT signaling": "orangered",
    "Metabolic regulation": "gold",
    "Neuronal signaling": "olive",
    "Nuclear receptor signaling": "chartreuse",
    "PKC signaling": "plum",
    "Protein folding & Protein degradation": "indigo",
    "TGF/BMP signaling": "cyan",
    "Tyrosine kinase signaling": "red",
    "Other": "orchid",
    "Vehicle": "lightblue",
}

# %%
# Cross method comparison plots
all_results = load_results(results_paths)
sciplex_metrics_df = all_results["sciplex_metrics"]

for dataset_name in sciplex_metrics_df["dataset_name"].unique():
    plot_df = sciplex_metrics_df[sciplex_metrics_df["dataset_name"] == dataset_name]
    for metric in sciplex_metrics_df.columns:
        if metric in ["model_name", "dataset_name", "phase"]:
            continue
        fig = (
            p9.ggplot(
                plot_df,
                p9.aes(x="model_name", y=metric, fill="model_name", color="phase"),
            )
            + p9.geom_boxplot()
            + p9.theme_classic()
            + p9.coord_flip()
            + p9.theme(
                legend_position="none",
                figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
            )
            + p9.labs(x="model_name", y=metric)
        )
        fig.save(os.path.join(output_dir, f"{dataset_name}.{metric}.svg"))

# %%
cell_lines = ["A549", "MCF7", "K562"]
method_names = ["scviv2", "scviv2_nonlinear"]

# Per dataset plots
for method_name in method_names:
    for cl in cell_lines:
        normalized_dists_path = f"sciplex_{cl}_significant_all_phases.{method_name}.normalized_distance_matrices.nc"
        check_if_in_results(normalized_dists_path)
        normalized_dists = xr.open_dataset(normalized_dists_path)
        dists_path = (
            f"sciplex_{cl}_significant_all_phases.{method_name}.distance_matrices.nc"
        )
        check_if_in_results(dists_path)
        dists = xr.open_dataset(dists_path)

        adata_path = f"sciplex_{cl}_significant_all_phases.{method_name}.h5ad"
        check_if_in_results(adata_path)
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

        # Pathway annotated clustermap filtered down to the same product doses
        for phase in dists.phase_name.values:
            # unnormalized version
            unnormalized_vmax = np.percentile(dists.phase.values, 90)
            g_dists = sns.clustermap(
                dists.phase.sel(
                    phase_name=phase,
                    sample_x=normalized_dists.sample_x,
                    sample_y=normalized_dists.sample_y,
                ).to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
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
                f"sciplex_{cl}_significant_phase_{phase}.{method_name}.distance_matrices_heatmap",
            )
            plt.clf()

            # normalized with same order
            normalized_vmax = np.percentile(normalized_dists.phase.values, 90)
            dists_sample_order = g_dists.data.columns[
                g_dists.dendrogram_col.reordered_ind
            ]
            g = sns.clustermap(
                normalized_dists.phase.sel(
                    phase_name=phase,
                    sample_x=dists_sample_order,
                    sample_y=dists_sample_order,
                ).to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                row_cluster=False,
                col_cluster=False,
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
                f"sciplex_{cl}_significant_phase_{phase}.{method_name}.normalized_distance_matrices_heatmap",
            )
            plt.clf()

            # normalized with clustered on clipped values
            clipped_normalized_dists = normalized_dists.phase.sel(
                phase_name=phase,
                sample_x=normalized_dists.sample_x,
                sample_y=normalized_dists.sample_y,
            ).to_pandas()
            clipped_normalized_dists = clipped_normalized_dists.clip(lower=1, upper=4)
            g = sns.clustermap(
                clipped_normalized_dists,
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                vmin=1,
                vmax=4,
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
                f"sciplex_{cl}_significant_phase_{phase}.{method_name}.clipped_normalized_distance_matrices_heatmap",
            )
            plt.clf()

# %%
