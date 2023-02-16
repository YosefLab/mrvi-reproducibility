# %%
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import scvi
from matplotlib.patches import Patch

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
if not RUN_WITH_PARSER:
    bulk_sim_dfs = {
        "A549": pd.read_csv(
            "/home/justin/ghrepos/scvi-v2-reproducibility/data/l1000_signatures/A549_deg_sim.csv"
        ),
        "MCF7": pd.read_csv(
            "/home/justin/ghrepos/scvi-v2-reproducibility/data/l1000_signatures/MCF7_deg_sim.csv"
        ),
    }

    dists_paths = [
        re.match(
            r"sciplex_[a-zA-Z0-9]+_significant_all_phases.[a-zA-Z0-9_]+.distance_matrices.nc",
            results_path,
        )[0]
        for results_path in results_paths
    ]

    in_product_all_dist_ratio = []
    in_product_top_2_dist_ratio = []
    bulk_compare_rf = []
    for dists_path in dists_paths:
        match = re.match(
            r"sciplex_([a-zA-Z0-9]+)_significant_all_phases.([a-zA-Z0-9_]+).distance_matrices.nc",
            dists_path,
        )
        cl = match.group(1)
        method_name = match.group(2)
        base_row = {"method_name": method_name, "cell_type": cl}

        dists = xr.open_dataset(
            os.path.join(output_dir, "../distance_matrices/", dists_path)
        )
        all_products = set()
        all_doses = set()
        for sample_name in dists.sample_x.data:
            product_name, dose = sample_name.split("_")
            if product_name != "Vehicle":
                all_products.add(product_name)
            if dose != "0":
                all_doses.add(dose)

        # Compute metrics
        top_two_doses = ["1000", "10000"]
        for phase in dists.phase_name.data:
            phase_dists = dists.sel(phase_name=phase).phase
            phase_dists_arr = phase_dists.data
            non_diag_mask = (
                np.ones(shape=phase_dists_arr.shape)
                - np.identity(phase_dists_arr.shape[0])
            ).astype(bool)
            off_diag_dist_avg = phase_dists_arr[non_diag_mask].mean()
            in_prod_mask = np.zeros(shape=phase_dists_arr.shape, dtype=bool)
            in_prod_top_two_mask = np.zeros(shape=phase_dists_arr.shape, dtype=bool)
            for product_name in all_products:
                for dosex in all_doses:
                    for dosey in all_doses:
                        if dosex == dosey:
                            continue
                        dosex_idx = np.where(
                            phase_dists.sample_x.data == f"{product_name}_{dosex}"
                        )[0][0]
                        dosey_idx = np.where(
                            phase_dists.sample_y.data == f"{product_name}_{dosey}"
                        )[0][0]
                        in_prod_mask[dosex_idx, dosey_idx] = True

                        if dosex in top_two_doses and dosey in top_two_doses:
                            in_prod_top_two_mask[dosex_idx, dosey_idx] = True
            in_prod_dist_avg = phase_dists_arr[in_prod_mask].mean()
            in_prod_top_two_dist_avg = phase_dists_arr[in_prod_top_two_mask].mean()
            ratio = in_prod_dist_avg / off_diag_dist_avg
            in_product_all_dist_ratio.append(
                {"in_product_all_dist_ratio": ratio, "phase": phase, **base_row}
            )

            top_two_ratio = in_prod_top_two_dist_avg / off_diag_dist_avg
            in_product_top_2_dist_ratio.append(
                {
                    "in_product_top_2_dist_ratio": top_two_ratio,
                    "phase": phase,
                    **base_row,
                }
            )


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

        # Histograms showing before and after normalization dists
        # model_dir_path = os.path.join(
        #     base_output_dir, f"models/sciplex_{cl}_significant_all_phases.{method_name}"
        # )

        # model_input_adata = sc.read(
        #         f"sciplex_{cl}_significant_all_phases.preprocessed.h5ad",
        # )
        # check_if_in_results(model_input_adata)
        # model = scvi_v2.MrVI.load(model_dir_path, adata=model_input_adata)
        # baseline_means, baseline_vars = model._compute_local_baseline_dists(None)
        # avg_baseline_mean = np.mean(baseline_means)
        # avg_baseline_var = np.mean(baseline_vars)

        # vmax = max(
        #     np.percentile(dists.phase.data, 95),
        #     np.percentile(normalized_dists.phase.data, 95),
        # )
        # binwidth = 0.1
        # bins = np.arange(0, vmax + binwidth, binwidth)
        # dists_hist = plt.hist(
        #     dists.phase.data.flatten(), bins=bins, alpha=0.5, label="distances"
        # )
        # plt.hist(
        #     normalized_dists.phase.data.flatten(),
        #     bins=bins,
        #     alpha=0.5,
        #     label="normalized distances",
        # )
        # x = np.linspace(-0.5, vmax + 0.5, 100)
        # p = np.max(dists_hist[0]) * norm.pdf(
        #     x, avg_baseline_mean, avg_baseline_var**0.5
        # )

        # plt.plot(x, p, "k", linewidth=2)

        # plt.xlim(-0.5, vmax + 0.5)
        # plt.legend()
        # save_figures(
        #     f"sciplex_{cl}_significant_all_phases.{method_name}.distance_matrix_hist_compare",
        # )
        # plt.clf()

        # u and z UMAPs for each cell line
        adata.obsm["mrvi_u_mde"] = scvi.model.utils.mde(
            adata.obsm["X_mrvi_u"], repulsive_fraction=1.3
        )
        adata.obsm["mrvi_z_mde"] = scvi.model.utils.mde(
            adata.obsm["X_mrvi_z"], repulsive_fraction=1.3
        )

        sc.pl.embedding(
            adata, "mrvi_u_mde", color=["pathway", "phase"], ncols=1, show=False
        )
        save_figures(f"sciplex_{cl}_significant_all_phases.{method_name}.u_latent_mde")
        plt.clf()

        sc.pl.embedding(
            adata, "mrvi_z_mde", color=["pathway", "phase"], ncols=1, show=False
        )
        save_figures(f"sciplex_{cl}_significant_all_phases.{method_name}.z_latent_mde")
        plt.clf()

# %%
