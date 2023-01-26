# %%
import os

import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import scvi
import scvi_v2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

# %%
results_path = "../results/simple_pipeline/"
figures_path = os.path.join(results_path, "figures")
cell_lines = ["A549", "K562", "MCF7"]
if not os.path.exists(figures_path): 
    os.mkdir(figures_path)

# %%
pathway_color_map = {
      "Antioxidant" : "aquamarine",
      "Apoptotic regulation" : "goldenrod",
      "Cell cycle regulation" : "azure",
      "DNA damage & DNA repair" : "grey",
      "Epigenetic regulation" : "navy",
      "Focal adhesion signaling" :
        "brown",
      "HIF signaling" : "darkgreen",
      "JAK/STAT signaling" : "orangered",
      "Metabolic regulation" : "gold",
      "Neuronal signaling" : "olive",
      "Nuclear receptor signaling" : "chartreuse",
      "PKC signaling" : "plum",
      "Protein folding & Protein degradation" : "indigo",
      "TGF/BMP signaling" : "cyan",
      "Tyrosine kinase signaling" : "red",
      "Other" : "orchid",
      "Vehicle" : "lightblue"
}

# %%
method_names = ["scviv2", "scviv2_nonlinear"]

# Same figures for all phases
for method_name in method_names:
    for cl in cell_lines:
        normalized_dists_path = os.path.join(
            results_path,
            f"distance_matrices/sciplex_{cl}_significant_all_phases.{method_name}.normalized_distance_matrices.nc",
        )
        normalized_dists = xr.open_dataset(normalized_dists_path)

        dists_path = os.path.join(
            results_path,
            f"distance_matrices/sciplex_{cl}_significant_all_phases.{method_name}.distance_matrices.nc",
        )
        dists = xr.open_dataset(dists_path)

        adata_path = os.path.join(results_path, f"data/sciplex_{cl}_significant_all_phases.{method_name}.h5ad")
        adata = sc.read(adata_path)

        sample_to_pathway = adata.obs[["product_dose", "pathway_level_1"]].drop_duplicates().set_index("product_dose")["pathway_level_1"].to_dict()
        sample_to_color_df = normalized_dists.sample_x.to_series().map(sample_to_pathway).map(pathway_color_map)

        # Pathway annotated clustermap filtered down to the same product doses
        for phase in dists.phase_name.values:
            # unnormalized version
            unnormalized_vmax = np.percentile(dists.phase.values, 90)
            g_dists = sns.clustermap(
                dists.phase.sel(phase_name=phase,sample_x=normalized_dists.sample_x, sample_y=normalized_dists.sample_y).to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                vmin=0,
                vmax=unnormalized_vmax,
            )
            g_dists.ax_heatmap.set_xticklabels(g_dists.ax_heatmap.get_xmajorticklabels(), fontsize = 2)
            g_dists.ax_heatmap.set_yticklabels(g_dists.ax_heatmap.get_ymajorticklabels(), fontsize = 2)

            handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
            product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                    bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
            plt.gca().add_artist(product_legend)
            plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_phase_{phase}.{method_name}.distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
            plt.clf()

            # normalized with same order
            dists_sample_order = g_dists.data.columns[g_dists.dendrogram_col.reordered_ind]
            g = sns.clustermap(
                normalized_dists.phase.sel(phase_name=phase,sample_x=dists_sample_order, sample_y=dists_sample_order).to_pandas(),
                cmap="YlGnBu",
                yticklabels=True,
                xticklabels=True,
                col_colors=sample_to_color_df,
                row_cluster=False,
                col_cluster=False,
                vmin=1,
                vmax=4,
            )
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 2)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 2)

            handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
            product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                    bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
            plt.gca().add_artist(product_legend)
            plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_phase_{phase}.{method_name}.normalized_distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
            plt.clf()

            # normalized with clustered on clipped values
            clipped_normalized_dists = normalized_dists.phase.sel(phase_name=phase,sample_x=normalized_dists.sample_x, sample_y=normalized_dists.sample_y).to_pandas()
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
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 2)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 2)

            handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
            product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                    bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
            plt.gca().add_artist(product_legend)
            plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_phase_{phase}.{method_name}.clipped_normalized_distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
            plt.clf()

        # Histograms showing before and after normalization dists
        model_dir_path = os.path.join(results_path, f"models/sciplex_{cl}_significant_all_phases.{method_name}")

        model_input_adata = sc.read(os.path.join(results_path, f"data/sciplex_{cl}_significant_all_phases.preprocessed.h5ad"))
        model = scvi_v2.MrVI.load(model_dir_path, adata=model_input_adata)
        baseline_means, baseline_vars = model._compute_local_baseline_dists(None)
        avg_baseline_mean = np.mean(baseline_means)
        avg_baseline_var = np.mean(baseline_vars)

        vmax = max(np.percentile(dists.phase.data, 95), np.percentile(normalized_dists.phase.data, 95))
        binwidth = 0.1
        bins = np.arange(0, vmax + binwidth, binwidth)
        dists_hist = plt.hist(dists.phase.data.flatten(), bins=bins, alpha=0.5, label="distances")
        plt.hist(
            normalized_dists.phase.data.flatten(), bins=bins, alpha=0.5, label="normalized distances"
        )
        x = np.linspace(-0.5, vmax + 0.5, 100)
        p = np.max(dists_hist[0]) * norm.pdf(x, avg_baseline_mean, avg_baseline_var**0.5)
        
        plt.plot(x, p, 'k', linewidth=2)

        plt.xlim(-0.5, vmax + 0.5)
        plt.legend()
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.{method_name}.distance_matrix_hist_compare.png"), bbox_inches="tight", dpi=300)
        plt.clf()

        # u and z UMAPs for each cell line
        adata.obsm["mrvi_u_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_u"], repulsive_fraction=1.3)
        adata.obsm["mrvi_z_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_z"], repulsive_fraction=1.3)

        sc.pl.embedding(adata, "mrvi_u_mde", color=["pathway", "phase"], ncols=1, show=False)
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.{method_name}.u_latent_mde.png"), bbox_inches="tight", dpi=300)
        plt.clf()

        sc.pl.embedding(adata, "mrvi_z_mde", color=["pathway", "phase"], ncols=1, show=False)
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.{method_name}.z_latent_mde.png"), bbox_inches="tight", dpi=300)
        plt.clf()

# %%
