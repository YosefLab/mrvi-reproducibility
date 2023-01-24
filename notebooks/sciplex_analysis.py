# %%
import os

import numpy as np
import pandas as pd
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

# # %%
# for cl in cell_lines:
#     normalized_dists_path = os.path.join(
#         results_path,
#         f"distance_matrices/sciplex_{cl}_significant.scviv2.normalized_distance_matrices.nc",
#     )
#     normalized_dists = xr.open_dataarray(normalized_dists_path)

#     dists_path = os.path.join(
#         results_path,
#         f"distance_matrices/sciplex_{cl}_significant.scviv2.distance_matrices.nc",
#     )
#     dists = xr.open_dataarray(dists_path)

#     adata_path = os.path.join(results_path, f"data/sciplex_{cl}_significant.scviv2.h5ad")
#     adata = sc.read(adata_path)
#     sample_to_pathway = adata.obs[["product_dose", "pathway_level_1"]].drop_duplicates().set_index("product_dose")["pathway_level_1"].to_dict()
#     sample_to_color_df = normalized_dists.sample_x.to_series().map(sample_to_pathway).map(pathway_color_map)

#     # Pathway annotated clustermap
#     g = sns.clustermap(
#         normalized_dists.mean(dim="cell_name").to_pandas(),
#         cmap="YlGnBu",
#         yticklabels=True,
#         xticklabels=True,
#         col_colors=sample_to_color_df,
#     )
#     if cl == "MCF7":
#         g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 5)
#         g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 5)

#     handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
#     product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
#             bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
#     plt.gca().add_artist(product_legend)
#     plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant.scviv2.normalized_distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
#     plt.clf()

#     # unnormalized version
#     g = sns.clustermap(
#         dists.mean(dim="cell_name").to_pandas(),
#         cmap="YlGnBu",
#         yticklabels=True,
#         xticklabels=True,
#         col_colors=sample_to_color_df,
#     )
#     g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 5)
#     g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 5)

#     handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
#     product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
#             bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
#     plt.gca().add_artist(product_legend)
#     plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant.scviv2.distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
#     plt.clf()

#     # Histograms showing before and after normalization dists
#     vmax = np.percentile(normalized_dists.values, 95)
#     binwidth = 0.1
#     bins = np.arange(0, vmax + binwidth, binwidth)
#     plt.hist(dists.data.flatten(), bins=bins, alpha=0.5, label="distances")
#     plt.hist(
#         normalized_dists.data.flatten(), bins=bins, alpha=0.5, label="normalized distances"
#     )
#     plt.xlim(-0.5, vmax + 0.5)
#     plt.legend()
#     plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant.scviv2.distance_matrix_hist_compare.png"), bbox_inches="tight", dpi=300)
#     plt.clf()

#     # u and z UMAPs for each cell line
#     adata.obsm["mrvi_u_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_u"])
#     adata.obsm["mrvi_z_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_z"])

#     sc.pl.embedding(adata, "mrvi_u_mde", color=["product_dose", "pathway"], ncols=1, show=False)
#     plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant.scviv2.u_latent_mde.png"), bbox_inches="tight", dpi=300)
#     plt.clf()

#     sc.pl.embedding(adata, "mrvi_z_mde", color=["product_dose", "pathway"], ncols=1, show=False)
#     plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant.scviv2.z_latent_mde.png"), bbox_inches="tight", dpi=300)
#     plt.clf()

# %%
# Filter for product doses significantly diff from Vehicle in MCF7
# Note: This is hackily run after running below block for MCF7
# Pretty arbitrary!
sig_prod_dose_idx = g.data.columns.get_loc('Cediranib (AZD2171)_10000')
sig_prod_dose_reordered_idx = g.dendrogram_col.reordered_ind.index(sig_prod_dose_idx)
sig_prod_dose_list = g.data.columns[g.dendrogram_col.reordered_ind[sig_prod_dose_reordered_idx:]]

# %%
# Same figures for all phases
for cl in cell_lines:
    normalized_dists_path = os.path.join(
        results_path,
        f"distance_matrices/sciplex_{cl}_significant_all_phases.scviv2.normalized_distance_matrices.nc",
    )
    normalized_dists = xr.open_dataset(normalized_dists_path)

    dists_path = os.path.join(
        results_path,
        f"distance_matrices/sciplex_{cl}_significant_all_phases.scviv2.distance_matrices.nc",
    )
    dists = xr.open_dataset(dists_path)

    adata_path = os.path.join(results_path, f"data/sciplex_{cl}_significant_all_phases.scviv2.h5ad")
    adata = sc.read(adata_path)

    sample_to_pathway = adata.obs[["product_dose", "pathway_level_1"]].drop_duplicates().set_index("product_dose")["pathway_level_1"].to_dict()
    sample_to_color_df = normalized_dists.sample_x.to_series().map(sample_to_pathway).map(pathway_color_map)

    # Pathway annotated clustermap filtered down to the same product doses
    for phase in dists.phase_name.values:
        normalized_vmax = np.max(normalized_dists.phase.values)
        g = sns.clustermap(
            normalized_dists.phase.sel(phase_name=phase,sample_x=normalized_dists.sample_x, sample_y=normalized_dists.sample_y).to_pandas(),
            cmap="YlGnBu",
            yticklabels=True,
            xticklabels=True,
            col_colors=sample_to_color_df,
            vmin=0,
            vmax=5,
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 2)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 2)

        handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
        product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
        plt.gca().add_artist(product_legend)
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_phase_{phase}.scviv2.normalized_distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
        plt.clf()

        # unnormalized version
        unnormalized_vmax = np.percentile(dists.phase.values, 90)
        g = sns.clustermap(
            dists.phase.sel(phase_name=phase,sample_x=normalized_dists.sample_x, sample_y=normalized_dists.sample_y).to_pandas(),
            cmap="YlGnBu",
            yticklabels=True,
            xticklabels=True,
            col_colors=sample_to_color_df,
            vmin=0,
            vmax=unnormalized_vmax,
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 2)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 2)

        handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
        product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
        plt.gca().add_artist(product_legend)
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_phase_{phase}.scviv2.distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
        plt.clf()

        # unnormalized version with same prod doses unclustered
        filtered_dists = dists.phase.sel(phase_name=phase,sample_x=sig_prod_dose_list, sample_y=sig_prod_dose_list).to_pandas()
        unnormalized_vmax = np.percentile(filtered_dists, 90)
        g = sns.clustermap(
            filtered_dists,
            cmap="YlGnBu",
            yticklabels=True,
            xticklabels=True,
            col_colors=sample_to_color_df,
            vmin=0,
            vmax=unnormalized_vmax,
            row_cluster=False,
            col_cluster=False,
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 5)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 5)

        handles = [Patch(facecolor=pathway_color_map[name]) for name in pathway_color_map]
        product_legend = plt.legend(handles, pathway_color_map, title='Product Name',
                bbox_to_anchor=(1.3, 0.9), bbox_transform=plt.gcf().transFigure, loc='upper right')
        plt.gca().add_artist(product_legend)
        plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_filtered_phase_{phase}.scviv2.distance_matrices_heatmap.png"), bbox_inches="tight", dpi=300)
        plt.clf()

    # Histograms showing before and after normalization dists
    model_dir_path = os.path.join(results_path, f"models/sciplex_{cl}_significant_all_phases.scviv2")

    model_input_adata = sc.read(os.path.join(results_path, f"data/sciplex_{cl}_significant_all_phases.preprocessed.h5ad"))
    model = scvi_v2.MrVI.load(model_dir_path, adata=model_input_adata)
    baseline_means, baseline_vars = model._compute_local_baseline_dists(None)
    avg_baseline_mean = np.mean(baseline_means)
    avg_baseline_var = np.mean(baseline_vars)

    vmax = np.percentile(normalized_dists.cell.values, 95)
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
    plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.scviv2.distance_matrix_hist_compare.png"), bbox_inches="tight", dpi=300)
    plt.clf()

    # u and z UMAPs for each cell line
    adata.obsm["mrvi_u_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_u"], repulsive_fraction=1.3)
    adata.obsm["mrvi_z_mde"] = scvi.model.utils.mde(adata.obsm["X_mrvi_z"], repulsive_fraction=1.3)

    sc.pl.embedding(adata, "mrvi_u_mde", color=["pathway", "phase"], ncols=1, show=False)
    plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.scviv2.u_latent_mde.png"), bbox_inches="tight", dpi=300)
    plt.clf()

    sc.pl.embedding(adata, "mrvi_z_mde", color=["pathway", "phase"], ncols=1, show=False)
    plt.savefig(os.path.join(figures_path, f"sciplex_{cl}_significant_all_phases.scviv2.z_latent_mde.png"), bbox_inches="tight", dpi=300)
    plt.clf()

# %%
