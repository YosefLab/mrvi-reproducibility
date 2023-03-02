"""Filter out products that do not have a significant difference with the Vehicle."""
# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc

import scanpy as sc
import seaborn as sns
import scipy
import xarray as xr
import leidenalg as la
import igraph as ig
# %%
method_name = "scviv2"
cell_lines = ["A549", "MCF7", "K562"]

base_dir_path = "/home/justin/ghrepos/scvi-v2-reproducibility"
results_dir_path = os.path.join(base_dir_path, "results/3_1_sciplex_pipeline")
 
# %%
leiden_vehicle_sim_prods = set()
deg_vehicle_sim_prods = set()
for cell_line in cell_lines:
    dists_path = os.path.join(
        results_dir_path,
        f"distance_matrices/sciplex_{cell_line}_significant_all_phases.{method_name}.distance_matrices.nc"
    )
    dists_arr = xr.load_dataarray(dists_path)

    avged_dists = dists_arr.mean("phase_name").values
    off_diag_vals = avged_dists[~np.eye(avged_dists.shape[0], dtype=bool)]
    scaled_dists = np.clip(avged_dists, a_min=off_diag_vals.min(), a_max=None) - off_diag_vals.min()
    sims = ((scaled_dists.max() - scaled_dists) / scaled_dists.max())
    sims[sims < np.quantile(sims, 0.2)] = 0.

    g_adj = ig.Graph.Weighted_Adjacency(sims, mode="undirected")
    partition = la.RBConfigurationVertexPartition(g_adj, resolution_parameter=1)
    optimizer = la.Optimiser()
    optimizer.optimise_partition(partition)

    # Get the cluster labels
    cluster_labels = np.array(partition.membership)

    print(np.unique(cluster_labels, return_counts=True))

    # color_list = [
    #     'red',
    #     'blue',
    #     'green',
    #     'cyan',
    #     'pink',
    #     'orange',
    #     'grey',
    #     'yellow',
    #     'white',
    #     'black',
    #     'purple'
    # ]
    # ig.plot(g_adj, vertex_color=[color_list[k % len(color_list)] for k in cluster_labels])

    prod_dose_names = list(dists_arr.coords["sample_x"].data)
    vehicle_leiden_cluster = cluster_labels[prod_dose_names.index("Vehicle_0")]
    cluster_idxs = np.where(cluster_labels == vehicle_leiden_cluster)[0]
    prods, prod_cts = np.unique([prod_dose.split('_')[0] for prod_dose in np.array(prod_dose_names)[cluster_idxs]], return_counts=True)
    # Filter on at least two doses being in same cluster as vehicle
    prods = prods[prod_cts >= 2]
    leiden_vehicle_sim_prods = leiden_vehicle_sim_prods.union(prods)

    adata_path = os.path.join(
        results_dir_path,
        f"data/sciplex_{cell_line}_significant_all_phases.preprocessed.h5ad",
    )
    adata = sc.read_h5ad(adata_path)

    adata.layers["log1p"] = sc.pp.log1p(adata, copy=True).X
    adata.uns["log1p"] = {"base": None} 
    sc.tl.rank_genes_groups(
        adata,
        f"product_name",
        layer="log1p",
        method="wilcoxon",
    )
    rgg_fig = sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=10,
        values_to_plot="logfoldchanges",
        cmap="bwr",
        vmin=-4,
        vmax=4,
        min_logfoldchange=0.5,
        return_fig=True,
    )
    # rgg_fig.savefig(os.path.join(base_dir_path, f"notebooks/figures/3_1_figures/sciplex_{cell_line}_significant_all_phases.{method_name}.product_name.deg_dotplot.png"))

    # Collect products similar degs to vehicle at least in one cell line
    d = scipy.cluster.hierarchy.dendrogram(adata.uns["dendrogram_product_name"]["linkage"], labels=adata.uns["dendrogram_product_name"]["categories_ordered"])
    prod_names, clusters = d["ivl"], d["color_list"]
    vehicle_cluster = clusters[prod_names.index("Vehicle")]
    cluster_idxs = np.where(np.array(clusters) == vehicle_cluster)[0]
    deg_vehicle_sim_prods = deg_vehicle_sim_prods.union(set(np.array(prod_names)[cluster_idxs]))

# %%
# Compare vehicle sim prods by DEG to low distance prods
print(leiden_vehicle_sim_prods)
print(deg_vehicle_sim_prods)
print(f"Intersection: {leiden_vehicle_sim_prods.intersection(deg_vehicle_sim_prods)}")
print(f"Intersection fraction of leiden {len(leiden_vehicle_sim_prods.intersection(deg_vehicle_sim_prods)) / len(leiden_vehicle_sim_prods)}")
print(f"Intersection fraction of deg {len(leiden_vehicle_sim_prods.intersection(deg_vehicle_sim_prods)) / len(deg_vehicle_sim_prods)}")
print(f"Fraction of all products: {len(leiden_vehicle_sim_prods) / len(adata.obs['product_name'].unique())}")

# %%
# Save leiden vehicle sim prods
leiden_vehicle_sim_prods_path = os.path.join(
    base_dir_path,
    "notebooks/output/leiden_vehicle_sim_prods.txt",
)
with open(leiden_vehicle_sim_prods_path, "w") as f:
    for prod in leiden_vehicle_sim_prods:
        f.write(f"{prod}\n")
# %%
