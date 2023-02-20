import os
from pathlib import Path

import pandas as pd
import xarray as xr
from tree_utils import hierarchical_clustering
from utils import (
    determine_if_file_empty,
    load_config,
    make_parents,
    wrap_kwargs,
)


@wrap_kwargs
def compute_rf(
    *,
    distance_matrices,
    distance_matrices_gt,
    config_in,
    table_out,
):
    """Main fn for computing RF distance.

    Parameters
    ----------
    adata_in :
        path to anndata file containing cell-type specific distance matrices
    config_in :
        path to config file
    table_out :
        desired output table file
    """
    basename = os.path.basename(distance_matrices)
    model_name = basename.split(".")[1]
    config = load_config(config_in)
    celltype_key = config["labels_key"]
    dim_name = f"{celltype_key}_name"

    # Linkage method to use for hierarchical clustering
    clustering_method = config["clustering_method"]
    make_parents(table_out)

    if determine_if_file_empty(distance_matrices_gt):
        Path(table_out).touch()
        return

    gt_mats = xr.open_dataarray(distance_matrices_gt)
    try:
        inferred_mats = xr.open_dataarray(distance_matrices)
    except ValueError:
        inferred_mats = xr.open_dataset(distance_matrices)[celltype_key]
    inferred_mats = inferred_mats.rename("distance")
    aligned_mats = xr.merge([gt_mats, inferred_mats], join="left")
    dists = []
    cts = []

    clusters = aligned_mats.coords[dim_name].values
    for cluster_name in clusters:
        cts.append(cluster_name)

        dist_gt = aligned_mats.distance_gt.loc[cluster_name].values
        z_gt = hierarchical_clustering(dist_gt, method=clustering_method)
        dist_inferred = aligned_mats.distance.loc[cluster_name].values
        z_inferred = hierarchical_clustering(dist_inferred, method=clustering_method)

        rf_dist = z_gt.robinson_foulds(z_inferred)
        norm_rf = rf_dist[0] / rf_dist[1]
        dists.append(norm_rf)
    dists = pd.DataFrame({"rf_dist": dists, "cell_type": cts}).assign(
        model_name=model_name
    )
    dists.to_csv(table_out, index=False)


if __name__ == "__main__":
    compute_rf()
