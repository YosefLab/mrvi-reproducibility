import os
from pathlib import Path

import ete3
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from utils import (determine_if_file_empty, load_config, make_parents,
                   wrap_kwargs)


def linkage_to_ete(linkage_obj):
    """Converts to ete3 tree representation."""
    R = to_tree(linkage_obj)
    root = ete3.Tree()
    root.dist = 0
    root.name = "root"
    item2node = {R.get_id(): root}
    to_visit = [R]

    while to_visit:
        node = to_visit.pop()
        cl_dist = node.dist / 2.0

        for ch_node in [node.get_left(), node.get_right()]:
            if ch_node:
                ch_node_id = ch_node.get_id()
                ch_node_name = (
                    f"t{int(ch_node_id) + 1}" if ch_node.is_leaf() else str(ch_node_id)
                )
                ch = ete3.Tree()
                ch.dist = cl_dist
                ch.name = ch_node_name

                item2node[node.get_id()].add_child(ch)
                item2node[ch_node_id] = ch
                to_visit.append(ch_node)
    return root


def hierarchical_clustering(dist_mtx, method="ward"):
    """Perform hierarchical clustering on squared distance matrix."""
    assert dist_mtx.shape[0] == dist_mtx.shape[1]
    red_mtx = squareform(dist_mtx)
    z = linkage(red_mtx, method=method)
    return linkage_to_ete(z)


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

    # Linkage method to use for hierarchical clustering
    clustering_method = config["clustering_method"]
    make_parents(table_out)

    if determine_if_file_empty(distance_matrices_gt):
        Path(table_out).touch()
        return

    gt_mats = xr.open_dataarray(distance_matrices_gt)
    inferred_mats = xr.open_dataarray(distance_matrices)

    aligned_mats = xr.merge([gt_mats, inferred_mats], join="left")
    dists = []
    cts = []

    clusters = aligned_mats.coords[celltype_key].values
    for cluster_name in clusters:
        cts.append(cluster_name)

        dist_gt = aligned_mats.distance_gt.loc[cluster_name].values
        z_gt = hierarchical_clustering(dist_gt, method=clustering_method)
        dist_inferred = aligned_mats.group_distances.loc[cluster_name].values
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
