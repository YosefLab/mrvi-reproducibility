import argparse
from pathlib import Path

import ete3
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from utils import load_config, make_parents


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


def hierarchical_clustering(dist_mtx):
    """Perform hierarchical clustering on squared distance matrix."""
    assert dist_mtx.shape[0] == dist_mtx.shape[1]
    red_mtx = squareform(dist_mtx.values)
    z = linkage(red_mtx, method="ward")
    return linkage_to_ete(z)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute RF distance")
    parser.add_argument("--adata_in", type=str, required=True, help="Input adata file")
    parser.add_argument(
        "--config_in", type=str, required=True, help="Path to dataset file"
    )
    parser.add_argument(
        "--table_out", type=str, required=True, help="Output table file"
    )
    return vars(parser.parse_args())


def main(
    adata_in,
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
    adata = sc.read_h5ad(adata_in)
    model_name = adata.uns["model_name"]
    config = load_config(config_in)

    ct_key = config["labels_key"]
    make_parents(table_out)
    inferred_distance_key = adata.uns["group_key_to_dist_keys"][ct_key]

    if "gt_distance_matrix" not in adata.uns:
        Path(table_out).touch()
        return

    dists = []
    for cluster_name in adata.uns["gt_distance_matrix"]:
        dist_gt = adata.uns["gt_distance_matrix"][cluster_name]
        sample_ordering = dist_gt.index.values
        z_gt = hierarchical_clustering(dist_gt)

        dist_inferred = adata.uns[inferred_distance_key][cluster_name]
        dist_inferred = dist_inferred.loc[sample_ordering, sample_ordering]
        z_inferred = hierarchical_clustering(dist_inferred)
        assert (dist_inferred.index == dist_gt.index).all()
        assert (dist_inferred.columns == dist_gt.columns).all()

        rf_dist = z_gt.robinson_foulds(z_inferred)
        norm_rf = rf_dist[0] / rf_dist[1]
        dists.append(norm_rf)
    dists = pd.DataFrame({"rf_dist": dists}).assign(model_name=model_name)
    dists.to_csv(table_out, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
