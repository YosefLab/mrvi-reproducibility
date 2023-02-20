import warnings

import ete3
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.linalg import issymmetric
from scipy.spatial.distance import squareform


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
    is_symmetric = issymmetric(dist_mtx)
    has_zero_diag = (dist_mtx.diagonal() == 0).all()
    if not (is_symmetric and has_zero_diag):
        warnings.warn("Distance matrix may be invalid.")
        dist_mtx = dist_mtx - np.diag(dist_mtx.diagonal())
        dist_mtx = (dist_mtx + dist_mtx.T) / 2.0
    red_mtx = squareform(dist_mtx)
    z = linkage(red_mtx, method=method)
    return linkage_to_ete(z)
