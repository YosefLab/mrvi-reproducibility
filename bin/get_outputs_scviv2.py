import pandas as pd
import scanpy as sc
from utils import load_config, make_parents, save_pickle, wrap_kwargs


@wrap_kwargs
def get_outputs_scviv2(
    *,
    config_in: str,
    adata_in: str,
    adata_out: str,
    distance_matrices_out: str,
):
    """Get final outputs for scVIV2.

    This includes: cell-type-specific distance matrices.
    """
    config = load_config(config_in)
    groups = config["group_keys"]
    _adata = sc.read_h5ad(adata_in)
    # compute group-specific distance matrices
    cell_specific_dists = _adata.obsm["mrvi_local_sample_dists"]
    group_key_to_dist_keys = {}

    sample_ordering_key = _adata.uns["sample_order_key"]
    sample_ordering = _adata.uns[sample_ordering_key]
    for group_key in groups:
        cell_groups = _adata.obs[group_key]
        group_to_keys = {}
        for group in cell_groups.unique():
            group_mask = cell_groups == group
            dists = cell_specific_dists[group_mask].mean(0)
            dists = pd.DataFrame(dists, index=sample_ordering, columns=sample_ordering)
            group_to_keys[group] = dists
        uns_key = f"mrvi_local_sample_dists_{group_key}"
        _adata.uns[uns_key] = group_to_keys
        group_key_to_dist_keys[group_key] = uns_key
    _adata.uns["group_key_to_dist_keys"] = group_key_to_dist_keys

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    make_parents(distance_matrices_out)
    save_pickle(_adata.uns["group_key_to_dist_keys"], distance_matrices_out)


if __name__ == "__main__":
    get_outputs_scviv2()
