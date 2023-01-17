import numpy as np
import scanpy as sc
import xarray as xr
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def get_outputs_mrvi(
    *,
    config_in: str,
    adata_in: str,
    adata_out: str,
    distance_matrices_out: str,
):
    """Get final outputs for MrVI.

    This includes: cell-type-specific distance matrices.
    """
    config = load_config(config_in)
    group_key = config["group_keys"]
    _adata = sc.read_h5ad(adata_in)
    # compute group-specific distance matrices
    cell_specific_dists = _adata.obsm["mrvi_local_sample_dists"]

    sample_ordering_key = _adata.uns["sample_order_key"]
    sample_ordering = _adata.uns[sample_ordering_key]

    cell_groups = _adata.obs[group_key]
    all_dists = []
    groups = cell_groups.unique()
    for group in groups:
        group_mask = cell_groups == group
        dists = cell_specific_dists[group_mask].mean(0)
        all_dists.append(dists[None])
    all_dists = np.concatenate(all_dists, axis=0)
    all_dists = xr.DataArray(
        all_dists,
        dims=[group_key, "sample_x", "sample_y"],
        coords={
            group_key: groups,
            "sample_x": sample_ordering,
            "sample_y": sample_ordering,
        },
        name="distance",
    )

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    make_parents(distance_matrices_out)
    all_dists.to_netcdf(distance_matrices_out)


if __name__ == "__main__":
    get_outputs_mrvi()
