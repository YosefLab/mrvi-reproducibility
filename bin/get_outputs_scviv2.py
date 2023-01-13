import numpy as np
import scanpy as sc
import xarray as xr
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def get_outputs_scviv2(
    *,
    config_in: str,
    adata_in: str,
    adata_out: str,
    cell_distance_matrices_in: str,
    cell_normalized_distance_matrices_in: str,
    distance_matrices_out: str,
    normalized_distance_matrices_out: str,
):
    """Get final outputs for scVIV2.

    This includes: cell-type-specific distance matrices.
    """
    config = load_config(config_in)
    group_key = config["group_keys"]
    _adata = sc.read_h5ad(adata_in)
    # compute group-specific distance matrices
    cell_specific_dists = xr.open_dataset(cell_distance_matrices_in).sample_distances
    cell_specific_normalized_dists = xr.open_dataset(cell_normalized_distance_matrices_in).sample_distances

    sample_ordering = cell_specific_dists.coords["sample"]

    cell_groups = _adata.obs[group_key]
    all_dists = []
    all_normalized_dists = []
    groups = cell_groups.unique()
    for group in groups:
        group_mask = cell_groups == group
        dists = cell_specific_dists.sel(obs_name=group_mask).mean("obs_name")
        normalized_dists = cell_specific_normalized_dists.sel(obs_name=group_mask).mean("obs_name")
        all_dists.append(dists.data)
        all_normalized_dists.append(normalized_dists.data)
    all_dists = np.concatenate(all_dists, axis=0)
    all_dists = xr.DataArray(
        all_dists,
        dims=[group_key, "sample", "sample"],
        coords={group_key: groups, "sample": sample_ordering},
        name="group_distances",
    )
    all_normalized_dists = np.concatenate(all_normalized_dists, axis=0)
    all_normalized_dists = xr.DataArray(
        all_normalized_dists,
        dims=[group_key, "sample", "sample"],
        coords={group_key: groups, "sample": sample_ordering},
        name="normalized_group_distances",
    )

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    make_parents(distance_matrices_out)
    all_dists.to_netcdf(distance_matrices_out)
    make_parents(normalized_distance_matrices_out)
    all_normalized_dists.to_netcdf(normalized_distance_matrices_out)


if __name__ == "__main__":
    get_outputs_scviv2()
