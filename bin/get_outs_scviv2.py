import scanpy as sc

from utils import load_config, wrap_kwargs

@wrap_kwargs
def get_outs_scviv2(
    *,
    config_in: str,
    adata_in: str,
):
    config = load_config(config_in)
    groups = config["groups"]
    _adata = sc.read_h5ad(adata_in)
    # compute group-specific distance matrices
    cell_specific_dists = _adata.uns["mrvi_local_sample_dists"]
    for group_key in groups:
        cell_groups = _adata.obs[group_key]
        group_to_keys = {}
        for group in cell_groups.unique():
            group_mask = cell_groups == group
            dists = cell_specific_dists[group_mask].mean(0)
            group_to_keys[group] = dists
        _adata.uns[f"mrvi_local_sample_dists_{group_key}"] = group_to_keys


if __name__ == "__main__":
    get_outs_scviv2()
