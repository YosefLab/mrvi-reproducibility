import scanpy as sc
import scvi_v2
from anndata import AnnData
import numpy as np
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def get_latent_scviv2(
    *,
    adata_in: str,
    model_in: str,
    config_in: str,
    adata_out: str,
    cell_representations_out: str,
    distance_matrices_out: str,
    normalized_distance_matrices_out: str,
) -> AnnData:
    """
    Get latent space from a trained MrVI instance.

    Create a new AnnData object with latent spaces as `obsm` layers and
    keys stored in `uns`. Only copies over `obs` from the input AnnData.

    Parameters
    ----------
    adata_in
        Path to the setup AnnData object.
    model_in
        Path to the trained MrVI model.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the latent AnnData object.

    """
    config = load_config(config_in)
    labels_key = config.get("labels_key", None)
    adata = sc.read_h5ad(adata_in)
    model = scvi_v2.MrVI.load(model_in, adata=adata)

    _adata = AnnData(obs=adata.obs, uns=adata.uns)
    _adata.uns["model_name"] = "scVIV2"
    u_latent_key = "X_mrvi_u"
    z_latent_key = "X_mrvi_z"
    _adata.obsm[u_latent_key] = model.get_latent_representation(adata, give_z=False)
    _adata.obsm[z_latent_key] = model.get_latent_representation(adata, give_z=True)
    _adata.uns["latent_keys"] = [u_latent_key, z_latent_key]
    make_parents(adata_out)
    _adata.write(filename=adata_out)
    del _adata

    cell_reps = model.get_local_sample_representation(adata)
    make_parents(cell_representations_out)
    cell_reps.to_netcdf(cell_representations_out)
    del cell_reps

    cell_dists = model.get_local_sample_distances(adata, keep_cell=False, groupby=labels_key)
    make_parents(distance_matrices_out)
    cell_dists.to_netcdf(distance_matrices_out)
    del cell_dists

    cell_normalized_dists = model.get_local_sample_distances(
        adata, use_mean=False, normalize_distances=True, keep_cell=False, groupby=labels_key
    )
    make_parents(normalized_distance_matrices_out)
    cell_normalized_dists.to_netcdf(normalized_distance_matrices_out)
    del cell_normalized_dists

    return adata_out, distance_matrices_out, normalized_distance_matrices_out


if __name__ == "__main__":
    get_latent_scviv2()
