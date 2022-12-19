import scanpy as sc
import scvi_v2
from anndata import AnnData
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def get_latent_scviv2(
    *,
    adata_in: str,
    model_in: str,
    config_in: str,
    adata_out: str,
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
    load_config(config_in)
    adata = sc.read_h5ad(adata_in)
    model = scvi_v2.MrVI.load(model_in, adata=adata)

    _adata = AnnData(obs=adata.obs)
    u_latent_key = "X_mrvi_u"
    z_latent_key = "X_mrvi_z"
    _adata.obsm[u_latent_key] = model.get_latent_representation(adata, give_z=False)
    _adata.obsm[z_latent_key] = model.get_latent_representation(adata, give_z=True)
    _adata.uns["latent_keys"] = [u_latent_key, z_latent_key]

    local_sample_rep_key = "mrvi_local_sample_rep"
    _adata.obsm[local_sample_rep_key] = model.get_local_sample_representation(adata)
    _adata.uns["local_sample_rep_key"] = local_sample_rep_key

    local_sample_dists_key = "mrvi_local_sample_dists"
    _adata.obsm[local_sample_dists_key] = model.get_local_sample_representation(
        adata, return_distances=True
    )
    _adata.uns["local_sample_dists_key"] = local_sample_dists_key

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    return adata_out


if __name__ == "__main__":
    get_latent_scviv2()
