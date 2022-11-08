import mrvi
import scanpy as sc
from anndata import AnnData

from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def get_latent_mrvi(
    *,
    adata_in: str,
    model_in: str,
    config_in: str,
    adata_out: str,
) -> AnnData:
    """
    Get latent space from a trained MrVI instance.

    Saves it as a `.obsm` field in a new AnnData object.
    
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
    adata = sc.read_h5ad(adata_in)
    model = mrvi.MrVI.load(model_in, adata=adata)
    latent_key = config.get("latent_key", None)

    adata.obsm[latent_key] = model.get_latent_representation(adata, give_z=False)

    make_parents(adata_out)
    adata.write(filename=adata_out)
    return adata

if __name__ == "__main__":
    get_latent_mrvi()
