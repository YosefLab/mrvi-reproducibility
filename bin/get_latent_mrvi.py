import mrvi
import scanpy as sc

from utils import load_config, make_parents, wrap_kwargs


def get_latent_mrvi(
    *,
    adata_in: str,
    model_in: str,
    config_in: str,
    adata_out: str,
) -> None:
    """Get latent space from MrVI model."""
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)
    model = mrvi.MrVI.load(model_in)
    latent_key = config.get("latent_key", "X_mrvi_u")

    adata[latent_key] = model.get_latent_representation(adata, give_z=False)

    make_parents(adata_out)
    adata.write(filename=adata_out)

if __name__ == "__main__":
    wrap_kwargs(get_latent_mrvi)()
