import scanpy as sc
import pymde
from utils import make_parents, wrap_kwargs


mde_kwargs = dict(
    embedding_dim=2,
    constraint=pymde.Standardized(),
    repulsive_fraction=0.7,
    device="cuda",
    n_neighbors=15,
)


@wrap_kwargs
def compute_2dreps(
    *,
    adata_in,
    adata_out,
):
    """Computes low dimensional representations for existing latent representations.

    Parameters
    ----------
    adata_in :
        Path to anndata containing representations
    adata_out :
        Path to save the file
    """
    adata = sc.read_h5ad(adata_in)
    latent_keys = adata.uns.get("latent_keys", None)
    make_parents(adata_out)

    if latent_keys is None:
        adata.write_h5ad(adata_out)

    latent_pca_keys = []
    latent_mde_keys = []
    for latent_key in latent_keys:
        latent = adata.obsm[latent_key]

        latent_pca = pymde.pca(latent, embedding_dim=2).cpu().numpy()
        latent_pca_key = f"{latent_key}_pca"
        adata.obsm[latent_pca_key] = latent_pca
        latent_pca_keys.append(latent_pca_key)

        latent_mde = pymde.preserve_neighbors(latent, **mde_kwargs).embed().cpu().numpy()
        latent_mde_key = f"{latent_key}_mde"
        adata.obsm[latent_mde_key] = latent_mde
        latent_mde_keys.append(latent_mde_key)
    adata.uns["latent_pca_keys"] = latent_pca_keys
    adata.uns["latent_mde_keys"] = latent_mde_keys

    adata.write_h5ad(adata_out)

if __name__ == "__main__":
    compute_2dreps()
