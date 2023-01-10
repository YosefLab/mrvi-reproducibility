import scanpy as sc
from tsnecuda import TSNE
from utils import make_parents, wrap_kwargs


@wrap_kwargs
def compute_2drep(
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
    latent_2d_keys = []
    if latent_keys is None:
        adata.write_h5ad(adata_out)
    for latent_key in latent_keys:
        latent = adata.obsm[latent_key]
        latent2d = TSNE(perplexity=30).fit_transform(latent)
        latent_2d_key = f"{latent_key}_2d"
        adata.obsm[latent_2d_key] = latent2d
        latent_2d_keys.append(latent_2d_key)
    adata.uns["latent_2d_keys"] = latent_2d_keys
    adata.write_h5ad(adata_out)


if __name__ == "__main__":
    compute_2drep()
