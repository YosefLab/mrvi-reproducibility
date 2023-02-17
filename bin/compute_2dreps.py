import pymde
import scanpy as sc
from anndata import AnnData
from tsnecuda import TSNE

from utils import write_h5ad, wrap_kwargs


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
    adata_in: str,
    adata_out: str,
) -> AnnData:
    """
    Computes low dimensional representations for existing latent representations.

    Iterates over keys in `.uns["latent_keys"]` and computes the following:

    * :meth:`~pymde.pca` with `embedding_dim=2`
    * :meth:`~tsnecuda.TSNE.fit_transform` with `perplexity=30`
    * :meth:`~pymde.preserve_neighbors` with `mde_kwargs`

    Parameters
    ----------
    adata_in
        Path to :class:`anndata.AnnData` containing representations.
    adata_out
        Path to save the output :class:`anndata.AnnData` with 2D representations.

    Returns
    -------
    :class:`anndata.AnnData`
        Annotated data matrix with the following:

        * `.uns["latent_pca_keys"]`: Keys in `.obsm` corresponding to PCA representations.
        * `.uns["latent_tsne_keys"]`: Keys in `.obsm` corresponding to t-SNE representations.
        * `.uns["latent_mde_keys"]`: Keys in `.obsm` corresponding to MDE representations.
    """
    adata = sc.read_h5ad(adata_in)
    latent_keys = adata.uns.get("latent_keys", None)
    
    if latent_keys is None:
        adata.write_h5ad(adata_out)

    latent_pca_keys = []
    latent_tsne_keys = []
    latent_mde_keys = []
    for latent_key in latent_keys:
        latent = adata.obsm[latent_key]

        latent_pca = pymde.pca(latent, embedding_dim=2).cpu().numpy()
        latent_pca_key = f"{latent_key}_pca"
        adata.obsm[latent_pca_key] = latent_pca
        latent_pca_keys.append(latent_pca_key)

        latent_tsne = TSNE(perplexity=30).fit_transform(latent)
        latent_tsne_key = f"{latent_key}_tsne"
        adata.obsm[latent_tsne_key] = latent_tsne
        latent_tsne_keys.append(latent_tsne_key)

        latent_mde = pymde.preserve_neighbors(latent, **mde_kwargs).embed().cpu().numpy()
        latent_mde_key = f"{latent_key}_mde"
        adata.obsm[latent_mde_key] = latent_mde
        latent_mde_keys.append(latent_mde_key)

    adata.uns["latent_pca_keys"] = latent_pca_keys
    adata.uns["latent_tsne_keys"] = latent_tsne_keys
    adata.uns["latent_mde_keys"] = latent_mde_keys

    return write_h5ad(adata, adata_out)


if __name__ == "__main__":
    compute_2dreps()
