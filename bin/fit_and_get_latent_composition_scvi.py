from anndata import AnnData
import scanpy as sc

from utils import load_config, make_parents, wrap_kwargs
from composition_baseline import CompositionBaseline


@wrap_kwargs
def fit_and_get_latent_composition_scvi(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
) -> AnnData:
    """
    Fit and get the latent space from a CompositionScVI model.

    Parameters
    ----------
    adata_in
        Path to the preprocessed AnnData object.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the latent AnnData object.
    """
    config = load_config(config_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    model_kwargs = config.get("composition_scvi_model_kwargs", {})
    train_kwargs = config.get("composition_scvi_train_kwargs", {})
    adata = sc.read(adata_in)
    _adata = AnnData(obs=adata.obs)

    composition_scvi = CompositionBaseline(
        adata,
        batch_key,
        sample_key,
        "SCVI",
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
    )
    composition_scvi.fit()

    latent_key = "X_scVI"
    _adata.obsm[latent_key] = composition_scvi.get_cell_representation()
    _adata.uns["latent_keys"] = [latent_key]

    local_sample_dists_key = "composition_scvi_local_sample_dists"
    _adata.obsm[local_sample_dists_key] = composition_scvi.get_local_sample_distances()
    _adata.uns["local_sample_dists_key"] = local_sample_dists_key

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    return _adata


if __name__ == "__main__":
    fit_and_get_latent_composition_scvi()
