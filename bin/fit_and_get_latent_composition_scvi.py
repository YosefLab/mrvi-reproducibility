import numpy as np
import scanpy as sc
import xarray as xr
from anndata import AnnData
from composition_baseline import CompositionBaseline
from sklearn.metrics import pairwise_distances
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_and_get_latent_composition_scvi(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
    distance_matrices_out: str,
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
    distance_matrices_out
        Path to write the distance matrices.
    """
    config = load_config(config_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    group_key = config.get("group_keys", None)
    model_kwargs = config.get("composition_scvi_model_kwargs", {})
    train_kwargs = config.get("composition_scvi_train_kwargs", {})
    adata = sc.read(adata_in)
    _adata = AnnData(obs=adata.obs, uns=adata.uns)
    _adata.uns["model_name"] = "CompositionSCVI"

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

    make_parents(distance_matrices_out)
    freqs_all = composition_scvi.get_local_sample_representation()
    unique_samples = list(adata.obs[sample_key].unique())
    dists = []
    celltypes = []
    for celltype, freqs in freqs_all.items():
        freqs_ = freqs.reindex(unique_samples, fill_value=0)
        print(freqs_.shape)
        celltypes.append(celltype)
        dist_ = pairwise_distances(freqs_, metric="euclidean")[None]
        print(dist_.shape)
        dists.append(dist_)
    dists = np.concatenate(dists, axis=0)
    distances = xr.DataArray(
        dists,
        dims=[group_key, "sample", "sample"],
        coords={group_key: celltypes, "sample": unique_samples},
        name="distance",
    )
    distances.to_netcdf(distance_matrices_out)

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    return _adata


if __name__ == "__main__":
    fit_and_get_latent_composition_scvi()
