import numpy as np
import scanpy as sc
import xarray as xr
from anndata import AnnData
from composition_baseline import CompositionBaseline
from sklearn.metrics import pairwise_distances
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_and_get_latent_composition_pca(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
    distance_matrices_out: str,
) -> AnnData:
    """
    Fit and get the latent space from a CompositionPCA model.

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
    label_key = config.get("labels_key", None)
    model_kwargs = config.get("composition_pca_model_kwargs", {})
    train_kwargs = config.get("composition_pca_train_kwargs", {})
    adata = sc.read(adata_in)
    _adata = AnnData(obs=adata.obs, uns=adata.uns)
    _adata.uns["model_name"] = "CompositionPCA"

    model_kwargs["clustering_on"] = "cluster_key"
    model_kwargs["cluster_key"] = label_key

    composition_pca = CompositionBaseline(
        adata,
        batch_key,
        sample_key,
        "PCA",
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
    )
    composition_pca.fit()

    latent_key = "X_pca"
    _adata.obsm[latent_key] = composition_pca.get_cell_representation()
    _adata.uns["latent_keys"] = [latent_key]

    # local_sample_dists_key = "composition_pca_local_sample_dists"
    # _adata.obsm[local_sample_dists_key] = composition_pca.get_local_sample_distances()
    # _adata.uns["local_sample_dists_key"] = local_sample_dists_key

    make_parents(distance_matrices_out)
    freqs_all = composition_pca.get_local_sample_representation()
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

    dim_label_key = f"{label_key}_name"
    distances = xr.DataArray(
        dists,
        dims=[dim_label_key, "sample_x", "sample_y"],
        coords={
            dim_label_key: celltypes,
            "sample_x": unique_samples,
            "sample_y": unique_samples,
        },
        name="distance",
    )
    distances.to_netcdf(distance_matrices_out)

    make_parents(adata_out)
    _adata.write(filename=adata_out)
    return _adata


if __name__ == "__main__":
    fit_and_get_latent_composition_pca()
