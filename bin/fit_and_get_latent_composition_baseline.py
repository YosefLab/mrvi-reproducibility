import numpy as np
import scanpy as sc
import xarray as xr
from anndata import AnnData
from composition_baseline import CompositionBaseline
from sklearn.metrics import pairwise_distances
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_and_get_latent_composition_baseline(
    *,
    method_name: str,
    adata_in: str,
    config_in: str,
    adata_out: str,
    distance_matrices_out: str,
) -> AnnData:
    """
    Fit and get the latent space from a CompositionBaseline model.

    Parameters
    ----------
    method_name
        Name of the method used to fit the model. The name of method should reflect the
        set of parameters used to fit the model. It should follow the structure:
        `<dimensionality_reduction_method>_<clustering_on>_sub_<subcluster_resolution>`.
        Since periods will disrupt the scheme of the file name, decimals will be inferred
        if there is a leading 0.
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
    model_kwargs = config.get(f"composition_{method_name}_model_kwargs", {})
    train_kwargs = config.get(f"composition_{method_name}_train_kwargs", {})
    adata = sc.read(adata_in)
    _adata = AnnData(obs=adata.obs, uns=adata.uns)

    rep, clustering_on_full, subcluster_resolution_full = method_name.split("_")

    _adata.uns[
        "model_name"
    ] = f"Composition{rep.upper()}ClusterOn{clustering_on_full.capitalize()}{subcluster_resolution_full.capitalize()}"

    cluster_resolution = None
    clustering_on = clustering_on_full
    if clustering_on.startswith("leiden"):
        cluster_resolution = (
            float(clustering_on[len("leiden") :])
            if clustering_on[len("leiden")] != "0"
            else float(f"0.{clustering_on[len('leiden') + 1:]}")
        )
        clustering_on = "leiden"

    subcluster_resolution = (
        float(subcluster_resolution_full[len("subleiden") :])
        if subcluster_resolution_full[len("subleiden")] != "0"
        else float(f"0.{subcluster_resolution_full[len('subleiden') + 1:]}")
    )

    model_kwargs["cluster_key"] = label_key

    composition_baseline = CompositionBaseline(
        adata,
        batch_key,
        sample_key,
        rep,
        clustering_on=clustering_on,
        cluster_resolution=cluster_resolution,
        subcluster_resolution=subcluster_resolution,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
    )
    composition_baseline.fit()

    latent_key = f"X_{method_name}"
    _adata.obsm[latent_key] = composition_baseline.get_cell_representation()
    _adata.uns["latent_keys"] = [latent_key]
    _adata.uns["cluster_key"] = composition_baseline.cluster_key
    _adata.obs["clustering"] = composition_baseline.adata.obs[
        composition_baseline.cluster_key
    ]

    make_parents(distance_matrices_out)
    freqs_all = composition_baseline.get_local_sample_representation()
    unique_samples = list(adata.obs[sample_key].unique())
    dists = []
    celltypes = []
    for celltype, freqs in freqs_all.items():
        freqs_ = freqs.reindex(unique_samples, fill_value=0)
        celltypes.append(celltype)
        dist_ = pairwise_distances(freqs_, metric="euclidean")[None]
        dists.append(dist_)
    dists = np.concatenate(dists, axis=0)

    dim_label_key = composition_baseline.cluster_key
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
    fit_and_get_latent_composition_baseline()
