import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics as metrics
from anndata import AnnData

from utils import load_config, make_parents, wrap_kwargs


def categorical_obs(adata: AnnData, key: str) -> np.ndarray:
    return np.array(adata.obs[key].astype("category").cat.codes).ravel()


@wrap_kwargs
def compute_scib_metrics(
    *,
    adata_in: str,
    config_in: str,
    table_out: str,
) -> pd.DataFrame:
    """
    Compute integration metrics.

    Parameters
    ----------
    adata_in
        Path to input AnnData object with integrated data.
    config_in
        Path to the dataset configuration file.
    table_out
        Path to write output CSV table with integration metrics.
    """
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    labels_key = config.get("labels_key", None)
    latent_keys = adata.uns["latent_keys"]

    all_metrics = {}
    for latent_key in latent_keys:
        X_latent = adata.obsm[latent_key]
        labels = categorical_obs(adata, labels_key)
        batch = categorical_obs(adata, batch_key)
        sample = categorical_obs(adata, sample_key)

        isolated_label_score = metrics.isolated_labels(X_latent, labels, batch)
        silhouette_label_score = metrics.silhouette_label(
            X_latent, labels, rescale=True
        )
        silhouette_sample_score = metrics.silhouette_label(
            X_latent, sample, rescale=True
        )
        silhouette_batch_score = metrics.silhouette_batch(X_latent, labels, batch, rescale=True)
        # (
        #     nmi_kmeans_label_score,
        #     ari_kmeans_label_score,
        # ) = metrics.nmi_ari_cluster_labels_kmeans(X_latent, labels)
        # (
        #     nmi_leiden_label_score,
        #     ari_leiden_label_score,
        # ) = metrics.nmi_ari_cluster_labels_leiden(X_latent, labels)

        latent_metrics = {
            f"{latent_key}_isolated_label_score": [
                latent_key,
                "isolated_label_score",
                isolated_label_score,
            ],
            f"{latent_key}_silhouette_label_score": [
                latent_key,
                "silhouette_label_score",
                silhouette_label_score,
            ],
            f"{latent_key}_silhouette_sample_score": [
                latent_key,
                "silhouette_sample_score",
                silhouette_sample_score,
            ],
            f"{latent_key}_silhouette_batch_score": [
                latent_key,
                "silhouette_batch_score",
                silhouette_batch_score,
            ],
            # f"{latent_key}_nmi_kmeans_label_score": [
            #     latent_key,
            #     "nmi_kmeans_label_score",
            #     nmi_kmeans_label_score,
            # ],
            # f"{latent_key}_ari_kmeans_label_score": [
            #     latent_key,
            #     "ari_kmeans_label_score",
            #     ari_kmeans_label_score,
            # ],
            # f"{latent_key}_nmi_leiden_label_score": [
            #     latent_key,
            #     "nmi_leiden_label_score",
            #     nmi_leiden_label_score,
            # ],
            # f"{latent_key}_ari_leiden_label_score": [
            #     latent_key,
            #     "ari_leiden_label_score",
            #     ari_leiden_label_score,
            # ],
        }
        all_metrics.update(latent_metrics)

    df = pd.DataFrame.from_dict(
        all_metrics, orient="index", columns=["latent_key", "metric_name", "metric_value"]
    )
    make_parents(table_out)
    df.to_csv(table_out)

    return df


if __name__ == "__main__":
    compute_scib_metrics()
