from typing import Literal, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scvi.model import SCVI
from sklearn.metrics import pairwise_distances


class CompositionBaseline:
    """
    Baseline method for computing sample-sample distances within clusters.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Key in `adata.obs` corresponding to batch/site information.
    sample_key
        Key in `adata.obs` corresponding to sample information.
    rep
        Representation to compute distances on. One of the following:

        * `"PCA"`: :func:`~scanpy.pp.pca`
        * `"SCVI"`: :class:`~scvi.model.SCVI`
    model_kwargs
        Keyword arguments to pass into :class:`~scvi.model.SCVI`. Also contains the
        following keys:

        * `"n_dim"`: Number of components to use for PCA
        * `"clustering_on"`: One of `"leiden"` or `"cluster_key"`. If `"leiden"`,
            clusters are computed using :func:`~scanpy.tl.leiden`. If `"cluster_key"`,
            clusters correspond to values in `adata.obs[cluster_key]`.
        * `"cluster_key"`: Key in `adata.obs` corresponding to cluster information.
    train_kwargs
        Keyword arguments to pass into :func:`~scvi.model.SCVI.train`.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: Optional[str],
        sample_key: Optional[str],
        rep: Union[Literal["PCA"], Literal["SCVI"]],
        model_kwargs: dict = None,
        train_kwargs: dict = None,
    ):
        assert rep in ["PCA", "SCVI"]
        self.adata = adata.copy()
        self.batch_key = batch_key
        self.sample_key = sample_key
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.train_kwargs = train_kwargs if train_kwargs is not None else {}

        self.rep = rep
        self.rep_key = "X_rep"
        self.n_dim = self.model_kwargs.pop("n_dim", 50)
        self.clustering_on = self.model_kwargs.pop(
            "clustering_on", "leiden"
        )  # one of leiden, cluster_key
        self.cluster_key = model_kwargs.pop("cluster_key", None)
        self.subcluster_key = "leiden_subcluster"

    def preprocess_data(self):
        """Preprocess data for PCA or SCVI."""
        if self.rep == "PCA":
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
        # TODO: snakemake workflow had hvg selection, should we add it here too?

    def fit(self):
        """Fit PCA or SCVI."""
        self.preprocess_data()
        if self.rep == "PCA":
            sc.pp.pca(self.adata, n_comps=self.n_dim)  # saves "X_pca" in obsm
            self.adata.obsm[self.rep_key] = self.adata.obsm["X_pca"]
        elif self.rep == "SCVI":
            SCVI.setup_anndata(self.adata, batch_key=self.batch_key)
            scvi_model = SCVI(self.adata, **self.model_kwargs)
            scvi_model.train(**self.train_kwargs)
            self.adata.obsm[self.rep_key] = scvi_model.get_latent_representation()

    def get_cell_representation(self):
        """Get cell representations."""
        return self.adata.obsm[self.rep_key]

    def get_local_sample_representation(self):
        """
        Computes local sample representations within each cluster.

        Represents samples as proportion of total sample cells within Leiden subclusters.

        Returns
        -------
        Dictionary
            Keys are unique labels in `adata.obs[self.cluster_key]`.
            Values are :class:`pandas.DataFrame` with rows corresponding to samples
            within that cluster and columns corresponding to the frequency of each
            sample within each subcluster.
        """
        if self.clustering_on == "leiden":
            sc.pp.neighbors(self.adata, n_neighbors=30, use_rep=self.rep_key)
            sc.tl.leiden(self.adata, resolution=1.0, key_added="leiden_1.0")
            self.cluster_key = "leiden_1.0"
        elif (self.clustering_on == "cluster_key") and (self.cluster_key is not None):
            pass
        else:
            raise ValueError(
                "clustering_on must be one of leiden, cluster_key. "
                "If clustering_on is cluster_key, cluster_key must be provided."
            )

        freqs_all = {}
        for unique_cluster in self.adata.obs[self.cluster_key].unique():
            cell_is_selected = self.adata.obs[self.cluster_key] == unique_cluster
            subann = self.adata[cell_is_selected].copy()

            # Step 1: subcluster
            sc.pp.neighbors(subann, n_neighbors=30, use_rep=self.rep_key)
            sc.tl.leiden(subann, resolution=1.0, key_added=self.subcluster_key)

            # Step 2: compute number of cells per subcluster-sample pair
            szs = (
                subann.obs.groupby([self.subcluster_key, self.sample_key])
                .size()
                .to_frame("n_cells")
                .reset_index()
            )
            # Step 3: compute total number of cells per sample
            szs_total = (
                szs.groupby(self.subcluster_key)
                .sum()
                .rename(columns={"n_cells": "n_cells_total"})
            )
            # Step 4: compute frequency of each subcluster per sample
            comps = szs.merge(szs_total, on=self.subcluster_key).assign(
                freqs=lambda x: x.n_cells / x.n_cells_total
            )
            # Step 5: compute representation of each sample as the vector
            # of frequencies in each subcluster
            freqs = (
                comps.loc[:, [self.sample_key, self.subcluster_key, "freqs"]]
                .set_index([self.sample_key, self.subcluster_key])
                .squeeze()
                .unstack()
            )
            freqs_ = freqs
            freqs_all[unique_cluster] = freqs_
            # n_donors, n_clusters
        return freqs_all

    def get_local_sample_distances(self):
        """
        Computes local sample distances for each cell.

        Sample-sample distances for a particular cell are computed as the Euclidean
        distance between the sample representations within the cell's cluster.
        Distances for samples absent in a cluster are imputed as `1.0`.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of shape `(n_cells, n_samples, n_samples)`. Cells in the same cluster
            have the same distance matrix.
        """
        freqs_all = self.get_local_sample_representation()
        n_sample = self.adata.obs[self.sample_key].nunique()
        sample_order = self.adata.obs[self.sample_key].unique()
        local_dists = np.zeros((self.adata.n_obs, n_sample, n_sample))

        for cluster, freqs in freqs_all.items():
            cell_is_selected = self.adata.obs[self.cluster_key] == cluster
            ordered_freqs = freqs.reindex(sample_order).fillna(1.0)
            cluster_reps = ordered_freqs.values
            cluster_dists = pairwise_distances(cluster_reps, metric="euclidean")
            local_dists[cell_is_selected] = cluster_dists

        return local_dists
