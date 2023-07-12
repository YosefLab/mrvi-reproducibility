from typing import Literal, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scvi.model import SCVI
from sklearn.metrics import pairwise_distances

class CompositionBaseline:
    def __init__(
        self,
        adata: AnnData,
        batch_key: Optional[str],
        sample_key: Optional[str],
        rep: Union[Literal["PCA"], Literal["SCVI"]],
        clustering_on: Union[Literal["clusterkey"], Literal["leiden"]] = "leiden",
        cluster_resolution: Optional[float] = None,
        subcluster_resolution: Optional[float] = None,
        model_kwargs=None,
        train_kwargs=None,
    ):
        assert rep in ["PCA", "SCVI"]
        self.adata = adata.copy()
        self.batch_key = batch_key
        self.sample_key = sample_key
        self.model_kwargs = model_kwargs if model_kwargs is not None else dict()
        self.train_kwargs = train_kwargs if train_kwargs is not None else dict()

        self.rep = rep
        self.rep_key = "X_rep"
        self.n_dim = model_kwargs.pop("n_dim", 50)
        self.clustering_on = clustering_on  # one of leiden, cluster_key
        self.cluster_resolution = cluster_resolution or 1.0
        self.cluster_key = model_kwargs.pop("cluster_key", None)
        self.subcluster_key = "leiden_subcluster"
        self.subcluster_resolution = subcluster_resolution or 1.0

    def preprocess_data(self):
        if self.rep == "PCA":
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)

    def fit(self):
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
        return self.adata.obsm[self.rep_key]

    def get_local_sample_representation(self):

        if self.clustering_on == "leiden":
            self.cluster_key = f"leiden_{self.cluster_resolution}"
            sc.pp.neighbors(self.adata, n_neighbors=30, use_rep=self.rep_key)
            sc.tl.leiden(
                self.adata,
                resolution=self.cluster_resolution,
                key_added=self.cluster_key,
            )
        elif (self.clustering_on == "clusterkey") and (self.cluster_key is not None):
            pass
        else:
            raise ValueError(
                "clustering_on must be one of leiden, cluster_key. "
                "If clustering_on is cluster_key, cluster_key must be provided."
            )

        freqs_all = dict()
        for unique_cluster in self.adata.obs[self.cluster_key].unique():
            cell_is_selected = self.adata.obs[self.cluster_key] == unique_cluster
            subann = self.adata[cell_is_selected].copy()

            # Step 1: subcluster
            sc.pp.neighbors(subann, n_neighbors=30, use_rep=self.rep_key)
            sc.tl.leiden(
                subann,
                resolution=self.subcluster_resolution,
                key_added=self.subcluster_key,
            )

            szs = (
                subann.obs.groupby([self.subcluster_key, self.sample_key])
                .size()
                .to_frame("n_cells")
                .reset_index()
            )
            szs_total = szs.groupby(self.sample_key)["n_cells"].sum().rename({self.sample_key: "n_cells_total"}})
            comps = szs.merge(szs_total, left_on=self.sample_key, right_index=True).assign(freqs=lambda x: x.n_cells / x.n_cells_total)
            freqs = comps.loc[:, [self.sample_key, self.subcluster_key, "freqs"]].set_index([self.sample_key, self.subcluster_key]).squeeze().unstack()
            freqs_all[unique_cluster] = freqs
            # n_donors, n_clusters
        return freqs_all

    def get_local_sample_distances(self):
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
