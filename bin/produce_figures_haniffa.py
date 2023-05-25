# %%
import glob
import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import seaborn as sns
import xarray as xr
from scib_metrics.benchmark import Benchmarker
from scib_metrics.nearest_neighbors import NeighborsOutput
from scipy.cluster.hierarchy import fcluster
from tree_utils import hierarchical_clustering


def faiss_hnsw_nn(X: np.ndarray, k: int):
    """Gpu HNSW nearest neighbor search using faiss.

    See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    for index param details.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    M = 32
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsOutput(indices=indices, distances=np.sqrt(distances))


def faiss_brute_force_nn(X: np.ndarray, k: int):
    """Gpu brute force nearest neighbor search using faiss."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(X.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsOutput(indices=indices, distances=np.sqrt(distances))


# %%
sc.set_figure_params(dpi_save=500)
plt.rcParams["axes.grid"] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "/data1/scvi-v2-reproducibility/experiments/haniffa"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad("../results/aws_pipeline/haniffa.preprocessed.h5ad")
adata_files = glob.glob("../results/aws_pipeline/data/haniffa.*.final.h5ad")
# %%
# %%
from scvi_v2 import MrVI

model = MrVI.load(
    "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa.scviv2_attention_noprior",
    adata=adata,
)
# model = MrVI.load(
#     "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa.scviv2_attention_no_prior_mog", adata=adata
# )

# %%
donor_info = (
    model.adata.obs.drop_duplicates("_scvi_sample")
    .set_index("_scvi_sample")
    .sort_index()
)
donor_embeds = np.array(model.module.params["qz"]["Embed_0"]["embedding"])

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
donor_embeds_tsne = tsne.fit_transform(donor_embeds)
donor_info.loc[:, ["tsne_1", "tsne_2"]] = donor_embeds_tsne

(
    p9.ggplot(donor_info, p9.aes(x="tsne_1", y="tsne_2", color="Status"))
    + p9.geom_point()
)

# %%
pca = PCA(n_components=2, random_state=42)
donor_embeds_pca = pca.fit_transform(donor_embeds)
donor_info.loc[:, ["pc_1", "pc_2"]] = donor_embeds_pca

(p9.ggplot(donor_info, p9.aes(x="pc_1", y="pc_2", color="Status")) + p9.geom_point())

# %%
sc.set_figure_params(dpi_save=200)
for adata_file in adata_files:
    try:
        adata_ = sc.read_h5ad(adata_file)
    except:
        continue
    print(adata_.shape)
    for obsm_key in adata_.obsm.keys():
        print(obsm_key)
        if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
            print(obsm_key)
            rdm_perm = np.random.permutation(adata.shape[0])[:2000]
            sc.pl.embedding(
                adata_[rdm_perm],
                basis=obsm_key,
                color=["initial_clustering", "Status", "sample_id"],
                ncols=1,
                save="_haniffa_legend.svg",
            )
            sc.pl.embedding(
                adata_[rdm_perm],
                basis=obsm_key,
                color=["initial_clustering", "Status", "sample_id"],
                ncols=1,
                save="_haniffa.png",
            )

# %%
# adata_file =  '../results/aws_pipeline/data/haniffa.scviv2_attention_noprior.final.h5ad'
adata_file = (
    "../results/aws_pipeline/data/haniffa.scviv2_attention_no_prior_mog.final.h5ad"
)
adata_ = sc.read_h5ad(adata_file)
print(adata_.shape)
for obsm_key in adata_.obsm.keys():
    if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
        print(obsm_key)
        rdm_perm = np.random.permutation(adata.shape[0])
        sc.pl.embedding(
            adata_[rdm_perm],
            basis=obsm_key,
            color=["initial_clustering", "Status", "Site", "patient_id"],
            save=f"haniffa.{obsm_key}.png",
            ncols=1,
        )

# %%
# Full SCIB metrics
keys_of_interest = [
    "X_SCVI_clusterkey_subleiden1",
    "X_PCA_clusterkey_subleiden1",
    # "X_scviv2_u",
    # "X_scviv2_mlp_u",
    # "X_scviv2_mlp_smallu_u",
    # "X_scviv2_attention_u",
    # "X_scviv2_attention_smallu_u",
    "X_scviv2_attention_noprior_u",
    "X_scviv2_attention_no_prior_mog_u",
    # "X_PCA_leiden1_subleiden1",
    # "X_SCVI_leiden1_subleiden1",
]
for adata_file in adata_files:
    try:
        adata_ = sc.read_h5ad(adata_file)
    except:
        continue
    obsm_keys = list(adata_.obsm.keys())
    obsm_key_is_relevant = np.isin(obsm_keys, keys_of_interest)
    if obsm_key_is_relevant.any():
        assert obsm_key_is_relevant.sum() == 1
        idx_ = np.where(obsm_key_is_relevant)[0][0]
        print(obsm_keys[idx_])
        obsm_key = obsm_keys[idx_]
        adata.obsm[obsm_key] = adata_.obsm[obsm_key]

adata_sub = adata.copy()
sc.pp.subsample(adata_sub, n_obs=25000)
# %%
bm = Benchmarker(
    adata_sub,
    batch_key="patient_id",
    label_key="initial_clustering",
    embedding_obsm_keys=keys_of_interest,
    # pre_integrated_embedding_obsm_key="X_pca",
    n_jobs=-1,
)

# bm.prepare(neighbor_computer=faiss_brute_force_nn)
bm.prepare()
bm.benchmark()
# %%
bm.plot_results_table(
    min_max_scale=False,
    save_dir=FIGURE_DIR,
)

# %%
donor_info = adata_.obs.drop_duplicates("patient_id").set_index("patient_id")
color_covid = donor_info["Status"].map({"Covid": "#9E1800", "Healthy": "#019E5D"})
color_sex = donor_info["Sex"].map({"Male": "#4791FF", "Female": "#EBA315"})
color_outcome = donor_info["Outcome"].map(
    {"Home": "#466EB8", "Death": "#B80033", "unknown": "#718085"}
)
color_site = donor_info["Site"].map({"Ncl": "#eb4034", "Cambridge": "#3452eb"})
color_age = donor_info["Age_interval"].map(
    {
        "(20, 29]": "#f7fcf5",
        "(30, 39]": "#dbf1d6",
        "(40, 49]": "#aedea7",
        "(50, 59]": "#73c476",
        "(60, 69]": "#37a055",
        "(70, 79]": "#0b7734",
        "(80, 89]": "#00441b",
    }
)
color_worst_status = donor_info["Worst_Clinical_Status"].map(
    {
        "Healthy": "#fffefe",
        "LPS": "#fffefe",
        "Asymptomatic": "#ffd4d4",
        "Mild": "#ffaaaa",
        "Moderate": "#ff7e7e",
        "Severe": "#ff5454",
        "Critical": "#ff2a2a",
        "Death": "#000000",
    }
)
donor_info["color_worst_status"] = color_age
donor_info["color_age"] = color_age
colors = pd.concat([color_age, color_covid, color_worst_status], axis=1)
# %%
dmat_files = glob.glob("../results/aws_pipeline/distance_matrices/haniffa.*.nc")
dmat_files

# %%
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa.scviv2_attention.distance_matrices.nc"
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa.scviv2_attention_no_prior_mog.distance_matrices.nc"
d = xr.open_dataset(dmat_file)
# %%
selected_ct = "CD8"
plt.rcParams["axes.grid"] = False

# %%
mask_samples = donor_info.index
d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
d1 = d1.loc[dict(sample_x=mask_samples)].loc[dict(sample_y=mask_samples)]
Z = hierarchical_clustering(d1.values, method="ward", return_ete=False)
# clusters = fcluster(Z, t=3, criterion="maxclust")
clusters = fcluster(Z, t=2, criterion="maxclust")
colors_ = colors.loc[d1.sample_x.values]
donor_info_ = donor_info.loc[d1.sample_x.values]
donor_info_.loc[:, "cluster_id"] = clusters
colors_.loc[:, "cluster"] = clusters
colors_.loc[:, "cluster"] = colors_.cluster.map(
    {1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5", 4: "#FF8000"}
    # red, blue, white
).values

sns.clustermap(d1.to_pandas(), row_linkage=Z, col_linkage=Z, row_colors=colors_)
plt.savefig(os.path.join(FIGURE_DIR, f"clustermap_{selected_ct}.svg"))

# %%

donor_info_.loc[:, "donor_group"] = donor_info_.cluster_id.map(
    # {
    #     1: "Covid2",
    #     2: "Healthy",
    #     3: "Covid1",
    # }
    {
        1: "Healthy",
        2: "Covid",
    }
)


# %%
adata_log = adata.copy()
sc.pp.normalize_total(adata_log)
sc.pp.log1p(adata_log)
adata_log.obs.loc[:, "donor_status"] = adata_log.obs.patient_id.map(
    donor_info_.loc[:, "donor_group"]
).values
pop = adata_log[(adata_log.obs.initial_clustering == selected_ct)].copy()


# %%
sc.tl.rank_genes_groups(pop, "donor_status", method="wilcoxon", n_genes=1000)
sc.pl.rank_genes_groups_heatmap(
    pop,
    n_genes=10,
    save=f"haniffa.{selected_ct}.clustered.svg",
)

# %%
sc.pl.rank_genes_groups_dotplot(
    pop,
    n_genes=10,
    swap_axes=True,
    save=f"DOThaniffa.{selected_ct}.clustered.svg",
)

# %%
# pop_ = pop[pop.obs.donor_status != "Healthy"].copy()
# pop_.obs.loc[:, "donor_status_"] = pop_.obs.donor_status.astype("str")
# sc.tl.rank_genes_groups(pop_, "donor_status_", method="wilcoxon", n_genes=1000)
# sc.pl.rank_genes_groups_heatmap(
#     pop_,
#     n_genes=10,
#     save=f"haniffa.{selected_ct}.clustered.svg",
# )

# %%
donor_keys = [
    "Sex",
    "Status",
    "age_int",
]

res = model.perform_multivariate_analysis(
    donor_keys=donor_keys,
    adata=None,
    batch_size=256,
    normalize_design_matrix=True,
    offset_design_matrix=False,
)

# %%
es_keys = [f"es_{cov}" for cov in res.covariate.values]
is_sig_keys_ = [f"is_sig_{cov}_" for cov in res.covariate.values]
is_sig_keys = [f"is_sig_{cov}" for cov in res.covariate.values]

adata.obs.loc[:, es_keys] = res["effect_size"].values
adata.obs.loc[:, is_sig_keys_] = res["padj"].values <= 0.1
adata.obs.loc[:, is_sig_keys] = adata.obs.loc[:, is_sig_keys_].astype(str).values

# %%
sc.pl.umap(
    adata,
    color=["initial_clustering", "Status"] + es_keys,
    vmax="p95",
    ncols=1,
    save=f"haniffa_multivariate.png",
)

rdm_idx = np.random.choice(
    np.arange(adata.shape[0]),
    size=2000,
)
sc.pl.umap(
    adata,
    color=["initial_clustering", "Status"] + es_keys,
    vmax="p95",
    ncols=1,
    save=f"haniffa_multivariate.svg",
)


# %%
