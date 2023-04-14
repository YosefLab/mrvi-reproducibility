# %%
import scanpy as sc
import plotnine as p9
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from ete3 import Tree
import xarray as xr
from tree_utils import hierarchical_clustering, linkage_to_ete


def compute_ratio(dist, sample_to_mask):
    ratios = []
    for sample_id, sample_dists in enumerate(dist):
        mask_same = sample_to_mask == sample_to_mask[sample_id]
        mask_same[sample_id] = False
        dist_to_same = sample_dists[mask_same].min()

        mask_diff = sample_to_mask != sample_to_mask[sample_id]
        dist_to_diff = sample_dists[mask_diff].min()
        ratios.append(dist_to_same / dist_to_diff)
    return np.array(ratios)


adata = sc.read_h5ad(
    "../results/aws_pipeline/pbmcs68k.preprocessed.h5ad"
)
adata
# %%
sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_scvi")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["leiden", "sample_assignment", "subcluster_assignment"])

# %%
sc.pl.umap(adata[adata.obs.leiden == "0"], color=["subcluster_assignment"])

# %%
tree = adata.uns["cluster0_tree_gt"]
t = Tree(tree)
print(t)

# %%
adata_files = glob.glob(
    "../results/aws_pipeline/data/pbmc*.final.h5ad"
)
for adata_file in adata_files:
    adata_ = sc.read_h5ad(adata_file)
    for obsm_key in adata_.obsm.keys():
        if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
            print(obsm_key)
            rdm_perm = np.random.permutation(adata.shape[0])
            sc.pl.embedding(adata_[rdm_perm], basis=obsm_key, color=["leiden", "subcluster_assignment"])

# %%
scibv_files = glob.glob(
    "../results/aws_pipeline/metrics/pbmc*.scib.csv"
)
scib_metrics = pd.DataFrame()
for dmat_file in scibv_files:
    d = pd.read_csv(dmat_file, index_col=0)
    scib_metrics = pd.concat([scib_metrics, d], axis=0)
scib_metrics.loc[:, "method"] = scib_metrics.latent_key.str.split("_").str[1:-1].apply(lambda x: "_".join(x))
scib_metrics.loc[:, "latent"] = scib_metrics.latent_key.str.split("_").str[-1]


# %%
# scib_metrics_scviv2 = scib_metrics[scib_metrics.method.str.startswith("scviv2")].copy()
scib_metrics_ = scib_metrics.copy()
scib_metrics_.loc[:, "latent"] = scib_metrics_.latent.str.replace("subleiden1", "u")
# scib_metrics_ = scib_metrics_.loc[lambda x: x.latent == "u", :]
(
    p9.ggplot(scib_metrics_, p9.aes(x="method", y="metric_value", fill="method"))
    + p9.geom_point(stroke=0, size=3)
    + p9.facet_grid("latent~metric_name", scales="free")
    + p9.coord_flip()
    + p9.labs(
        x="",
        y="",
    )
    + p9.theme(
        legend_position="none",
    )
)

# %%
dmat_files = glob.glob(
    "../results/aws_pipeline/distance_matrices/pbmc*.nc"
)

# %%
rf_metrics = pd.DataFrame()
for dmat_file in dmat_files:
    print(dmat_file)
    try:
        d = xr.open_dataset(dmat_file, engine="netcdf4")
    except:
        continue
    basename = os.path.basename(dmat_file).split(".")
    modelname = basename[1]
    distname = basename[2]
    if "scviv2" not in modelname:
        continue
    print(basename)
    res_ = []
    for leiden in d.leiden_name.values:
        d_ = d.loc[dict(leiden_name=leiden)]["leiden"]
        tree_ = hierarchical_clustering(d_.values, method="complete")
        Z = hierarchical_clustering(d_.values, method="complete", return_ete=False)

        gt_tree_key = f"cluster{leiden}_tree_gt"
        if gt_tree_key not in adata.uns.keys():
            # print("{} missing in adata.uns".format(gt_tree_key))
            continue
        gt_tree = Tree(adata.uns[gt_tree_key])
        rf_dist = gt_tree.robinson_foulds(tree_)
        norm_rf = rf_dist[0] / rf_dist[1]
        res_.append(dict(rf=norm_rf, leiden=leiden))
    res_ = pd.DataFrame(res_).assign(model=modelname, dist=distname)
    rf_metrics = pd.concat([rf_metrics, res_], axis=0)
rf_metrics = (
    rf_metrics.assign(
        modeldistance=lambda x: x.model + "_" + x.dist,
    )
)

# %%
sample_to_group = (
    adata.obs
    .query("leiden == '0'")
    .drop_duplicates("sample_assignment")
    .set_index("sample_assignment")
    .sort_index()
    .subcluster_assignment
    .astype(str)
    .apply(lambda x: np.int32(x))
)

# %%
all_res = []
for dmat_file in dmat_files:
    print(dmat_file)
    try:
        d = xr.open_dataset(dmat_file, engine="netcdf4")
    except:
        continue
    basename = os.path.basename(dmat_file).split(".")
    modelname = basename[1]
    distname = basename[2]
    if "scviv2" not in modelname:
        continue
    print(distname)
    print(basename)
    if distname == "normalized_distance_matrices":
        continue
    # if distname == "distance_matrices":
    #     continue
    res_ = []

    d_foreground = d.loc[dict(leiden_name="0")]
    Z = hierarchical_clustering(d_foreground["leiden"].values, method="complete", return_ete=False)
    sns.clustermap(d_foreground["leiden"].values, row_linkage=Z, col_linkage=Z)
    plt.suptitle(f"{modelname}_{distname} Cluster 0")

    d_background = d.loc[dict(leiden_name="1")]
    sns.clustermap(d_background["leiden"].values, row_linkage=Z, col_linkage=Z)
    plt.suptitle(f"{modelname}_{distname} Cluster 1")

    d_background = d.loc[dict(leiden_name="2")]
    sns.clustermap(d_background["leiden"].values, row_linkage=Z, col_linkage=Z)
    plt.suptitle(f"{modelname}_{distname} Cluster 2")

    for leiden in d.leiden_name.values:
        d_ = d.loc[dict(leiden_name=leiden)]["leiden"]
        ratios = compute_ratio(d_.values, sample_to_group.values)
        mean_d = d_.values[np.triu_indices(d_.shape[0], k=1)].mean()
        all_res.append(
            dict(
                ratio=ratios.mean(),
                leiden=leiden,
                model=modelname,
                method=distname,
                mean_d=mean_d,
            )
        )
all_res = pd.DataFrame(all_res)

# %%
# adat
(
    p9.ggplot(all_res.query("leiden == '0'"), p9.aes(x="model", y="ratio", fill="model"))
    + p9.geom_col()
)

# %%
(
    p9.ggplot(all_res.query("leiden != '0'"), p9.aes(x="model", y="ratio", fill="model"))
    + p9.geom_boxplot()
)

# %%
mean_d_foreground = all_res.query("leiden == '0'").groupby("model").mean_d.mean().to_frame("foreground_mean_d")
relative_d = (
    all_res
    .query("leiden != '0'")
    .merge(mean_d_foreground, left_on="model", right_index=True)
    .assign(
        relative_d=lambda x: x.mean_d / x.foreground_mean_d,
    )
)
relative_d

# %%
(
    p9.ggplot(relative_d, p9.aes(x="model", y="relative_d", fill="model"))
    + p9.geom_boxplot()
)

