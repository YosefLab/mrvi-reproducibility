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
from utils import INCH_TO_CM, ALGO_RENAMER, SHARED_THEME

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

sc.set_figure_params(dpi_save=500)
plt.rcParams['axes.grid'] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "/data1/scvi-v2-reproducibility/experiments/pbmcs68k"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad(
    "../results/aws_pipeline/pbmcs68k.preprocessed.h5ad"
)

# %%
(
    adata.obs.loc[lambda x: x.subcluster_assignment != "NA"]
    .drop_duplicates("sample_assignment")
    .loc[:, ["sample_assignment", "subcluster_assignment"]]
    .sort_values("subcluster_assignment")
)


# %%
sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_scvi")
sc.tl.umap(adata)

# %%
adata.obs.loc[:, "UMAP1"] = adata.obsm["X_umap"][:, 0]
adata.obs.loc[:, "UMAP2"] = adata.obsm["X_umap"][:, 1]

fig = (
    p9.ggplot(adata.obs, p9.aes(x="UMAP1", y="UMAP2", color="subcluster_assignment"))
    + p9.geom_point(stroke=0.0, size=1.0)
    + SHARED_THEME
    + p9.theme_void()
    + p9.scale_color_manual(
        values=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#878787"
        ]
    )
)
fig.save(os.path.join(FIGURE_DIR, "pbmcs68k_subcluster_assignment.png"), dpi=500)

# %%
sc.pl.umap(adata, color=["leiden", "sample_assignment", "subcluster_assignment"])
sc.pl.umap(adata[adata.obs.leiden == "0"], color=["subcluster_assignment"])
# %%
adata_sub = adata[adata.obs.leiden == "0"].copy()
sc.pp.neighbors(adata_sub, n_neighbors=30, use_rep="X_scvi")
sc.tl.umap(adata_sub)
sc.pl.umap(adata_sub, color=["subcluster_assignment"])

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
    print(adata_.shape)
    for obsm_key in adata_.obsm.keys():
        if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
            print(obsm_key)
            rdm_perm = np.random.permutation(adata.shape[0])
            sc.pl.embedding(adata_[rdm_perm], basis=obsm_key, color=["leiden", "subcluster_assignment"])

# %%
scibv_files = glob.glob(
    "../results/aws_pipeline/metrics/pbmc*scviv2*.csv"
)
scib_metrics = pd.DataFrame()
for dmat_file in scibv_files:
    d = pd.read_csv(dmat_file, index_col=0)
    scib_metrics = pd.concat([scib_metrics, d], axis=0)
scib_metrics.loc[:, "method"] = scib_metrics.latent_key.str.split("_").str[1:-1].apply(lambda x: "_".join(x))
scib_metrics.loc[:, "latent"] = scib_metrics.latent_key.str.split("_").str[-1]


# %%
# scib_metrics_scviv2 = scib_metrics[scib_metrics.method.str.startswith("scviv2")].copy()
scib_metrics_ = (
    scib_metrics.copy()
    .assign(
        metric_v=lambda x: np.round(x.metric_value, 3).astype(str),
        latent=lambda x: x.latent.str.replace("subleiden1", "u"),
    )
)
plot_df = (
    scib_metrics_.loc[lambda x: x.latent == "u"]
    # .assign
)
# scib_metrics_ = scib_metrics_.loc[lambda x: x.latent == "u", :]
(
    p9.ggplot(plot_df, p9.aes(x="method", y="metric_name", fill="metric_value"))
    + p9.geom_tile()
    + p9.geom_text(p9.aes(label="metric_v"), size=8)
    # + p9.geom_point(stroke=0, size=3)
    # + p9.facet_grid("latent~metric_name", scales="free")
    + p9.coord_flip()
    + p9.labs(
        x="",
        y="",
    )
    # + p9.theme(
    #     legend_position="none",
    # )
)

# %%
dmat_files = glob.glob(
    "../results/aws_pipeline/distance_matrices/pbmc*.nc"
)
dmat_files

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
    print(d)
    if "leiden_1.0" in d:
        continue
    if "leiden_name" in d:
        ct_coord_name = "leiden_name"
        dmat_name = "leiden"
    else:
        ct_coord_name = "leiden"
        dmat_name = "distance"
    print(basename)
    res_ = []
    for leiden in d[ct_coord_name].values:
        d_ = d.loc[{ct_coord_name: leiden}][dmat_name]
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
    rf_metrics
    .assign(
        modeldistance=lambda x: x.model + "_" + x.dist,
        Model=lambda x: pd.Categorical(x.model.replace(ALGO_RENAMER), categories=ALGO_RENAMER.values()),
    )
)

# %%
rf_metrics

# %%
plot_df = (
    rf_metrics.loc[lambda x: x.dist == "distance_matrices"]
)

fig = (
    p9.ggplot(plot_df, p9.aes(x="Model", y="rf"))
    + p9.geom_col(fill="#3480eb")
    + p9.theme_classic()
    + p9.coord_flip()
    + p9.theme(
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + SHARED_THEME
    + p9.labs(
        x="", y="RF distance"
    )
)
fig.save(os.path.join(FIGURE_DIR, "rf_distance.svg"))
fig

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
sample_order = adata.obs["sample_assignment"].cat.categories

all_res = []
for dmat_file in dmat_files:
# for dmat_file in ["../results/aws_pipeline/distance_matrices/pbmcs68k.composition_PCA_clusterkey_subleiden1.distance_matrices.nc"]:
    print(dmat_file)
    try:
        d = xr.open_dataset(dmat_file, engine="netcdf4")
    except:
        continue
    basename = os.path.basename(dmat_file).split(".")
    modelname = basename[1]
    distname = basename[2]
    if "leiden_1.0" in d:
        continue
    if "leiden_name" in d:
        ct_coord_name = "leiden_name"
        dmat_name = "leiden"
    else:
        ct_coord_name = "leiden"
        dmat_name = "distance"
    print(distname)
    print(basename)
    if distname == "normalized_distance_matrices":
        break
    res_ = []

    # d_foreground = d.loc[{ct_coord_name: "0"}]
    # Z = hierarchical_clustering(d_foreground[dmat_name].values, method="complete", return_ete=False)
    # sns.clustermap(d_foreground[dmat_name].values, row_linkage=Z, col_linkage=Z)
    # plt.suptitle(f"{modelname}_{distname} Cluster 0")

    # d_background = d.loc[{ct_coord_name: "1"}]
    # sns.clustermap(d_background[dmat_name].values, row_linkage=Z, col_linkage=Z)
    # plt.suptitle(f"{modelname}_{distname} Cluster 1")

    # d_background = d.loc[{ct_coord_name: "2"}]
    # sns.clustermap(d_background[dmat_name].values, row_linkage=Z, col_linkage=Z)
    # plt.suptitle(f"{modelname}_{distname} Cluster 2")

    d_foreground = d.loc[{ct_coord_name: "0"}]
    sns.heatmap(d_foreground[dmat_name].values)
    plt.suptitle(f"{modelname}_{distname}")
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{modelname}_{distname}_cluster0_heatmap.svg")
    )
    plt.show()
    plt.close()

    for leiden in d[ct_coord_name].values:
        d_ = d.loc[{ct_coord_name: leiden}][dmat_name]
        d_ = d_.loc[dict(sample_x=sample_order)].loc[dict(sample_y=sample_order)]
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
all_res = (
    pd.DataFrame(all_res)
    .assign(
        Model=lambda x: pd.Categorical(x.model.replace(ALGO_RENAMER), categories=ALGO_RENAMER.values()),
    )
)

# %%
# adat
fig = (
    p9.ggplot(all_res.query("leiden == '0'"), p9.aes(x="Model", y="ratio"))
    + p9.geom_col(fill="#3480eb")
    + p9.theme_classic()
    + p9.coord_flip()
    + p9.theme(
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + SHARED_THEME
    + p9.labs(
        x="", y="Intra-cluster distance ratio"
    )
)
fig.save(os.path.join(FIGURE_DIR, "intra_distance_ratios.svg"))


# %%
# (
#     p9.ggplot(all_res.query("leiden != '0'"), p9.aes(x="model", y="ratio", fill="model"))
#     + p9.geom_boxplot()
# )

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
# relative_d

fig = (
    p9.ggplot(relative_d, p9.aes(x="Model", y="relative_d"))
    + p9.geom_boxplot(fill="#3480eb")
    + p9.geom_abline(slope=0, intercept=1, color="black", linetype="dashed", size=1)
    + p9.theme_classic()
    + SHARED_THEME
    + p9.theme(
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
        legend_position="none",
    )
    + p9.ylim(0, 1.2)
    + p9.coord_flip()
    + p9.labs(y="Inter cluster distance ratio", x="")
)
fig.save(os.path.join(FIGURE_DIR, "inter_distance_ratios.svg"))
fig

# %%
