# %%
import scanpy as sc
import plotnine as p9
import glob
import os
import pandas as pd
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from ete3 import Tree

from tree_utils import hierarchical_clustering

import xarray as xr
from plot_utils import INCH_TO_CM, ALGO_RENAMER, SHARED_THEME


SUBCLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
]


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
plt.rcParams["axes.grid"] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "../results/milo/experiments/pbmcs68k_for_subsample"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad("../results/milo/data/pbmcs68k_for_subsample.preprocessed.h5ad")

# %%
import scipy.cluster.hierarchy as sch

adata.uns.keys()

# %%
n_subclusters = 8
Z = adata.uns["cluster0_tree_linkage"]
L = adata.uns["cluster0_tree_linkage_clusters"]
M = adata.uns["cluster0_tree_linkage_leaders"]


sch.dendrogram(Z, truncate_mode="lastp", p=n_subclusters, no_plot=False)
plt.savefig(os.path.join(FIGURE_DIR, "pbmcs68k_dendrogram.svg"), dpi=500)
# %%
(
    adata.obs.loc[lambda x: x.subcluster_assignment != "NA"]
    .drop_duplicates("sample_assignment")
    .loc[:, ["sample_assignment", "subcluster_assignment"]]
    .sort_values("subcluster_assignment")
)


# %%
# sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_rep_subclustering")
# sc.tl.umap(adata)

# # %%

# adata.obs.loc[:, "UMAP1"] = adata.obsm["X_umap"][:, 0]
# adata.obs.loc[:, "UMAP2"] = adata.obsm["X_umap"][:, 1]

# fig = (
#     p9.ggplot(adata.obs, p9.aes(x="UMAP1", y="UMAP2", color="subcluster_assignment"))
#     + p9.geom_point(stroke=0.0, size=1.0)
#     + SHARED_THEME
#     + p9.theme_void()
#     + p9.scale_color_manual(
#         values=SUBCLUSTER_COLORS + ["#878787"]
#     )
# )
# fig.save(os.path.join(FIGURE_DIR, "pbmcs68k_subcluster_assignment.png"), dpi=500)
# fig

# # %%

# # %%
# sc.pl.umap(adata, color=["leiden", "sample_assignment", "subcluster_assignment"])
# sc.pl.umap(adata[adata.obs.leiden == "0"], color=["subcluster_assignment"])
# %%
# adata_sub = adata[adata.obs.leiden == "0"].copy()
# sc.pp.neighbors(adata_sub, n_neighbors=30, use_rep="X_scvi")
# sc.tl.umap(adata_sub)
# sc.pl.umap(adata_sub, color=["subcluster_assignment"])

# %%
adata_files = glob.glob("../results/milo/data/pbmcs68k_for_subsample*.final.h5ad")
for adata_file in adata_files:
    adata_ = sc.read_h5ad(adata_file)
    for obsm_key in adata_.obsm.keys():
        if obsm_key.endswith("mde") & ("mrvi" in obsm_key):
            print(obsm_key)
            rdm_perm = np.random.permutation(adata.shape[0])
            adata_.obs["cell_type"] = adata.obs.leiden.copy().astype(str)
            adata_.obs.loc[~adata_.obs.leiden.isin(("0", "1")), "cell_type"] = "other"
            adata_.obs["cell_type"] = adata_.obs["cell_type"].astype("category")
            adata_.uns["cell_type_colors"] = {
                "0": "#7AB5FF",
                "1": "#FF7A7A",
                "other": "lightgray",
            }
            adata_.uns["subcluster_assignment_colors"] = {
                str(i + 1): c for i, c in enumerate(SUBCLUSTER_COLORS)
            }
            adata_.uns["subcluster_assignment_colors"]["NA"] = "lightgray"

            fig = sc.pl.embedding(
                adata_[rdm_perm],
                basis=obsm_key,
                color="cell_type",
                palette=adata_.uns["cell_type_colors"],
                return_fig=True,
                show=False,
            )
            plt.savefig(
                os.path.join(FIGURE_DIR, f"{obsm_key}_ct.svg"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(FIGURE_DIR, f"{obsm_key}_ct.png"),
                bbox_inches="tight",
            )
            plt.clf()
            fig = sc.pl.embedding(
                adata_[rdm_perm],
                basis=obsm_key,
                color="subcluster_assignment",
                palette=adata_.uns["subcluster_assignment_colors"],
                return_fig=True,
                show=False,
            )
            plt.savefig(
                os.path.join(FIGURE_DIR, f"{obsm_key}_subcluster.svg"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(FIGURE_DIR, f"{obsm_key}_subcluster.png"),
                bbox_inches="tight",
            )
            plt.clf()

# # %%
# scibv_files = glob.glob(
#     "../results/milo/metrics/pbmcs68k_for_subsample*mrvi*.csv"
# )
# scib_metrics = pd.DataFrame()
# for dmat_file in scibv_files:
#     d = pd.read_csv(dmat_file, index_col=0)
#     scib_metrics = pd.concat([scib_metrics, d], axis=0)
# scib_metrics.loc[:, "method"] = scib_metrics.latent_key.str.split("_").str[1:-1].apply(lambda x: "_".join(x))
# scib_metrics.loc[:, "latent"] = scib_metrics.latent_key.str.split("_").str[-1]


# %%
# scib_metrics_ = (
#     scib_metrics.copy()
#     .assign(
#         metric_v=lambda x: np.round(x.metric_value, 3).astype(str),
#         latent=lambda x: x.latent.str.replace("subleiden1", "u"),
#     )
# )
# plot_df = (
#     scib_metrics_.loc[lambda x: x.latent == "u"]
# )
# (
#     p9.ggplot(plot_df, p9.aes(x="method", y="metric_name", fill="metric_value"))
#     + p9.geom_tile()
#     + p9.geom_text(p9.aes(label="metric_v"), size=8)
#     + p9.coord_flip()
#     + p9.labs(
#         x="",
#         y="",
#     )
# )

# %%
dmat_files = glob.glob("../results/milo/distance_matrices/pbmcs68k_for_subsample*.nc")
dmat_files

# %%
sample_to_group = (
    adata.obs.query("leiden == '0'")
    .drop_duplicates("sample_assignment")
    .set_index("sample_assignment")
    .sort_index()
    .subcluster_assignment.astype(str)
    .apply(lambda x: np.int32(x))
)

# %%
sample_order = adata.obs["sample_assignment"].cat.categories

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
    if "leiden_1.0" in d:
        continue
    elif "leiden_name" in d:
        ct_coord_name = "leiden_name"
        dmat_name = "leiden"
    else:
        ct_coord_name = "leiden"
        dmat_name = "distance"
    print(distname)
    print(basename)
    if distname == "normalized_distance_matrices":
        continue
    # reorder samples
    d = d.sel(
        sample_x=[str(i) for i in range(1, len(d.sample_x) + 1)],
        sample_y=[str(i) for i in range(1, len(d.sample_y) + 1)],
    )
    res_ = []

    d_foreground = d.loc[{ct_coord_name: "0"}]
    sns.heatmap(d_foreground[dmat_name].values)
    plt.suptitle(f"{modelname}_{distname} CT 0")
    plt.savefig(os.path.join(FIGURE_DIR, f"{modelname}_{distname}_ct0_heatmap.svg"))
    plt.show()
    plt.close()

    for ct in ["1", "2", "3"]:
        d_subsampled = d.loc[{ct_coord_name: ct}]
        sns.heatmap(
            d_subsampled[dmat_name].values,
            vmin=0,
            vmax=d_foreground[dmat_name].values.max(),
        )
        plt.suptitle(f"{modelname}_{distname} CT {ct}")
        plt.savefig(
            os.path.join(FIGURE_DIR, f"{modelname}_{distname}_ct{ct}_heatmap.svg")
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
all_res = pd.DataFrame(all_res).assign(
    Model=lambda x: pd.Categorical(
        x.model.replace(ALGO_RENAMER), categories=ALGO_RENAMER.values()
    ),
)

# %%
model_renamer = {
    # "mrvi_attention_noprior": "MrVI",
    "mrvi_attention_no_prior_mog": "MrVI (MoG)",
    # "mrvi_attention_mog": "MrVI (MoG Large)",
    # "mrvi_attention_no_prior_mog_large": "MrVI (MoG Large w/ Prior)",
    "composition_PCA_clusterkey_subleiden1": "Composition (PCA)",
    "composition_SCVI_clusterkey_subleiden1": "Composition (SCVI)",
}

all_res_ = all_res.loc[lambda x: x.model.isin(model_renamer.keys())].assign(
    Model=lambda x: pd.Categorical(
        x.model.replace(model_renamer), categories=model_renamer.values()
    )
)

# %%
fig = (
    p9.ggplot(all_res_.query("leiden == '0'"), p9.aes(x="Model", y="ratio"))
    + p9.geom_col(fill="#3480eb")
    + p9.theme_classic()
    + p9.scale_x_discrete(limits=list(model_renamer.values())[::-1])
    + p9.coord_flip()
    + p9.theme(
        figure_size=(5.8 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + SHARED_THEME
    + p9.labs(x="", y="Intra-cluster distance ratio")
)
fig.save(os.path.join(FIGURE_DIR, "intra_distance_ratios.svg"))
fig

# %%
mean_d_foreground = (
    all_res.query("leiden == '0'")
    .groupby("model")
    .mean_d.mean()
    .to_frame("foreground_mean_d")
)
relative_d = (
    all_res.query("leiden != '0'")
    .merge(mean_d_foreground, left_on="model", right_index=True)
    .assign(
        relative_d=lambda x: x.mean_d / x.foreground_mean_d,
    )
)
relative_d = relative_d.loc[lambda x: x.model.isin(model_renamer.keys())].assign(
    Model=lambda x: pd.Categorical(
        x.model.replace(model_renamer), categories=list(model_renamer.values())[::-1]
    )
)

fig = (
    p9.ggplot(relative_d, p9.aes(x="Model", y="relative_d"))
    + p9.geom_boxplot(fill="#3480eb")
    + p9.geom_abline(slope=0, intercept=1, color="black", linetype="dashed", size=1)
    + p9.theme_classic()
    + SHARED_THEME
    + p9.theme(
        figure_size=(5.8 * INCH_TO_CM, 4 * INCH_TO_CM),
        legend_position="none",
    )
    + p9.ylim(0, 1.2)
    + p9.coord_flip()
    + p9.labs(y="Inter cluster distance ratio", x="")
)
fig.save(os.path.join(FIGURE_DIR, "inter_distance_ratios.svg"))
fig

# %%
# Plot variance of dist to sample 8 per rank
sample_to_group_and_rank = pd.DataFrame(sample_to_group).reset_index()
sample_to_group_and_rank["sample_assignment_int"] = (
    sample_to_group_and_rank.sample_assignment.astype(int)
)
sample_to_group_and_rank["rank"] = (
    sample_to_group_and_rank.groupby("subcluster_assignment")["sample_assignment_int"]
    .rank(method="dense", ascending=True)
    .astype(int)
)

variance_res = []
for dmat_file in dmat_files:
    try:
        d = xr.open_dataset(dmat_file, engine="netcdf4")
    except:
        continue
    basename = os.path.basename(dmat_file).split(".")
    modelname = basename[1]
    distname = basename[2]
    if "leiden_1.0" in d:
        continue
    elif "leiden_name" in d:
        ct_coord_name = "leiden_name"
        dmat_name = "leiden"
    else:
        ct_coord_name = "leiden"
        dmat_name = "distance"
    if distname == "normalized_distance_matrices":
        continue
    res_ = []

    sample_4_dists = d[dmat_name].sel(sample_y="4")
    sample_4_dists_df = pd.DataFrame(
        sample_4_dists.values, columns=sample_4_dists.sample_x.values
    )
    sample_4_dists_df["leiden"] = d[ct_coord_name].values.astype(int)

    for rank in range(1, 8):
        samples_in_rank = sample_to_group_and_rank[
            (sample_to_group_and_rank["rank"] == rank)
            & (sample_to_group_and_rank["subcluster_assignment"] == 1)
        ]["sample_assignment"].values
        sample_4_dists_df[f"rank_{rank}"] = sample_4_dists_df[samples_in_rank].mean(
            axis=1
        )
    for cluster in sample_to_group_and_rank["subcluster_assignment"].unique():
        samples_in_cluster = sample_to_group_and_rank[
            sample_to_group_and_rank["subcluster_assignment"] == cluster
        ]["sample_assignment"].values
        sample_4_dists_df[f"cluster_{cluster}"] = sample_4_dists_df[
            samples_in_cluster
        ].mean(axis=1)

    sample_4_dists_melt_df = pd.melt(
        sample_4_dists_df,
        id_vars=["leiden"],
        value_vars=[f"rank_{rank}" for rank in range(1, 4)],
        var_name="rank",
    )
    sample_4_dists_melt_df["rank"] = (
        sample_4_dists_melt_df["rank"]
        .map({f"rank_{rank}": rank for rank in range(1, 4)})
        .astype(int)
    )
    sub_melt_df = sample_4_dists_melt_df[
        sample_4_dists_melt_df["leiden"].isin([0, 1, 2])
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12 * INCH_TO_CM, 12 * INCH_TO_CM))
    sns.barplot(x="rank", y="value", hue="leiden", data=sub_melt_df, ax=ax)
    plt.title("Mean Distance to non-subsampled donor in donor group 1")
    plt.ylabel("Mean Distance")
    plt.xlabel("Donor (labeled by subsample rate)")
    L = plt.legend()
    L.get_texts()[0].set_text("Positive cluster")
    L.get_texts()[1].set_text("Subsampled cluster")
    L.get_texts()[2].set_text("Non-subsampled cluster")
    ax.set_xticklabels([0.1, 0.3, 0.6])
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{modelname}_{distname}_sample_4_dists_by_rank.svg"),
        bbox_inches="tight",
    )
    plt.show()
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(12 * INCH_TO_CM, 12 * INCH_TO_CM))
    sample_4_dists_melt_df = pd.melt(
        sample_4_dists_df,
        id_vars=["leiden"],
        value_vars=[f"cluster_{cluster}" for cluster in range(1, 9)],
        var_name="cluster",
    )
    sample_4_dists_melt_df["cluster"] = (
        sample_4_dists_melt_df["cluster"]
        .map({f"cluster_{cluster}": cluster for cluster in range(1, 9)})
        .astype(int)
    )
    sub_melt_df = sample_4_dists_melt_df[
        sample_4_dists_melt_df["leiden"].isin([0, 1, 2])
    ]
    sns.barplot(x="cluster", y="value", hue="leiden", data=sub_melt_df, ax=ax)
    plt.title("Mean Distance to non-subsampled subcluster 1 donor")
    plt.ylabel("Mean Distance")
    plt.xlabel("Donor Subcluster Assignment")
    L = plt.legend()
    L.get_texts()[0].set_text("Positive cluster")
    L.get_texts()[1].set_text("Subsampled cluster")
    L.get_texts()[2].set_text("Non-subsampled cluster")
    plt.savefig(
        os.path.join(
            FIGURE_DIR, f"{modelname}_{distname}_sample_4_dists_by_cluster.svg"
        ),
        bbox_inches="tight",
    )
    print(modelname)
    plt.show()
    plt.clf()
# %%
# Subsample ratio heatmap
fig, ax = plt.subplots(1, 1, figsize=(12 * INCH_TO_CM, 1 * INCH_TO_CM))
ss_ratio_df = pd.DataFrame(
    {
        "sample": [str(i) for i in range(1, 33)],
        "subsample rate": [0.7, 0.8, 0.9, 1.0] * 8,
    }
)
sns.heatmap(
    ss_ratio_df["subsample rate"].values[:, None].T,
    cmap="YlGnBu_r",
    vmin=0,
    vmax=1,
    ax=ax,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"subsample_ratio_heatmap.svg"), bbox_inches="tight"
)
plt.show()
plt.clf()
# # %%

# %%
# DEG Analysis
import mrvi

modelname = "mrvi_attention_mog"

adata_path = os.path.join(
    f"../results/milo/data/pbmcs68k_for_subsample.preprocessed.h5ad"
)
model_path = os.path.join(f"../results/milo/models/pbmcs68k_for_subsample.{modelname}")
adata = sc.read(adata_path)
model = mrvi.MrVI.load(model_path, adata=adata)
model

# %%
model_out_adata_path = os.path.join(
    f"../results/milo/data/pbmcs68k_for_subsample.{modelname}.final.h5ad"
)
model_out_adata = sc.read(model_out_adata_path)
model_out_adata

# %%
# DEG clusters of samples against each other.
# Create one hot obs columns for each group.
for group in sample_to_group.unique():
    adata.obs[f"group_{group}"] = 0
    adata.obs.loc[
        adata.obs.sample_assignment.isin(
            sample_to_group[sample_to_group == group].index
        ),
        f"group_{group}",
    ] = 1
obs_df = adata.obs.copy()
obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
model.donor_info = obs_df.set_index("_scvi_sample").sort_index()
adata.obs

# %%
mv_deg_res = model.perform_multivariate_analysis(
    adata,
    donor_keys=[f"group_{group}" for group in sample_to_group.unique()],
    store_lfc=True,
)
mv_deg_res

# %%
group_no = 1
model_out_adata.obs[f"group_{group_no}_eff_size"] = mv_deg_res.effect_size.sel(
    covariate=f"group_{group_no}"
)
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_u_mde",
    color=f"group_{group_no}_eff_size",
    vmax="p95",
    vmin="p5",
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_group_{group_no}_eff_size.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()

# %%
lfcs = mv_deg_res.lfc.sel(covariate="group_1").values
lfcs_in_one = lfcs[adata.obs.leiden == "0"].mean(0)
# lfcs_in_one = lfcs[adata.obs.leiden == "0"].reshape(-1)
lfcs_in_others = lfcs[adata.obs.leiden != "0"].mean(0)
# lfcs_in_others = lfcs[adata.obs.leiden != "0"].reshape(-1)
mylim = 0.05
bins = np.linspace(-mylim, mylim, 100)
plt.hist(lfcs_in_one, bins=bins, alpha=0.5, label="cluster 1")
plt.hist(lfcs_in_others, bins=bins, alpha=0.5, label="other clusters")
plt.legend()

# %%
# abs_lfcs = np.abs(lfcs)
# abs_lfcs_in_one = abs_lfcs[adata.obs.leiden == "0"].mean(0)
# abs_lfcs_in_others = abs_lfcs[adata.obs.leiden != "0"].mean(0)
# abs_lfcs_in_one = np.abs(lfcs_in_one)
# abs_lfcs_in_others = np.abs(lfcs_in_others)
# is_gene_used = adata.var["is_gene_for_subclustering"]
# plot_df = pd.concat(
#     [
#         pd.DataFrame(
#             {
#                 "abs_lfc": abs_lfcs_in_one,
#                 "is_gene_for_subclustering": is_gene_used,
#                 "cluster": "cluster 0",
#             }
#         ),
#         pd.DataFrame(
#             {
#                 "abs_lfc": abs_lfcs_in_others,
#                 "is_gene_for_subclustering": is_gene_used,
#                 "cluster": "other_clusters",
#             }
#         ),
#     ]
# )

# colors = ["#7AB5FF", "lightgray"]
# fig = (
#     p9.ggplot(
#         plot_df,
#         p9.aes(x="factor(is_gene_for_subclustering)", y="abs_lfc", fill="cluster"),
#     )
#     + p9.geom_boxplot(outlier_alpha=0.0)
#     + p9.ylim(0, 0.02)
#     + p9.scale_fill_manual(values=colors)
#     + p9.coord_flip()
#     + p9.labs(
#         x="",
#         y="Absolute LFC",
#     )
#     + p9.theme_classic()
#     + p9.theme(
#         figure_size=(8 * INCH_TO_CM, 4 * INCH_TO_CM),
#     )
#     + SHARED_THEME
# )
# fig.save(os.path.join(FIGURE_DIR, "DEGs.svg"))
# fig

# %%
from scipy.stats import wilcoxon, ranksums
from statsmodels.stats.multitest import multipletests

adata_pos = adata[adata.obs.leiden == "0"].copy()
sc.pp.normalize_total(adata_pos, target_sum=1e4)
sc.pp.log1p(adata_pos)
# gt_de_analysis = []
# for leiden in adata_pos.obs.sample_group.unique():
#     adata_pos_1 = adata_pos[adata_pos.obs.sample_group == leiden].copy()
#     x_1 = adata_pos_1.X.toarray()
#     adata_pos_2 = adata_pos[adata_pos.obs.sample_group != leiden].copy()
#     x_2 = adata_pos_2.X.toarray()

#     # stat_res = np.array([wilcoxon(x_, y_).pvalue for (x_, y_) in zip(x_1.T, x_2.T)])
#     stat_res = ranksums(x_1, x_2)

#     gt_de_analysis.append(
#         pd.DataFrame(
#             {
#                 "gene": adata_pos_1.var_names,
#                 "pvalue": stat_res.pvalue,
#                 "statistic": stat_res.statistic,
#                 "leiden": leiden,
#                 "padj": multipletests(stat_res.pvalue, method="fdr_bh")[1],
#             }
#         )
#     )
# gene_scores = (
#     gt_de_analysis
#     .groupby("gene")
#     .mean()
# )
# gt_de_analysis = pd.concat(gt_de_analysis)


adata_pos_1 = adata_pos[adata_pos.obs.sample_group == 1].copy()
x_1 = adata_pos_1.X.toarray()
adata_pos_2 = adata_pos[adata_pos.obs.sample_group != 1].copy()
x_2 = adata_pos_2.X.toarray()
stat_res = ranksums(x_1, x_2)

gt_de_analysis = pd.DataFrame(
    {
        "gene": adata_pos_1.var_names,
        "pvalue": stat_res.pvalue,
        "statistic": stat_res.statistic,
        "padj": multipletests(stat_res.pvalue, method="fdr_bh")[1],
    }
)
gt_de_analysis.loc[:, "is_gene_for_subclustering"] = gt_de_analysis.padj < 0.05


# %%
lfc_gp1 = pd.DataFrame(
    {
        "LFC": lfcs_in_one,
        "Group": "Group 1",
        "gene": adata.var_names,
    }
)
lfc_other = pd.DataFrame(
    {
        "LFC": lfcs_in_others,
        "Group": "Other",
        "gene": adata.var_names,
    }
)
lfcs_ = (
    pd.concat([lfc_gp1, lfc_other])
    .assign(
        abs_lfc=lambda x: np.abs(x.LFC),
    )
    # .merge(gt_de_analysis, left_on="gene", right_index=True)
    .merge(gt_de_analysis, on="gene")
    .assign(
        gene_type=lambda x: x.is_gene_for_subclustering.map(
            {
                True: "DE gene",
                False: "Not DE gene",
            }
        )
    )
)
lfcs_

(p9.ggplot(lfcs_, p9.aes(x="LFC", y="statistic", fill="Group")) + p9.geom_point())


# %%
fig = (
    p9.ggplot(lfcs_, p9.aes(x="Group", y="abs_lfc", fill="gene_type"))
    + p9.geom_boxplot(
        outlier_alpha=0.0,
    )
    + p9.ylim(0, 0.12)
    + p9.labs(
        x="",
        y="Absolute LFC",
    )
    + p9.theme_classic()
    + p9.theme(
        figure_size=(5 * INCH_TO_CM, 5 * INCH_TO_CM),
        axis_text=p9.element_text(size=6),
        axis_title=p9.element_text(size=7),
    )
)
fig.save(os.path.join(FIGURE_DIR, "semisynth_DEGs_legend.svg"))

fig = fig + p9.theme(legend_position="none")
fig.save(os.path.join(FIGURE_DIR, "semisynth_DEGs.svg"))

# %%
high_eff_size_cells = model_out_adata[
    model_out_adata.obs["group_1_eff_size"] > 400
].obs_names.to_numpy()
group_lfcs = (
    mv_deg_res.lfc.sel(covariate=f"group_{group_no}")
    .sel(cell_name=high_eff_size_cells)
    .mean("cell_name")
    .to_dataframe()
    .reset_index()
    .assign(
        abs_lfc=lambda x: np.abs(x.lfc),
    )
    .sort_values("abs_lfc", ascending=False)
)
group_lfcs["is_gene_for_subclustering"] = False
group_lfcs.loc[
    np.where(adata.var["is_gene_for_subclustering"])[0], "is_gene_for_subclustering"
] = True
group_lfcs

# %%
highest_group_gene = group_lfcs.iloc[0]["gene"]
model_out_adata.obs.loc[adata.obs_names, highest_group_gene] = adata[
    :, highest_group_gene
].X.toarray()
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_z_mde",
    color=[highest_group_gene, "subcluster_assignment"],
    vmax="p95",
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_group_{group_no}_de_top_gene.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()

# %%
highest_group_gene = group_lfcs.iloc[0]["gene"]
model_out_adata.obs.loc[:, f"{highest_group_gene}_lfc"] = (
    mv_deg_res.lfc.sel(covariate=f"group_{group_no}")
    .sel(gene=highest_group_gene)
    .sel(cell_name=model_out_adata.obs_names.to_numpy())
)
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_z_mde",
    color=[f"{highest_group_gene}_lfc", "subcluster_assignment"],
    vmax="p95",
    vcenter=0,
    cmap="RdBu",
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_group_{group_no}_de_lfc_top_gene.svg"),
    bbox_inches="tight",
)

# %%
# Admissibility for rank 1 subsampled (can subsample further to see effect)
model_out_adata[model_out_adata.obs["sample_assignment"] == "1"]
ball_res = model.get_outlier_cell_sample_pairs(
    flavor="ball", quantile_threshold=0.05, minibatch_size=1000
)
ball_res

# %%
model_out_adata.obs["sample_1_admissibility"] = ball_res.is_admissible.sel(
    sample="1"
).astype(str)
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_u_mde",
    color=["sample_1_admissibility", "leiden"],
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_sample_1_admissibility.svg"),
    bbox_inches="tight",
)

# %%
fig = sc.pl.embedding(
    model_out_adata[model_out_adata.obs["sample_assignment"] == "1"],
    basis="X_mrvi_attention_mog_u_mde",
    color=["leiden"],
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_sample_1_only.svg"),
    bbox_inches="tight",
)

# %%
fig = sc.pl.embedding(
    model_out_adata[model_out_adata.obs["sample_assignment"] == "4"],
    basis="X_mrvi_attention_mog_u_mde",
    color=["leiden"],
    return_fig=True,
    show=False,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_sample_4_only.svg"),
    bbox_inches="tight",
)


# %%
# Differential abundance rank 1 vs rank 4
ap_res = model.get_outlier_cell_sample_pairs(flavor="ap", minibatch_size=1000)
ap_res

# %%
model_out_adata.obs["sample_1_4_da"] = ap_res.log_probs.sel(
    sample="1"
) - ap_res.log_probs.sel(sample="4")
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_u_mde",
    color=["sample_1_4_da", "leiden"],
    vmax="p95",
    vmin="p5",
    vcenter=0,
    cmap="RdBu",
    show=False,
    return_fig=True,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_rank_1_4_da.svg"),
    bbox_inches="tight",
)
# %%
sample_to_metadata = (
    adata.obs.loc[:, ["sample_assignment", "sample_metadata2"]]
    .drop_duplicates()
    .sort_values("sample_assignment")
)
gp1 = sample_to_metadata.query("sample_metadata2")["sample_assignment"].to_numpy()
gp2 = sample_to_metadata.query("~sample_metadata2")["sample_assignment"].to_numpy()

log_p1 = ap_res.log_probs.loc[{"sample": gp1}]
log_p1 = logsumexp(log_p1, axis=1) - np.log(log_p1.shape[1])
log_p1.shape

log_p2 = ap_res.log_probs.loc[{"sample": gp2}]
log_p2 = logsumexp(log_p2, axis=1) - np.log(log_p2.shape[1])

log_ratios = log_p1 - log_p2

model_out_adata.obs.loc[:, "log_ratios"] = log_ratios
# %%
q5, q95 = np.quantile(log_ratios, [0.05, 0.95])
qval = np.maximum(np.abs(q5), np.abs(q95))
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_u_mde",
    color=["log_ratios", "leiden"],
    vmax=qval,
    vmin=-qval,
    vcenter=0,
    cmap="RdBu",
    show=False,
    return_fig=True,
)
plt.savefig(
    os.path.join(FIGURE_DIR, f"pres_{modelname}_comp.svg"),
    bbox_inches="tight",
)

# %%
import scipy.sparse as sp
from tqdm import tqdm

# Comparison to MILO
milo_analysis = pd.read_csv(
    "../results/milo/models/pbmcs68k_for_subsample.MILO.da_analysis.tsv", sep="\t"
)
milo_analysis
milo_neighborhoods = pd.read_csv(
    "../results/milo/models/pbmcs68k_for_subsample.MILO.assignments.mtx",
    sep="\s",
    skiprows=2,
    header=None,
)
cell_idx = milo_neighborhoods.iloc[:, 0].to_numpy() - 1
neigh_idx = milo_neighborhoods.iloc[:, 1].to_numpy() - 1
data_idx = np.ones_like(cell_idx)
mtx = sp.csr_matrix((data_idx, (cell_idx, neigh_idx)))

# %%
milo_cell_res = []
for row in tqdm(mtx):
    row_ = row.toarray().flatten()
    row_ = row_ > 0
    res_ = milo_analysis.iloc[row_]
    lfc = res_["logFC"].mean(0)
    pval = res_["PValue"].min(0)
    milo_cell_res.append(dict(lfc=lfc, pval=pval))
milo_cell_res = pd.DataFrame(milo_cell_res)


# %%
model_out_adata.obs.loc[:, "milo_lfc"] = milo_cell_res["lfc"].values
model_out_adata.obs.loc[:, "mrvi_lfc"] = log_ratios
fig = sc.pl.embedding(
    model_out_adata,
    basis="X_mrvi_attention_mog_u_mde",
    color=["milo_lfc", "mrvi_lfc", "leiden"],
    vmax=qval,
    vmin=-qval,
    vcenter=0,
    cmap="RdBu",
    show=False,
    return_fig=True,
)
fig.show()

# %%
from sklearn.metrics import precision_recall_curve


def plot_pr(y_pred, label=None):
    y_true = (adata.obs["leiden"] == "1").values
    y_pred_ = y_pred.copy()
    y_pred_[np.isnan(y_pred_)] = 0.0
    pre, rec, _ = precision_recall_curve(y_true, -y_pred_)
    # plt.plot(rec, pre, label=label)
    return pd.DataFrame(dict(precision=pre, recall=rec))


# plot_pr(log_ratios, label="MrVI")
# plot_pr(milo_cell_res["lfc"].values, label="MILO")
# plt.legend()
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.xlim(0.01, 1.0)
# plt.savefig(
#     os.path.join(FIGURE_DIR, f"semisynth_pr.svg"),
# )
# plt.show()

# %%
df1 = plot_pr(log_ratios, label="MrVI")
df2 = plot_pr(milo_cell_res["lfc"].values, label="MrVI")
df = pd.concat([df1.assign(model="MrVI"), df2.assign(model="MILO")])

fig = (
    p9.ggplot(df, p9.aes(x="recall", y="precision", color="model"))
    + p9.geom_line()
    + p9.theme_classic()
    + SHARED_THEME
    + p9.theme(
        figure_size=(5.8 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + p9.xlim(0.01, 1.0)
    + p9.labs(
        color="",
    )
)
fig.save(
    os.path.join(FIGURE_DIR, f"semisynth_pr.svg"),
)
fig

# %%
plot_df = pd.concat(
    [
        pd.DataFrame(
            dict(
                es=log_ratios,
                method="MrVI",
                is_cell_affected=(adata.obs["leiden"] == "1").values,
            )
        ),
        pd.DataFrame(
            dict(
                es=milo_cell_res["lfc"].values,
                method="MILO",
                is_cell_affected=(adata.obs["leiden"] == "1").values,
            )
        ),
    ]
)
(
    p9.ggplot(plot_df, p9.aes(x="factor(is_cell_affected)", y="es", fill="method"))
    + p9.geom_boxplot()
)

# %%
lib_size = adata.X.sum(1).A1
log_ratios


df_ = pd.DataFrame(
    {
        "lib_size": lib_size,
        "log_ratios": log_ratios,
        "population": adata.obs["leiden"].values,
    }
)
df_ = (
    df_.assign(
        is_cell_affected=lambda x: x["population"] == "1",
    )
    # .query("population != '1'")
    # .sort_values("log_ratios")
)

(
    p9.ggplot(
        df_.sample(frac=1.0),
        p9.aes(x="lib_size", y="log_ratios", fill="is_cell_affected"),
    )
    + p9.geom_point()
)

# %%
df_.query("~is_cell_affected").sort_values("log_ratios").assign(
    rank_=lambda x: np.arange(len(x))
).plot.scatter("rank_", "lib_size")


# %%
adata_files = glob.glob("../results/milo/data/pbmcs68k_for_subsample*.final.h5ad")
for adata_file in adata_files:
    print(adata_file)

# %%
selected_files = [
    dict(
        filename="../results/milo/data/pbmcs68k_for_subsample.mrvi_attention_mog.final.h5ad",
        modelname="MrVI",
        keyname="X_mrvi_attention_mog_u",
    ),
    dict(
        filename="../results/milo/data/pbmcs68k_for_subsample.composition_SCVI_leiden1_subleiden1.final.h5ad",
        modelname="SCVI",
        keyname="X_SCVI_leiden1_subleiden1",
    ),
    dict(
        filename="../results/milo/data/pbmcs68k_for_subsample.composition_PCA_leiden1_subleiden1.final.h5ad",
        modelname="PCA",
        keyname="X_PCA_leiden1_subleiden1",
    ),
]
embedding_obsm_keys = []
for mydic in selected_files:
    file = mydic["filename"]
    keyname = mydic["keyname"]
    modelname = mydic["modelname"]
    _adata = sc.read_h5ad(file)
    adata.obsm[modelname] = _adata.obsm[keyname]
    embedding_obsm_keys.append(modelname)

# %%
from scib_metrics.benchmark import Benchmarker

bm = Benchmarker(
    adata,
    batch_key="sample_assignment",
    label_key="leiden",
    embedding_obsm_keys=embedding_obsm_keys,
    # pre_integrated_embedding_obsm_key="X_pca",
    n_jobs=-1,
)

# bm.prepare(neighbor_computer=faiss_brute_force_nn)
bm.prepare()
bm.benchmark()
bm.plot_results_table(
    min_max_scale=False,
    save_dir=FIGURE_DIR,
)

# %%
model.history["reconstruction_loss_validation"].plot()
# %%
model.history["elbo_validation"].iloc[20:].plot()
