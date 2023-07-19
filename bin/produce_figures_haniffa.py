# %%
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from biothings_client import get_client
import xarray as xr
from sklearn.manifold import TSNE
from scib_metrics.benchmark import Benchmarker
from scib_metrics.nearest_neighbors import NeighborsOutput
from scipy.cluster.hierarchy import fcluster
from tree_utils import hierarchical_clustering
from biothings_client import get_client


INCH_TO_CM = 1 / 2.54

# %%
metad = pd.read_excel(
    "/data1/datasets/41591_2021_1329_MOESM3_ESM.xlsx",
    sheet_name=1,
    header=1,
    index_col=0,
)
metad.loc[:, "DFO"] = metad["Days from onset"].replace(
    {
        "Healthy": None,
        "Non_covid": None,
        "Not_known": None,
        "LPS": None,
    }
)
metad.loc[:, "DFO"] = metad["DFO"].astype(float)


SHARED_THEME = p9.theme(
    strip_background=p9.element_blank(),
    subplots_adjust={"wspace": 0.3},
    # panel_background=p9.element_blank(),
    axis_text=p9.element_text(family="sans-serif", size=7),
    axis_title=p9.element_text(family="sans-serif", size=8),
)


# %%
sc.set_figure_params(dpi_save=500)
plt.rcParams["axes.grid"] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "/data1/scvi-v2-reproducibility/experiments/haniffa2"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad("../results/aws_pipeline/haniffa2.preprocessed.h5ad")
adata.obs = adata.obs.merge(
    metad, left_on="sample_id", right_index=True, how="left", suffixes=("", "_y")
)
adata.obs.loc[:, "age_group"] = adata.obs.Age >= 60
adata.obs.loc[:, "age_group"] = adata.obs.age_group.replace(
    {True: ">=60", False: "<60"}
).astype("category")
adata_files = glob.glob("../results/aws_pipeline/data/haniffa2.*.final.h5ad")

# %%
mg = get_client("gene")
gene_conversion = mg.querymany(
    adata.var_names,
    scopes="symbol",
    fields="ensembl.gene,entrezgene",
    species="human",
    returnall=False,
    as_dataframe=True,
    df_index=False,
)
# ensembl_gene = (
#     gene_conversion[["query", "ensembl.gene", "_score"]]
#     .dropna()
#     .sort_values("_score", ascending=False)
#     .groupby("query")
#     .first()
# )

# adata.var.loc[:, "ensembl_gene"] = ensembl_gene.reindex(adata.var_names)["ensembl.gene"].values

# %%
from scvi_v2 import MrVI

model = MrVI.load(
    "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa2.scviv2_attention_mog",
    adata=adata,
)
# model = MrVI.load(
#     "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa.scviv2_attention_no_prior_mog", adata=adata
# )

# %%
model.history["elbo_validation"].iloc[50:].plot()

# %%
# import flax.linen as nn

# train_kwargs = {
#     "max_epochs": 400,
#     "batch_size": 1024,
#     "early_stopping": True,
#     "early_stopping_patience": 30,
#     "check_val_every_n_epoch": 1,
#     "early_stopping_monitor": "reconstruction_loss_validation",
#     "plan_kwargs": {
#         "n_epochs_kl_warmup": 50,
#         "lr": 3e-3
#     }
# }

# model_kwargs = {
#     "n_latent": 100,
#     "n_latent_u": 10,
#     "qz_nn_flavor": "attention",
#     "px_nn_flavor": "attention",
#     "qz_kwargs": {
#         "use_map": True,
#         "stop_gradients": False,
#         "stop_gradients_mlp": True,
#     },
#     "px_kwargs": {
#         "stop_gradients": False,
#         "stop_gradients_mlp": True,
#         "h_activation": nn.softmax,
#         "low_dim_batch": True,
#     },
#     "learn_z_u_prior_scale": False,
#     "z_u_prior": False,
#     "u_prior_mixture": True,
#     "u_prior_mixture_k": 20,
# }

# modelb = MrVI(
#     adata, **model_kwargs
# )
# modelb.train(**train_kwargs)

# %%
ax = model.history["reconstruction_loss_validation"].plot()
# modelb.history["reconstruction_loss_validation"].plot(ax=ax)
# modelb.history["reconstruction_loss_validation"].iloc[10:].plot()

# %%
donor_info = (
    model.adata.obs.drop_duplicates("_scvi_sample")
    .set_index("_scvi_sample")
    .sort_index()
    .merge(
        metad, left_on="sample_id", right_index=True, how="left", suffixes=("", "_y")
    )
)
donor_embeds = np.array(model.module.params["qz"]["Embed_0"]["embedding"])


# tsne = TSNE(n_components=2, random_state=42, metric="cosine")
# donor_embeds_tsne = tsne.fit_transform(donor_embeds)
# donor_info.loc[:, ["tsne_1", "tsne_2"]] = donor_embeds_tsne

# (
#     p9.ggplot(donor_info, p9.aes(x="tsne_1", y="tsne_2", color="Status"))
#     + p9.geom_point()
# )

# %%
pca = PCA(n_components=2, random_state=42)
donor_embeds_pca = pca.fit_transform(donor_embeds)
# donor_info.loc[:, ["pc_1", "pc_2"]] = donor_embeds_pca
donor_info.loc[:, "pc_1"] = donor_embeds_pca[:, 0]
donor_info.loc[:, "pc_2"] = donor_embeds_pca[:, 1]

(p9.ggplot(donor_info, p9.aes(x="pc_1", y="pc_2", color="Status")) + p9.geom_point())

# %%
(p9.ggplot(donor_info, p9.aes(x="pc_1", y="pc_2", color="age_int")) + p9.geom_point())

# %%
# (p9.ggplot(donor_info, p9.aes(x="pc_1", y="pc_2", color="DFO")) + p9.geom_point())

# %%
# sc.set_figure_params(dpi_save=200)
# for adata_file in adata_files:
#     try:
#         adata_ = sc.read_h5ad(adata_file)
#     except:
#         continue
#     print(adata_.shape)
#     for obsm_key in adata_.obsm.keys():
#         print(obsm_key)
#         if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
#             print(obsm_key)
#             rdm_perm = np.random.permutation(adata.shape[0])[:2000]
#             # sc.pl.embedding(
#             #     adata_[rdm_perm],
#             #     basis=obsm_key,
#             #     color=["initial_clustering", "Status", "sample_id"],
#             #     ncols=1,
#             #     save="_haniffa_legend.svg",
#             # )
#             sc.pl.embedding(
#                 adata_[rdm_perm],
#                 basis=obsm_key,
#                 color=["initial_clustering", "Status", "sample_id"],
#                 ncols=1,
#                 save="_haniffa.png",
#             )

# %%
# adata_file =  '../results/aws_pipeline/data/haniffa.scviv2_attention_noprior.final.h5ad'
adata_file = "../results/aws_pipeline/data/haniffa2.scviv2_attention_no_prior_mog_large.final.h5ad"
adata_ = sc.read_h5ad(adata_file)
print(adata_.shape)
for obsm_key in adata_.obsm.keys():
    if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
        print(obsm_key)
        fig = sc.pl.embedding(
            adata_,
            basis=obsm_key,
            color=["initial_clustering", "Status", "Site", "patient_id"],
            ncols=1,
            show=False,
            return_fig=True,
        )
        fig.savefig(os.path.join(FIGURE_DIR, f"haniffa.{obsm_key}.svg"))
        plt.clf()


# %%
keys_of_interest = {
    "X_SCVI_clusterkey_subleiden1": "SCVI",
    "X_PCA_clusterkey_subleiden1": "PCA",
    "X_scviv2_attention_mog_u": "MrVI",
}
for adata_file in adata_files:
    try:
        adata_ = sc.read_h5ad(adata_file)
    except:
        continue
    obsm_keys = list(adata_.obsm.keys())
    obsm_key_is_relevant = np.isin(obsm_keys, list(keys_of_interest.keys()))
    if obsm_key_is_relevant.any():
        assert obsm_key_is_relevant.sum() == 1
        idx_ = np.where(obsm_key_is_relevant)[0][0]
        print(obsm_keys[idx_])
        obsm_key = obsm_keys[idx_]

        clean_key = keys_of_interest[obsm_key]
        adata.obsm[clean_key] = adata_.obsm[obsm_key]

adata_sub = adata.copy()
sc.pp.subsample(adata_sub, n_obs=25000)
# %%
bm = Benchmarker(
    adata_sub,
    batch_key="patient_id",
    label_key="initial_clustering",
    embedding_obsm_keys=list(keys_of_interest.values()),
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
donor_info_ = donor_info.set_index("sample_id")


covid_legend = {"Covid": "#9E1800", "Healthy": "#019E5D"}
sex_legend = {"Male": "#4791FF", "Female": "#EBA315"}
outcome_legend = {"Home": "#466EB8", "Death": "#B80033", "unknown": "#718085"}
site_legend = {"Ncl": "#eb4034", "Cambridge": "#3452eb"}
age_legend = {
    "(20, 29]": "#f7fcf5",
    "(30, 39]": "#dbf1d6",
    "(40, 49]": "#aedea7",
    "(50, 59]": "#73c476",
    "(60, 69]": "#37a055",
    "(70, 79]": "#0b7734",
    "(80, 89]": "#00441b",
}
worst_clinical_status_legend = {
    "Healthy": "#fffefe",
    "LPS": "#fffefe",
    "Asymptomatic": "#ffd4d4",
    "Mild": "#ffaaaa",
    "Moderate": "#ff7e7e",
    "Severe": "#ff5454",
    "Critical": "#ff2a2a",
    "Death": "#000000",
}
all_legends = {
    "covid": covid_legend,
    "sex": sex_legend,
    "outcome": outcome_legend,
    "site": site_legend,
    "age": age_legend,
    "worst_clinical_status": worst_clinical_status_legend,
}

color_covid = donor_info_["Status"].map(covid_legend)
color_sex = donor_info_["Sex"].map(sex_legend)
color_outcome = donor_info_["Outcome"].map(outcome_legend)
color_site = donor_info_["Site"].map(site_legend)
color_age = donor_info_["Age_interval"].map(age_legend)
color_worst_status = donor_info_["Worst_Clinical_Status"].map(
    worst_clinical_status_legend
)
donor_info_["color_worst_status"] = color_age
donor_info_["color_age"] = color_age

donor_info_["_DFO"] = (donor_info_["DFO"] - donor_info_["DFO"].min()) / (
    donor_info_["DFO"].max() - donor_info_["DFO"].min()
)
dfo_colors = plt.get_cmap("viridis")(donor_info_["_DFO"])
dfo_colors = [rgb2hex(rgb) for rgb in dfo_colors]
donor_info_["DFO_color"] = dfo_colors
dfo_colors = donor_info_["DFO_color"]

colors = pd.concat([color_age, dfo_colors, color_covid, color_worst_status], axis=1)

# %%
from matplotlib.patches import Patch

for legend_name, my_legend in all_legends.items():
    handles = [Patch(facecolor=hex2color(my_legend[name])) for name in my_legend]
    plt.legend(
        handles,
        my_legend.keys(),
        title="Species",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    plt.savefig(
        os.path.join(FIGURE_DIR, f"legend_{legend_name}.svg"), bbox_inches="tight"
    )


# %%
adata_file = "../results/aws_pipeline/data/haniffa2.scviv2_attention_mog.final.h5ad"
adata_embs = sc.read_h5ad(adata_file)

# %%
zs = model.get_local_sample_representation()
u = model.get_latent_representation(give_z=False, use_mean=True)

# %%
# eps_ = zs.values - u[:, None]
eps_ = zs.values
eps_ = (eps_ - eps_.mean(axis=1, keepdims=True)) / eps_.std(axis=1, keepdims=True)
eps_ = eps_.reshape(eps_.shape[0], -1)
pca_eps = PCA(n_components=1)
eps_pca = pca_eps.fit_transform(eps_)
adata_embs.obs["eps_pca"] = eps_pca

# %%
print(adata_embs.shape)
for obsm_key in adata_embs.obsm.keys():
    if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
        print(obsm_key)
        # rdm_perm = np.random.permutation(adata.shape[0])
        fig = sc.pl.embedding(
            # adata_embs[rdm_perm],
            adata_embs,
            basis=obsm_key,
            color=["initial_clustering", "Status", "eps_pca", "patient_id"],
            ncols=1,
            show=True,
            return_fig=True,
        )
        fig.savefig(
            os.path.join(FIGURE_DIR, f"embedding_{obsm_key}.svg"), bbox_inches="tight"
        )

# %%
dmat_files = glob.glob("../results/aws_pipeline/distance_matrices/haniffa2.*.nc")
dmat_files

# %%
# dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention.distance_matrices.nc"
# dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_no_prior_mog_large.distance_matrices.nc"
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_mog.distance_matrices.nc"
d = xr.open_dataset(dmat_file)


# %%
pop1 = [
    "MH8919283",
    "newcastle65",
    "MH8919332",
    "MH8919282",
    "MH8919176",
    "MH8919177",
    "MH8919179",
    "newcastle74",
    "MH8919178",
    "MH8919333",
    "MH8919226",
    "MH8919227",
]

pop2 = [
    "MH9179825",
    "MH9143324",
    "MH8919330",
    "MH9143274",
    "newcastle20",
    "MH9143420",
    "newcastle004v2",
    "newcastle21",
    "newcastle21v2",
]

# %%
plt.rcParams["axes.grid"] = False

VMIN = 0.0
VMAX = 1.0
selected_cts = [
    "CD14",
    # "B_cell",
    # "CD4",
]
n_clusters = [
    3,
    # 3,
    # 2,
]

for idx, (selected_ct, n_cluster) in enumerate(zip(selected_cts, n_clusters)):
    mask_samples = donor_info_.index
    d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
    d1 = d1.loc[dict(sample_x=mask_samples)].loc[dict(sample_y=mask_samples)]
    Z = hierarchical_clustering(d1.values, method="ward", return_ete=False)

    colors_ = colors.loc[d1.sample_x.values]
    donor_info_ = donor_info_.loc[d1.sample_x.values]

    # Get clusters
    clusters = fcluster(Z, t=n_cluster, criterion="maxclust")
    donor_info_.loc[:, "donor_group"] = clusters
    colors_.loc[:, "cluster"] = clusters
    colors_.loc[:, "cluster"] = colors_.cluster.map(
        {1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5", 4: "#FF8000"}
        # red, blue, white
    ).values

    donor_cluster_key = f"donor_clusters_{selected_ct}"
    adata.obs.loc[:, donor_cluster_key] = adata.obs.patient_id.map(
        donor_info_.loc[:, "donor_group"]
    ).values
    adata.obs.loc[:, donor_cluster_key] = "cluster " + adata.obs.loc[
        :, donor_cluster_key
    ].astype(str)

    sns.clustermap(
        d1.to_pandas(),
        row_linkage=Z,
        col_linkage=Z,
        row_colors=colors_,
        vmin=VMIN,
        vmax=VMAX,
        # cmap="YlGnBu",
        yticklabels=True,
        figsize=(20, 20),
    )
    plt.savefig(os.path.join(FIGURE_DIR, f"clustermap_{selected_ct}.svg"))

    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log)
    sc.pp.log1p(adata_log)
    pop = adata_log[(adata_log.obs.initial_clustering == selected_ct)].copy()

    sc.tl.rank_genes_groups(
        pop,
        donor_cluster_key,
        method="t-test",
        n_genes=1000,
    )
    fig = sc.pl.rank_genes_groups_dotplot(
        pop,
        n_genes=5,
        min_logfoldchange=0.5,
        swap_axes=True,
        return_fig=True,
    )
    fig.savefig(os.path.join(FIGURE_DIR, f"DOThaniffa.{selected_ct}.clustered.svg"))

# %%
ood_res = model.get_outlier_cell_sample_pairs(
    subsample_size=5000, minibatch_size=256, quantile_threshold=0.05
)


# %%
n_admissible = (
    ood_res.to_dataframe()["is_admissible"].unstack().sum(1).to_frame("n_admissible")
)
obs_ = adata.obs.join(n_admissible)

atleast = 25
n_donors_with_atleast = (
    obs_.groupby(["initial_clustering"])
    .apply(lambda x: x.loc[x.n_admissible >= atleast].patient_id.nunique())
    .to_frame("n_donors_with_atleast")
)
n_donors_with_atleast

n_pred_donors = (
    obs_.groupby(["initial_clustering"]).n_admissible.mean().to_frame("n_pred_donors")
)

joined = n_donors_with_atleast.join(n_pred_donors)

# %%
(
    p9.ggplot(
        joined.reset_index(), p9.aes(x="n_donors_with_atleast", y="n_pred_donors")
    )
    + p9.geom_point()
    + p9.geom_abline(intercept=0, slope=1)
)

# %%
# %%
# adata_embs.obs.loc[:, "n_valid_donors"] = res["is_admissible"].values.sum(axis=1)
# for obsm_key in adata_embs.obsm.keys():
#     if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
#         print(obsm_key)
#         sc.pl.embedding(
#             # adata_embs[rdm_perm],
#             adata_embs,
#             basis=obsm_key,
#             color=["initial_clustering","n_valid_donors"],
#             save=f"haniffa.{obsm_key}.svg",
#             ncols=1,
#         )

# %%
donor_keys = [
    "Sex",
    "Status",
    "age_group",
]
# %%
res = model.perform_multivariate_analysis(
    donor_keys=donor_keys,
    adata=None,
    batch_size=256,
    normalize_design_matrix=True,
    offset_design_matrix=False,
    filter_donors=True,
    subsample_size=500,
    quantile_threshold=0.05,
)


# %%
es_keys = [f"es_{cov}" for cov in res.covariate.values]
is_sig_keys_ = [f"is_sig_{cov}_" for cov in res.covariate.values]
is_sig_keys = [f"is_sig_{cov}" for cov in res.covariate.values]

adata.obs.loc[:, es_keys] = res["effect_size"].values
adata.obs.loc[:, is_sig_keys_] = res["padj"].values <= 0.1
adata.obs.loc[:, is_sig_keys] = adata.obs.loc[:, is_sig_keys_].astype(str).values

adata_embs.obs.loc[:, es_keys] = res["effect_size"].values
adata_embs.obs.loc[:, is_sig_keys_] = res["padj"].values <= 0.1
adata_embs.obs.loc[:, is_sig_keys] = (
    adata_embs.obs.loc[:, is_sig_keys_].astype(str).values
)

# %%
for obsm_key in adata_embs.obsm.keys():
    if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
        print(obsm_key)
        sc.pl.embedding(
            # adata_embs[rdm_perm],
            adata_embs,
            basis=obsm_key,
            color=["initial_clustering"] + es_keys,
            save=f"haniffa.{obsm_key}.svg",
            ncols=1,
        )

# %%
plot_df = adata.obs.reset_index().melt(
    id_vars=["index", "initial_clustering"],
    value_vars=es_keys,
)

n_points = adata.obs.initial_clustering.value_counts().to_frame("n_points")
plot_df = plot_df.merge(
    n_points, left_on="initial_clustering", right_index=True, how="left"
).assign(
    variable_name=lambda x: x.variable.map(
        {
            "es_SexMale": "Sex",
            "es_StatusHealthy": "Status",
            "es_age_group>=60": "Age",
        }
    )
)


# %%
INCH_TO_CM = 1 / 2.54

plt.rcParams["svg.fonttype"] = "none"

fig = (
    p9.ggplot(
        # plot_df.loc[lambda x: x.n_points > 7000, :],
        plot_df,
        p9.aes(x="initial_clustering", y="value"),
    )
    + p9.geom_boxplot(outlier_shape="", fill="#3492eb")
    # + p9.facet_wrap("~variable", scales="free_x")
    + p9.facet_wrap("~variable_name")
    + p9.coord_flip()
    + p9.labs(y="Effect size", x="")
    + p9.theme(
        figure_size=(10 * INCH_TO_CM, 5 * INCH_TO_CM),
        axis_text=p9.element_text(size=7),
    )
)
fig.save(os.path.join(FIGURE_DIR, "haniffa_multivariate.svg"))
fig

# %%
adata.obs["donor_clusters_CD14"]


# %%
# donor_keys = [
#     "Sex",
#     "Status",
#     "age_group",
# ]

donor_keys = [
    "Sex",
    "Status",
    "age_group",
]

# %%
selected_cluster = "CD14"
adata.obsm = adata_embs.obsm
adata_ = adata[adata.obs.initial_clustering == selected_cluster].copy()
sc.pp.subsample(adata_, n_obs=50000, random_state=0)
adata_.obs.loc[:, "_indices"] = np.arange(adata_.shape[0])

res = model.perform_multivariate_analysis(
    donor_keys=donor_keys,
    adata=adata_,
    batch_size=128,
    normalize_design_matrix=True,
    offset_design_matrix=False,
    store_lfc=True,
)
gene_properties = (adata_.X != 0).mean(axis=0).A1
gene_properties = pd.DataFrame(
    gene_properties, index=adata_.var_names, columns=["sparsity"]
)

betas_ = res.lfc.transpose("cell_name", "covariate", "gene")
betas_ = betas_.loc[{"covariate": "StatusHealthy"}].values
plt.hist(betas_.mean(0), bins=100)

# %%
# res_bin = model.perform_multivariate_analysis(
#     donor_keys=["Status",],
#     adata=adata_,
#     batch_size=128,
#     normalize_design_matrix=True,
#     offset_design_matrix=False,
#     store_lfc=True,
# )
# betas_bin = res_bin.lfc.transpose("cell_name", "covariate", "gene")
# betas_bin = betas_bin.loc[{"covariate": "StatusHealthy"}].values
# plt.scatter(betas_.mean(0), betas_bin.mean(0))

# %%
lfc_df = pd.DataFrame(
    {
        "LFC": betas_.mean(0),
        "gene": adata_.var_names,
        "gene_index": np.arange(adata_.shape[1]),
        "ensembl_gene": adata_.var["ensembl_gene"],
    }
).assign(absLFC=lambda x: np.abs(x.LFC))

(p9.ggplot(lfc_df, p9.aes(x="absLFC")) + p9.stat_ecdf() + p9.scale_x_log10())

# %%
thresh = np.quantile(lfc_df.absLFC, 0.75)
cond = lfc_df.absLFC > thresh
# cond = np.ones_like(cond, dtype=bool)
# cond = cond & (lfc_df.LFC > 0)
betas_de = betas_[:, cond]
obs_de = lfc_df.loc[cond, :].reset_index(drop=True)

adata_t = sc.AnnData(
    X=betas_de.T,
    obs=obs_de,
)
sc.pp.neighbors(adata_t, n_neighbors=10, metric="cosine", use_rep="X")
sc.tl.umap(adata_t)
sc.tl.leiden(adata_t, resolution=0.2)
fig = sc.pl.umap(
    adata_t,
    color=["LFC", "leiden"],
    vmin=-0.1,
    vmax=0.1,
    cmap="coolwarm",
    return_fig=True,
)
plt.tight_layout()
fig.savefig(
    os.path.join(FIGURE_DIR, f"haniffa.{selected_cluster}.gene_umap.svg"),
)


# %%
import gseapy as gp

gp.get_library_name()
# below mimicks hallmark, CP, and ontology
gene_sets = [
    "MSigDB_Hallmark_2020",
    "WikiPathway_2021_Human",
    # "BioCarta_2016",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    # "Human_Phenotype_Ontology",
]

# %%
gene_info_ = adata_t.obs

beta_module_keys = []
all_enrichr_results = []
for leiden_cluster in np.arange(gene_info_.leiden.nunique()):
    beta_module_name = f"beta_module_leiden_{leiden_cluster}"
    gene_info_module = gene_info_.loc[
        gene_info_.leiden == str(leiden_cluster)
    ].sort_values("absLFC", ascending=False)
    genes = (
        # gene_info_module.loc[lambda x: ~x["ensembl_gene"].isna(), "gene"]
        gene_info_module.loc[:, "gene"]
        .str.strip()
        .str.split(".", expand=True)
        .loc[:, 0]
        .str.upper()
        .tolist()
    )
    gene_indices = gene_info_module.loc[:, "gene_index"].tolist()

    beta_module = betas_[:, gene_indices].mean(1)
    adata_.obs.loc[:, beta_module_name] = beta_module
    beta_module_keys.append(beta_module_name)

    is_done = False
    print(leiden_cluster)
    print()
    for _ in range(10):
        if is_done:
            break

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism="human",
                outdir=None,
                verbose=False,
            )
        except:
            print(f"Error with cluster {leiden_cluster}; retry")
            continue

        enr_results = enr.results.copy().sort_values("Adjusted P-value")
        enr_results.loc[:, "Significance score"] = -np.log10(
            enr_results.loc[:, "Adjusted P-value"]
        )
        # for _, term in enr.results.iloc[:10].iterrows():
        #     print(term.Term)
        # print()
        is_done = True
        all_enrichr_results.append(enr_results.assign(leiden_cluster=leiden_cluster))
all_enrichr_results = pd.concat(all_enrichr_results).astype({"Gene_set": "category"})

fig = sc.pl.embedding(
    adata_,
    basis="X_scviv2_attention_mog_u_mde",
    color=["full_clustering"] + beta_module_keys,
    vmin=-0.1,
    vmax=0.1,
    cmap="coolwarm",
    ncols=1,
    return_fig=True,
)
fig.savefig(os.path.join(FIGURE_DIR, f"haniffa.{selected_cluster}.beta_modules.svg"))

# %%
for idx, leiden_cluster in enumerate(all_enrichr_results.leiden_cluster.unique()):
    plot_df = (
        all_enrichr_results.loc[lambda x: x.leiden_cluster == leiden_cluster, :]
        .loc[lambda x: x["Adjusted P-value"] < 0.05, :]
        # .groupby("Gene_set")
        .sort_values("Adjusted P-value")
        .head(5)
        .sort_values("Gene_set")
    )
    scaler = len(plot_df)
    fig = (
        p9.ggplot(plot_df, p9.aes(x="Term", y="Significance score", fill="Gene_set"))
        + p9.geom_col()
        # + p9.coord_flip()
        + p9.scale_x_discrete(limits=plot_df.Term.tolist())
        + p9.labs(
            x="",
        )
        + p9.theme_classic()
        + p9.theme(
            strip_background=p9.element_blank(),
            subplots_adjust={"wspace": 0.3},
            # panel_background=p9.element_blank(),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            axis_text=p9.element_text(family="sans-serif", size=5),
            axis_title=p9.element_text(family="sans-serif", size=6),
            figure_size=(4 * INCH_TO_CM, 0.5 * scaler * INCH_TO_CM),
        )
    )
    if idx != 0:
        fig = fig + p9.theme(legend_position="none")
    fig.save(
        os.path.join(
            FIGURE_DIR,
            f"haniffa.{selected_cluster}.gsea.beta_modules.{leiden_cluster}.svg",
        )
    )
    fig.draw(show=True)

# %%
betas_ = res.lfc.transpose("cell_name", "covariate", "gene")
betas_ = betas_.loc[{"covariate": "StatusHealthy"}].values
betas_ = betas_.reshape(betas_.shape[0], -1)
beta_pca = PCA(n_components=2)
betas_rep = beta_pca.fit_transform(betas_)

adata_.obsm["betas_rep"] = betas_rep
sc.pl.embedding(
    adata_,
    basis="betas_rep",
)

# %%
plt.hist(beta_pca.components_[0], bins=100)

# %%
lfc_std = (
    res.lfc.loc[{"covariate": "StatusHealthy"}]
    .std("cell_name")
    .to_dataframe()
    .reset_index()
    .rename(columns={"lfc": "lfc_std"})
)
lfc_mean = (
    res.lfc.loc[{"covariate": "StatusHealthy"}]
    .mean("cell_name")
    .to_dataframe()
    .reset_index()
    .rename(columns={"lfc": "lfc_mean"})
)
lfc_mean = (
    lfc_mean.merge(lfc_std, on=["gene", "covariate"])
    .assign(
        cov=lambda x: np.abs(x.lfc_mean) / x.lfc_std,
    )
    .sort_values("cov", ascending=False)
)
lfc_mean


# %%
# umap_rep
gene_name = "IFI27"
_lfc = (
    res.lfc.loc[{"covariate": "StatusHealthy"}]
    .loc[{"gene": gene_name}]
    .to_dataframe()
    .loc[:, ["lfc"]]
)
_lfc = _lfc.loc[adata_.obs_names, :]
adata_.obs.loc[:, "lfc"] = _lfc.values
# gene_tag = f"{gene_name}_log"
# adata_.obs.loc[:, gene_tag] = np.log1p(adata_[:, gene_name].X.toarray().squeeze())
sc.pl.embedding(
    adata_,
    basis="X_scviv2_attention_no_prior_mog_large_u_mde",
    color=["full_clustering", "lfc"],
    ncols=1,
)
# %%
# umap_rep
gene_name = "IFI27"
gene_tag = f"{gene_name}_log"
adata_.obs.loc[:, gene_tag] = np.log1p(adata_[:, gene_name].X.toarray().squeeze())
sc.pl.embedding(
    adata_,
    basis="X_scviv2_attention_no_prior_mog_large_z_mde",
    color=["full_clustering", "Status"] + [gene_tag],
    ncols=1,
)

# %%
(
    p9.ggplot(adata_.obs, p9.aes(x="Status", y=gene_tag))
    + p9.geom_violin()
    + p9.facet_wrap("~full_clustering")
)


# %%
(
    p9.ggplot(lfc_mean, p9.aes(x="lfc_mean", y="lfc_std"))
    + p9.geom_point()
    + p9.geom_text(
        lfc_mean.loc[lambda x: x.lfc_std > 0.1, :],
        p9.aes(label="gene"),
        size=15,
        nudge_y=0.1,
    )
)

# %%
selected_genes = lfc_mean.loc[lambda x: x.lfc_std > 0.1]["gene"].tolist()
selected_genes

selected_lfcs = (
    res.lfc.loc[{"covariate": "StatusHealthy"}]
    .loc[{"gene": selected_genes}]
    .to_dataframe()
    .merge(adata.obs, left_on="cell_name", right_index=True, how="left")
    .reset_index()
    .assign(full_clustering=lambda x: x.full_clustering.astype(str))
)

selected_cts = (
    adata_.obs.full_clustering.value_counts().loc[lambda x: x > 1000].index.tolist()
)
selected_cts

selected_lfcs = selected_lfcs.loc[lambda x: x.full_clustering.isin(selected_cts), :]

(
    p9.ggplot(
        selected_lfcs,
        p9.aes(x="lfc", y=p9.after_stat("density"), fill="full_clustering"),
    )
    + p9.geom_histogram(bins=100, position="identity", alpha=0.5)
    + p9.facet_wrap("~gene", scales="free_y")
    + p9.theme(
        aspect_ratio=1.0,
    )
)

# %%
(
    p9.ggplot(
        selected_lfcs, p9.aes(x="full_clustering", y="lfc", fill="full_clustering")
    )
    + p9.geom_boxplot(
        outlier_alpha=0.0,
    )
    + p9.coord_flip()
    + p9.facet_wrap("~gene", scales="free_y")
    + p9.theme(
        aspect_ratio=1.0,
    )
)


# %%
lfcs = (
    res.lfc.mean("cell_name")
    .to_dataframe()
    .reset_index()
    .assign(
        abs_lfc=lambda x: np.abs(x.lfc),
    )
    .merge(gene_properties, left_on="gene", right_index=True, how="left")
)
lfcs.loc[lfcs.covariate == "StatusHealthy", "lfc"] = -lfcs.loc[
    lfcs.covariate == "StatusHealthy", "lfc"
]

(
    p9.ggplot(lfcs, p9.aes(x="lfc", fill="covariate"))
    + p9.geom_histogram(bins=100)
    + p9.facet_wrap("~covariate")
)


# %%
adata_.obs.full_clustering.value_counts()


# %%
top_genes = (
    # lfcs.query("covariate == 'StatusHealthy'")
    # lfcs.query("covariate == 'SexMale'")
    lfcs
    # .query("covariate == 'age_int'")
    .sort_values("abs_lfc", ascending=False).query("abs_lfc > 0.2")
    # .iloc[:10]
)

# %%
fig = (
    p9.ggplot(top_genes, p9.aes(y="sparsity", x="lfc", color="covariate"))
    + p9.geom_point(size=0.5)
    + p9.geom_vline(xintercept=0.2, linetype="dashed")
    + p9.geom_vline(xintercept=-0.2, linetype="dashed")
    + p9.geom_text(p9.aes(label="gene"), size=5, angle=45, nudge_y=0.05)
    + p9.theme_classic()
    + p9.theme(
        figure_size=(7 * INCH_TO_CM, 5 * INCH_TO_CM),
        legend_position="none",
        # text=p9.element_text(family="arial"),
        axis_title=p9.element_text(size=7),
        axis_text=p9.element_text(size=6),
    )
    + p9.labs(x="LFC", y="# of expressing cells")
)
fig.save(
    os.path.join(FIGURE_DIR, f"haniffa_{selected_cluster}_multivariate_top_genes.svg")
)
fig

# %%

(
    p9.ggplot(top_genes, p9.aes(x="covariate", y="lfc"))
    + p9.geom_point()
    + p9.coord_flip()
    # + p9.geom_vline(xintercept=0.2)
    # + p9.geom_vline(xintercept=-0.2)
    + p9.geom_text(p9.aes(label="gene"), size=7, nudge_x=0.1, angle=45)
)

# %%
gene_names = []
for gene in top_genes.gene.values:
    print(gene)
    gene_name = "gene_" + gene
    adata_.obs.loc[:, gene_name] = (
        adata_.X[:, adata_.var_names == gene].toarray().squeeze()
    )
    gene_names.append(gene_name)

top_genes.lfc.hist(bins=100)


# %%
mg = get_client("gene")
q = mg.querymany(
    top_genes.gene.values,
    scopes="symbol",
    fields="name,summary,genomic_pos.chr",
    species="human",
)
# pd.DataFrame(q).set_index("query").summary

all_res = []
for dico in q:
    chr = dico.get("genomic_pos", {})
    chr = chr.get("chr", "") if isinstance(chr, dict) else chr[0]["chr"]
    all_res.append(
        {
            "gene": dico["query"],
            "chr": chr,
            "summary": dico.get("summary", ""),
        }
    )
all_res = pd.DataFrame(all_res).set_index("gene")
all_res

# %%
sc.pl.embedding(
    adata_,
    color=["Status"] + gene_names,
    # basis="X_scviv2_attention_no_prior_mog_large_u",
    basis="X_scviv2_attention_no_prior_mog_large_z_mde",
    vmax="p95",
    legend_loc="on data",
)

# %%
sc.tl.pca(adata_, min_dist=0.5)

# %%
# lfcs_all.query("covariate == 'StatusHealthy'").query("gene == 'IGHV3-30'").hist()
# %%
adata_log = adata_.copy()
sc.pp.normalize_total(adata_log)
sc.pp.log1p(adata_log)
lfcs_pbulk = adata_log[adata_log.obs.Status == "Covid"].X.mean(0) - adata_log[
    adata_log.obs.Status == "Healthy"
].X.mean(0)
lfcs_pbulk = pd.DataFrame(
    lfcs_pbulk.A1, columns=["lfc_pbulk"], index=adata_log.var_names
)
# %%
lfcs_pbulk
# %%
lfcs_ = (
    lfcs.merge(lfcs_pbulk, left_on="gene", right_index=True, how="left")
    # .query("sparsity > 0.05")
)

# %%
(
    p9.ggplot(
        lfcs_,
        p9.aes(
            x="lfc_pbulk",
            y="lfc",
        ),
    )
    + p9.geom_point()
)

# %%
(
    p9.ggplot(
        lfcs_,
        p9.aes(
            x="sparsity",
            y="lfc",
        ),
    )
    + p9.geom_point()
)

# %%
lfcs
# %%
