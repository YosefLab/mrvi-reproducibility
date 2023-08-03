# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import seaborn as sns
import xarray as xr
from biothings_client import get_client
from matplotlib.colors import hex2color, rgb2hex
from scib_metrics.benchmark import Benchmarker
from scipy.cluster.hierarchy import fcluster
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from tree_utils import hierarchical_clustering
from utils import perform_gsea

INCH_TO_CM = 1 / 2.54

# %%
gene_sets = [
    "MSigDB_Hallmark_2020",
    "WikiPathway_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
]

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
ensembl_gene = (
    gene_conversion[["query", "ensembl.gene", "_score"]]
    .dropna()
    .sort_values("_score", ascending=False)
    .groupby("query")
    .first()
)

adata.var.loc[:, "ensembl_gene"] = ensembl_gene.reindex(adata.var_names)[
    "ensembl.gene"
].values

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
adata.obs.loc[:, "eps_pca"] = eps_pca

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
pca_eps = PCA(n_components=50)
eps_pca = pca_eps.fit_transform(eps_)
adata_embs.obsm["eps_PCs"] = eps_pca

import pymde

mde_kwargs = dict(
    embedding_dim=2,
    constraint=pymde.Standardized(),
    repulsive_fraction=0.7,
    device="cuda",
    n_neighbors=15,
)
latent_mde = pymde.preserve_neighbors(eps_, **mde_kwargs).embed().cpu().numpy()

adata_embs.obsm["eps_mde"] = latent_mde
sc.pl.embedding(
    adata_embs,
    basis="eps_mde",
    color=["initial_clustering", "Status", "eps_pca", "patient_id"],
)

# %%
# my_adata = model.adata.copy()
# sc.pp.subsample(my_adata, n_obs=50000)
adata.obs.loc[:, "_indices"] = np.arange(adata.shape[0])
dists = model.get_local_sample_distances(
    adata,
    keep_cell=True,
)

# %%
axis = 0
dmats = dists["cell"].values
dmats = np.array([dmat[np.triu_indices(dmat.shape[0], k=1)] for dmat in dmats])
dmats = (dmats - dmats.mean(axis=axis, keepdims=True)) / dmats.std(
    axis=axis, keepdims=True
)
# dmats = np.argsort(dmats, axis=1)
dmats_ = PCA(n_components=50).fit_transform(dmats)
latent_ = pymde.preserve_neighbors(dmats_, **mde_kwargs).embed().cpu().numpy()
adata.obsm["dmat_mde"] = latent_
sc.pp.neighbors(adata, use_rep="dmat_mde", n_neighbors=15)
# %%
sc.tl.leiden(adata, key_added="leiden_dmats", resolution=0.005)
# adata.obs.loc[:, "leiden_dmats"] = KMeans(n_clusters=6).fit_predict(adata.obsm["dmat_mde"])
adata.obs.loc[:, "leiden_dmats"] = adata.obs.loc[:, "leiden_dmats"].astype(str)

sc.pl.embedding(adata, basis="dmat_mde", color=["initial_clustering", "leiden_dmats"])

# %%
props_per_cluster = (
    adata.obs.groupby("leiden_dmats")
    .initial_clustering.value_counts(normalize=True)
    .to_frame("prop")
    .reset_index()
)
props_per_cluster

# %%
(
    p9.ggplot(
        props_per_cluster, p9.aes(x="leiden_dmats", y="prop", fill="initial_clustering")
    )
    + p9.geom_col(position="fill")
)

# %%
(
    p9.ggplot(
        props_per_cluster, p9.aes(x="initial_clustering", y="prop", fill="leiden_dmats")
    )
    + p9.geom_col(position="dodge")
    + p9.coord_flip()
)


# %%
mapper = {
    "0": "CD14",
    "1": "NK",
    "2": "CD4",
    "3": "CD8",
    "4": "CD4/CD8",
    "5": "B cell",
    "6": "CD16",
    # "7": "Platelets",
    "7": "Plasmablasts",
}
adata.obs.loc[:, "leiden_names"] = adata.obs.leiden_dmats.map(mapper)

# %%
dmat_files = glob.glob("../results/aws_pipeline/distance_matrices/haniffa2.*.nc")
dmat_files

# %%
# dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention.distance_matrices.nc"
# dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_no_prior_mog_large.distance_matrices.nc"
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_mog.distance_matrices.nc"
d = xr.open_dataset(dmat_file)


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


donor_keys = [
    "Sex",
    "Status",
    "age_group",
]

# %%
de_res = model.perform_multivariate_analysis(
    donor_keys=donor_keys,
    adata=None,
    batch_size=256,
    normalize_design_matrix=True,
    offset_design_matrix=False,
    filter_donors=True,
    subsample_size=500,
    quantile_threshold=0.05,
)
da_res = model.get_outlier_cell_sample_pairs(flavor="ap", minibatch_size=1000)

# %%
gp1 = model.donor_info.query('Status == "Covid"').patient_id.values
gp2 = model.donor_info.query('Status == "Healthy"').patient_id.values
log_p1 = da_res.log_probs.loc[{"sample": gp1}]
log_p1 = logsumexp(log_p1, axis=1) - np.log(log_p1.shape[1])
log_p2 = da_res.log_probs.loc[{"sample": gp2}]
log_p2 = logsumexp(log_p2, axis=1) - np.log(log_p2.shape[1])

log_ratios = log_p1 - log_p2

# %%
log_p_general = logsumexp(da_res.log_probs, axis=1) - np.log(da_res.log_probs.shape[1])
adata.obs.loc[:, "log_p"] = log_p_general
admissibility_threshold = 0.05
adata.obs.loc[:, "is_admissible"] = log_p_general > np.quantile(
    log_p_general, admissibility_threshold
)
adata.obs.loc[:, "is_admissible_"] = adata.obs.loc[:, "is_admissible"].astype(str)

de_es = de_res["effect_size"].loc[{"covariate": "StatusHealthy"}].values
adata.obs.loc[:, "da_es"] = np.clip(
    log_ratios,
    a_min=np.quantile(log_ratios, 0.01),
    a_max=np.quantile(log_ratios, 0.99),
)
adata.obs.loc[~adata.obs.is_admissible, "da_es"] = 0.0
adata.obs.loc[:, "de_es"] = de_es

# %%
sc.pl.embedding(
    adata,
    basis="X_scviv2_attention_mog_u_mde",
    color=["initial_clustering", "da_es", "log_p"],
    # vmin=-2,
    # vmax=2,
    # cmap="coolwarm",
)

# %%
(
    p9.ggplot(
        adata.obs,
        p9.aes("log_p", "da_es"),
    )
    + p9.geom_point()
    + p9.xlim(-15, 10)
    # + p9.ylim(-2, 2)
)
# %%
(
    p9.ggplot(
        adata.obs.query("is_admissible"),
        p9.aes("da_es", "de_es"),
    )
    + p9.geom_point()
    # + p9.xlim(-15, 10)
    # + p9.ylim(-2, 2)
)

# %%
my_adata = adata[adata.obs.initial_clustering == "Plasmablast"].copy()
my_adata.obs.da_es.hist(bins=100)
sc.pl.embedding(
    my_adata,
    basis="X_scviv2_attention_mog_u_mde",
    color=["initial_clustering", "da_es", "log_p", "is_admissible_"],
    ncols=1,
)

# %%
(
    (adata.obs.Status == "Healthy") & (adata.obs.initial_clustering == "Plasmablast")
).mean()

# %%
((adata.obs.Status == "Healthy") & (adata.obs.initial_clustering == "RBC")).mean()

# %%
fig = sc.pl.embedding(
    adata,
    basis="X_scviv2_attention_mog_u_mde",
    color=["initial_clustering", "da_es", "is_admissible_"],
    vmin=-2,
    vmax=2,
    cmap="coolwarm",
    return_fig=True,
    ncols=1,
)
fig.savefig(os.path.join(FIGURE_DIR, f"haniffa.DA_mde.svg"))

# %%
fig = sc.pl.embedding(
    adata,
    basis="X_scviv2_attention_mog_u_mde",
    color="de_es",
    return_fig=True,
)
fig.savefig(os.path.join(FIGURE_DIR, f"haniffa.DE_mde.svg"))

# %%

# %%
cross_df = adata.obs.assign(
    da_es=log_ratios,
    de_es=de_es,
)

# cross_df_key = "initial_clustering"
# cross_df_key = "leiden_dmats"
cross_df_key = "leiden_names"
cross_df_avg = (
    cross_df.groupby(cross_df_key)[["da_es", "de_es"]]
    .median()
    .reset_index()
    .merge(
        cross_df.groupby(cross_df_key).size().to_frame("n_points").reset_index(),
        on=cross_df_key,
    )
)

# %%
fig = (
    p9.ggplot(cross_df_avg, p9.aes(x="da_es", y="de_es"))
    + p9.geom_point(size=0.5)
    + p9.geom_text(p9.aes(label=cross_df_key), nudge_y=0.1, size=5)
    # + p9.xlim(-1, 2)
    # + p9.xlim(-1, 8)
    + p9.labs(
        x="DA score",
        y="DE score",
    )
    + p9.theme_classic()
    + p9.theme(
        axis_text=p9.element_text(family="sans-serif", size=5),
        axis_title=p9.element_text(family="sans-serif", size=6),
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
)
# fig.save(os.path.join(FIGURE_DIR, f"haniffa.DE_DA_cross.svg"))
fig


# %%
donor_keys = [
    "Sex",
    "Status",
    "age_group",
]
adata.obs.loc[:, "is_covid1"] = (
    adata.obs["donor_clusters_CD14"] == "cluster 2"
).astype(int)
adata.obs.loc[:, "is_covid2"] = (
    adata.obs["donor_clusters_CD14"] == "cluster 3"
).astype(int)
donor_keys_bis = ["is_covid1", "is_covid2"]

obs_df = adata.obs.copy()
obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
model.donor_info = obs_df.set_index("_scvi_sample").sort_index()


# %%
selected_cluster = "Monocytes"
# adata.obsm = adata_embs.obsm
# adata_ = adata[adata.obs.initial_clustering.isin(["CD14", "CD16"])].copy()

adata.obsm = adata_embs.obsm
adata_ = adata[adata.obs.leiden_names.isin(["CD14", "CD16"])].copy()

sc.pp.subsample(adata_, n_obs=50000, random_state=0)
adata_.obs.loc[:, "_indices"] = np.arange(adata_.shape[0])
adata_log_ = adata_.copy()
sc.pp.log1p(adata_log_)

res = model.perform_multivariate_analysis(
    donor_keys=donor_keys_bis,
    adata=adata_,
    batch_size=128,
    normalize_design_matrix=True,
    offset_design_matrix=False,
    store_lfc=True,
    eps_lfc=1e-4,
)
gene_properties = (adata_.X != 0).mean(axis=0).A1
gene_properties = pd.DataFrame(
    gene_properties, index=adata_.var_names, columns=["sparsity"]
)


# %%
betas_ = res.lfc.transpose("cell_name", "covariate", "gene")
betas_ = (
    betas_.loc[{"covariate": "is_covid2"}].values
    - betas_.loc[{"covariate": "is_covid1"}].values
)
betas_ = betas_ / np.log(2)
plt.hist(betas_.mean(0), bins=100)
plt.xlabel("LFC")
plt.show()

lfc_df = pd.DataFrame(
    {
        "LFC": betas_.mean(0),
        "LFC_std": betas_.std(0),
        "gene": adata_.var_names,
        "gene_index": np.arange(adata_.shape[1]),
        "ensembl_gene": adata_.var["ensembl_gene"],
    }
).assign(absLFC=lambda x: np.abs(x.LFC))

thresh = np.quantile(lfc_df.absLFC, 0.95)
lfc_df.absLFC.hist(bins=100)
plt.axvline(thresh, color="red")
plt.xlabel("AbsLFC")
plt.show()
print((lfc_df.absLFC > thresh).sum())

# %%

# %%
VMAX = 1.0
cond = lfc_df.absLFC > thresh
betas_de = betas_[:, cond]
obs_de = lfc_df.loc[cond, :].reset_index(drop=True)
obs_de.LFC.hist(bins=100)

# %%

adata_t = sc.AnnData(
    X=betas_de.T,
    obs=obs_de,
)
adata_t.X = (adata_t.X - adata_t.X.mean(0)) / adata_t.X.std(0)
# adata_t.X = (adata_t.X - adata_t.X.mean(1, keepdims=True)) / adata_t.X.std(1, keepdims=True)
sc.pp.neighbors(adata_t, n_neighbors=50, metric="cosine", use_rep="X")
sc.tl.umap(adata_t, min_dist=0.5)
sc.tl.leiden(adata_t, resolution=0.5)
fig = sc.pl.umap(
    adata_t,
    color=["leiden", "LFC"],
    vmin=-VMAX,
    vmax=VMAX,
    cmap="coolwarm",
    return_fig=True,
)
plt.tight_layout()
fig.savefig(
    os.path.join(FIGURE_DIR, f"haniffa.{selected_cluster}.gene_umap.svg"),
)

fig = sc.pl.umap(
    adata_t,
    color="LFC_std",
    return_fig=True,
)
fig.savefig(
    os.path.join(FIGURE_DIR, f"haniffa.{selected_cluster}.gene_umap_std.svg"),
)

# %%
cov_mat = np.corrcoef(adata_t.X)

X_pca = PCA(n_components=50).fit_transform(adata_t.X)
dissimilarity = pairwise_distances(X_pca)
clusters = KMeans(n_clusters=10).fit_predict(X_pca)
# dissimilarity = 1 - cov_mat
# dissimilarity = pairwise_distances(adata_t.X, metric="cosine")

gene_reps = MDS(n_components=2, dissimilarity="precomputed").fit_transform(
    dissimilarity
)
# gene_reps = TSNE(n_components=2, metric="precomputed", init="random", perplexity=50).fit_transform(dissimilarity)
adata_t.obsm["gene_reps"] = gene_reps
adata_t.obs.loc[:, "KMeans_clusters"] = clusters
adata_t.obs.loc[:, "KMeans_clusters"] = adata_t.obs.loc[:, "KMeans_clusters"].astype(
    str
)
sc.pl.embedding(
    adata_t,
    basis="gene_reps",
    color=["KMeans_clusters", "LFC"],
    vmin=-VMAX,
    vmax=VMAX,
    cmap="coolwarm",
)
fig = sc.pl.embedding(
    adata_t,
    basis="gene_reps",
    color="LFC_std",
)

# %%
clustering_key = "KMeans_clusters"
# %%
gene_info_ = adata_t.obs

beta_module_keys = []
all_enrichr_results = []
for cluster in np.arange(gene_info_[clustering_key].nunique()):
    beta_module_name = f"beta_module_{cluster}"
    gene_info_module = gene_info_.loc[
        gene_info_[clustering_key] == str(cluster)
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

    # beta_module = betas_[:, gene_indices].mean(1)
    beta_module = np.median(betas_[:, gene_indices], 1)
    adata_.obs.loc[:, beta_module_name] = beta_module
    beta_module_keys.append(beta_module_name)

    enr = perform_gsea(genes).assign(cluster=cluster)
    all_enrichr_results.append(enr)
all_enrichr_results = pd.concat(all_enrichr_results).astype({"Gene_set": "category"})


# %%
fig = sc.pl.embedding(
    adata_,
    basis="X_scviv2_attention_mog_u_mde",
    color=["initial_clustering"],
    vmax="p95",
    cmap="coolwarm",
    return_fig=True,
)
fig.savefig(
    os.path.join(FIGURE_DIR, f"haniffa.{selected_cluster}.beta_modules_cts.svg")
)

for beta_module_key in beta_module_keys:
    cluster = int(beta_module_key.split("_")[-1])
    vmin, vmax = np.quantile(adata_.obs[beta_module_key], [0.05, 0.95])
    if adata_.obs[beta_module_key].mean() > 0:
        cmap = "Reds"
        vmin = 0
    else:
        cmap = "Blues_r"
        vmax = 0

    fig = sc.pl.embedding(
        adata_,
        basis="X_scviv2_attention_mog_u_mde",
        color=beta_module_key,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        return_fig=True,
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            FIGURE_DIR,
            f"haniffa.{selected_cluster}.beta_modules_cts.{beta_module_key}.svg",
        )
    )

    genes = gene_info_.query(f"{clustering_key} == '{cluster}'").gene.tolist()
    cond = adata_log_.obs.is_covid1.astype(bool) | adata_log_.obs.is_covid2.astype(bool)
    adata_log_1 = adata_log_[cond, :]
    adata_log_1 = adata_log_1[:, genes].copy()
    adata_log_1.X = adata_log_1.X.toarray()
    adata_log_1.X = (adata_log_1.X - adata_log_1.X.mean(0)) / (
        1e-6 + adata_log_1.X.std(0)
    )
    adata_log_1.X = adata_log_1.X.clip(-5, 5)
    adata_log_1.obs["is_covid2"] = adata_log_1.obs["is_covid2"].astype(str)
    fig = sc.pl.heatmap(
        adata_log_1,
        genes,
        groupby="is_covid2",
        show=False,
        vmin=-2,
        vmax=2,
        cmap="coolwarm",
        show_gene_labels=True,
    )
    plt.tight_layout()
    ax = fig["heatmap_ax"]
    ax.figure.savefig(
        os.path.join(
            FIGURE_DIR,
            f"haniffa.{selected_cluster}.beta_modules_cts.{beta_module_key}.heatmap.svg",
        )
    )
    plt.show()

    plot_df = (
        all_enrichr_results.loc[lambda x: x.cluster == cluster, :]
        .loc[lambda x: x["Adjusted P-value"] < 0.1, :]
        .sort_values("Adjusted P-value")
        .head(5)
        .sort_values("Gene_set")
        .assign(
            Term=lambda x: x.Term.str.split(r" \(GO", expand=True).loc[:, 0],
        )
    )
    scaler = len(plot_df)
    fig = (
        p9.ggplot(plot_df, p9.aes(x="Term", y="Significance score"))
        + p9.geom_col(color="grey")
        + p9.scale_x_discrete(limits=plot_df.Term.tolist())
        + p9.labs(
            x="",
        )
        + p9.theme_classic()
        + p9.scale_y_continuous(expand=(0, 0))
        + p9.theme(
            strip_background=p9.element_blank(),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            axis_text=p9.element_text(family="sans-serif", size=5),
            axis_title=p9.element_text(family="sans-serif", size=6),
            # figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
        )
    )
    if idx != 0:
        fig = fig + p9.theme(legend_position="none")
    fig.save(
        os.path.join(
            FIGURE_DIR,
            f"haniffa.{selected_cluster}.beta_modules_cts.{beta_module_key}.gsea.svg",
        )
    )
    plt.tight_layout()
    fig.draw(show=True)


# %%
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
