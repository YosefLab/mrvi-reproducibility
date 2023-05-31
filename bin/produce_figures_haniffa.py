# %%
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
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


# %%
metad = pd.read_excel(
    "/data1/datasets/41591_2021_1329_MOESM3_ESM.xlsx",
    sheet_name=1,
    header=1,
    index_col=0,
)
metad.loc[:, "DFO"] = (
    metad["Days from onset"]
    .replace(
        {
            "Healthy": None,
            "Non_covid": None,
            "Not_known": None,
            "LPS": None,
        }
    )
)
metad.loc[:, "DFO"] = metad["DFO"].astype(float)

# %%
sc.set_figure_params(dpi_save=500)
plt.rcParams["axes.grid"] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "/data1/scvi-v2-reproducibility/experiments/haniffa2"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad("../results/aws_pipeline/haniffa2.preprocessed.h5ad")
adata_files = glob.glob("../results/aws_pipeline/data/haniffa2.*.final.h5ad")
# %%
from scvi_v2 import MrVI

model = MrVI.load(
    "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa2.scviv2_attention_no_prior_mog_large",
    adata=adata,
)
# model = MrVI.load(
#     "/data1/scvi-v2-reproducibility/results/aws_pipeline/models/haniffa.scviv2_attention_no_prior_mog", adata=adata
# )

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
donor_info.loc[:, ["pc_1", "pc_2"]] = donor_embeds_pca

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
adata_file = (
    "../results/aws_pipeline/data/haniffa2.scviv2_attention_no_prior_mog_large.final.h5ad"
)
adata_ = sc.read_h5ad(adata_file)
print(adata_.shape)
for obsm_key in adata_.obsm.keys():
    if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
        print(obsm_key)
        # rdm_perm = np.random.permutation(adata.shape[0])
        sc.pl.embedding(
            # adata_[rdm_perm],
            adata_,
            basis=obsm_key,
            color=["initial_clustering", "Status", "Site", "patient_id"],
            save=f"haniffa.{obsm_key}.svg",
            ncols=1,
        )
        plt.clf()


# %%
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
    "X_scviv2_attention_no_prior_mog_large_u",
    "X_scviv2_attention_mog_u",
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
color_worst_status = donor_info_["Worst_Clinical_Status"].map(worst_clinical_status_legend)
donor_info_["color_worst_status"] = color_age
donor_info_["color_age"] = color_age

donor_info_["_DFO"] = (donor_info_["DFO"]-donor_info_["DFO"].min())/(donor_info_["DFO"].max()-donor_info_["DFO"].min())
dfo_colors = plt.get_cmap("viridis")(donor_info_["_DFO"])
dfo_colors = [rgb2hex(rgb) for rgb in dfo_colors]
donor_info_["DFO_color"] = dfo_colors
dfo_colors = donor_info_["DFO_color"]

colors = pd.concat(
    [color_age, dfo_colors, color_covid, color_worst_status],
    axis=1
)

# %%
from matplotlib.patches import Patch

for legend_name, my_legend in all_legends.items():
    handles = [Patch(facecolor=hex2color(my_legend[name])) for name in my_legend]
    plt.legend(
        handles,
        my_legend.keys(),
        title='Species',
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper right'
    )
    plt.savefig(os.path.join(FIGURE_DIR, f"legend_{legend_name}.svg"), bbox_inches='tight')


# %%
adata_file = (
    "../results/aws_pipeline/data/haniffa2.scviv2_attention_no_prior_mog_large.final.h5ad"
)
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
        sc.pl.embedding(
            # adata_embs[rdm_perm],
            adata_embs,
            basis=obsm_key,
            color=["initial_clustering", "Status", "eps_pca", "patient_id"],
            # color=["initial_clustering", "Status", "patient_id"],
            save=f"haniffa.{obsm_key}.svg",
            ncols=1,
        )

# %%
dmat_files = glob.glob("../results/aws_pipeline/distance_matrices/haniffa2.*.nc")
dmat_files

# %%
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention.distance_matrices.nc"
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_no_prior_mog_large.distance_matrices.nc"
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa2.scviv2_attention_no_prior_mog_large.normalized_distance_matrices.nc"
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
VMAX = 1.2
selected_cts = [
    "CD14",
    "B_cell",
    "CD4",
]
n_clusters = [
    4,
    3,
    2,
]

for idx, (selected_ct, n_cluster) in enumerate(zip(selected_cts, n_clusters)):
    mask_samples = donor_info_.index
    d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
    d1 = d1.loc[dict(sample_x=mask_samples)].loc[dict(sample_y=mask_samples)]
    Z = hierarchical_clustering(d1.values, method="ward", return_ete=False)

    colors_ = colors.loc[d1.sample_x.values]
    donor_info_ = donor_info_.loc[d1.sample_x.values]
    if selected_ct == "CD14":
        clusters = []
        for patient in donor_info_.index:
            if patient in pop1:
                clusters.append(1)
            elif patient in pop2:
                clusters.append(2)
            else:
                clusters.append(3)
        clusters = np.array(clusters)
    else:
        clusters = fcluster(Z, t=n_cluster, criterion="maxclust")
    donor_info_.loc[:, "cluster_id"] = clusters
    colors_.loc[:, "cluster"] = clusters
    colors_.loc[:, "cluster"] = colors_.cluster.map(
        {1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5", 4: "#FF8000"}
        # red, blue, white
    ).values

    sns.clustermap(
        d1.to_pandas(),
        row_linkage=Z,
        col_linkage=Z,
        row_colors=colors_,
        vmin=VMIN,
        vmax=VMAX,
        cmap="YlGnBu",
        yticklabels=True,
        figsize=(20, 20),
    )
    plt.savefig(os.path.join(FIGURE_DIR, f"clustermap_{selected_ct}.svg"))

    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log)
    sc.pp.log1p(adata_log)
    donor_info_.loc[:, "donor_group"] = donor_info_.cluster_id
    adata_log.obs.loc[:, "donor_status"] = adata_log.obs.patient_id.map(
        donor_info_.loc[:, "donor_group"]
    ).values
    adata_log.obs.loc[:, "donor_status"] = "cluster " + adata_log.obs.loc[:, "donor_status"].astype(str)
    pop = adata_log[(adata_log.obs.initial_clustering == selected_ct)].copy()

    method = "t-test"
    sc.tl.rank_genes_groups(
        pop,
        "donor_status",
        method=method,
        n_genes=1000,
        # rankby_abs=False,
    )
    sc.pl.rank_genes_groups_dotplot(
        pop,
        n_genes=5,
        min_logfoldchange=0.5,
        swap_axes=True,
        save=f"DOThaniffa.{selected_ct}.clustered.svg",
    )


# %%

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
plot_df = (
    adata.obs
    .reset_index()
    .melt(
        id_vars=["index", "initial_clustering"],
        value_vars=es_keys,
    )
)

n_points = adata.obs.initial_clustering.value_counts().to_frame("n_points")
plot_df = (
    plot_df
    .merge(n_points, left_on="initial_clustering", right_index=True, how="left")
    .assign(
        variable_name=lambda x: x.variable.map(
            {
                "es_SexMale": "Sex",
                "es_StatusHealthy": "Status",
                "es_age_int": "Age",
            }
        )
    )
)



# %%
INCH_TO_CM = 1 / 2.54

plt.rcParams["svg.fonttype"] = "none"

fig = (
    p9.ggplot(
        plot_df.loc[lambda x: x.n_points > 7000, :],
        p9.aes(x="initial_clustering", y="value")
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
adata.obs.groupby("patient_id").Collection_Day.value_counts().unstack()

# %%

# %%
sc.pl.umap(
    adata,
    color=["initial_clustering"] + es_keys,
    vmax="p95",
    ncols=1,
    save=f"haniffa_multivariate.png",
    legend_loc="on data",
)


# %%
selected_cluster = "B_cell"
adata.obsm = adata_embs.obsm
adata_ = adata[adata.obs.initial_clustering == selected_cluster].copy()
res = model.perform_multivariate_analysis(
    donor_keys=donor_keys,
    adata=adata_,
    batch_size=128,
    normalize_design_matrix=True,
    offset_design_matrix=False,
    store_lfc=True,
    eps_lfc=1e-4,
)

gene_properties = (adata_.X != 0).mean(axis=0).A1
gene_properties = pd.DataFrame(gene_properties, index=adata_.var_names, columns=["sparsity"])

# %%
lfcs = (
    res
    .lfc
    .mean("cell_name")
    .to_dataframe()
    .reset_index()
    .assign(
        abs_lfc=lambda x: np.abs(x.lfc),
    )
    .merge(gene_properties, left_on="gene", right_index=True, how="left")
)
lfcs.loc[lfcs.covariate == "StatusHealthy", "lfc"] = -lfcs.loc[lfcs.covariate == "StatusHealthy", "lfc"]

(
    p9.ggplot(lfcs, p9.aes(x="lfc", fill="covariate"))
    + p9.geom_histogram(bins=100)
    + p9.facet_wrap("~covariate")
)

# %%
top_genes = (
    # lfcs.query("covariate == 'StatusHealthy'")
    # lfcs.query("covariate == 'SexMale'")
    lfcs
    # .query("covariate == 'age_int'")
    .sort_values("abs_lfc", ascending=False)
    .query("abs_lfc > 0.2")
    # .iloc[:10]
)



# %%
fig = (
    p9.ggplot(top_genes, p9.aes(y="sparsity", x="lfc", color="covariate"))
    + p9.geom_point(
        size=0.5
    )
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
    + p9.labs(
        x="LFC",
        y="# of expressing cells"
    )
)
fig.save(os.path.join(FIGURE_DIR, f"haniffa_{selected_cluster}_multivariate_top_genes.svg"))
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
    adata_.obs.loc[:, gene_name] = adata_.X[:, adata_.var_names == gene].toarray().squeeze()
    gene_names.append(gene_name)

top_genes.lfc.hist(bins=100)


# %%
mg = get_client('gene')
q = mg.querymany(top_genes.gene.values, scopes='symbol', fields='name,summary,genomic_pos.chr', species='human')
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
lfcs_pbulk = adata_log[adata_log.obs.Status == "Covid"].X.mean(0) - adata_log[adata_log.obs.Status == "Healthy"].X.mean(0)
lfcs_pbulk = pd.DataFrame(lfcs_pbulk.A1, columns=["lfc_pbulk"], index=adata_log.var_names)
# %%
lfcs_pbulk
# %%
lfcs_ = (
    lfcs.merge(lfcs_pbulk, left_on="gene", right_index=True, how="left")
    # .query("sparsity > 0.05")
)

# %%
(
    p9.ggplot(lfcs_, p9.aes(x="lfc_pbulk", y="lfc",))
    + p9.geom_point()
)

# %%
(
    p9.ggplot(lfcs_, p9.aes(x="sparsity", y="lfc",))
    + p9.geom_point()
)

# %%
lfcs
# %%
