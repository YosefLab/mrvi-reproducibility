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
from scipy.cluster.hierarchy import fcluster
from plot_utils import INCH_TO_CM, ALGO_RENAMER, SHARED_THEME

sc.set_figure_params(dpi_save=500)
plt.rcParams['axes.grid'] = False
plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = "/data1/scvi-v2-reproducibility/experiments/haniffa"
os.makedirs(FIGURE_DIR, exist_ok=True)

adata = sc.read_h5ad(
    "../results/aws_pipeline/haniffa.preprocessed.h5ad"
)

# %%
adata_files = glob.glob(
    "../results/aws_pipeline/data/haniffa.*.final.h5ad"
)
for adata_file in adata_files:
    adata_ = sc.read_h5ad(adata_file)
    print(adata_.shape)
    for obsm_key in adata_.obsm.keys():
        if obsm_key.endswith("mde") & ("scviv2" in obsm_key):
            print(obsm_key)
            rdm_perm = np.random.permutation(adata.shape[0])
            sc.pl.embedding(
                adata_[rdm_perm],
                basis=obsm_key,
                color=["initial_clustering", "Status", "Site"],
                save=f"haniffa.{obsm_key}.png",
                ncols=1,
            )
# %%
scibv_files = glob.glob(
    "../results/aws_pipeline/metrics/haniffa.*.csv"
)
scib_metrics = pd.DataFrame()
for dmat_file in scibv_files:
    d = pd.read_csv(dmat_file, index_col=0)
    scib_metrics = pd.concat([scib_metrics, d], axis=0)
scib_metrics.loc[:, "method"] = scib_metrics.latent_key.str.split("_").str[1:-1].apply(lambda x: "_".join(x))
scib_metrics.loc[:, "latent"] = scib_metrics.latent_key.str.split("_").str[-1]


# %%
scib_metrics_ = (
    scib_metrics.copy()
    .assign(
        metric_v=lambda x: np.round(x.metric_value, 3).astype(str),
        latent=lambda x: x.latent.str.replace("subleiden1", "u"),
        model=lambda x: x.method.replace(
            {
                "PCA_clusterkey": "composition_PCA_clusterkey_subleiden1",
                "SCVI_clusterkey": "composition_SCVI_clusterkey_subleiden1",
            }
        )
    )
    .loc[lambda x: x.model.isin(ALGO_RENAMER.keys())]
    .assign(
        Model=lambda x: pd.Categorical(x.model.replace(ALGO_RENAMER), categories=ALGO_RENAMER.values()),
    )
)
# %%
means_ = scib_metrics_.groupby("metric_name").mean()
stds_ = scib_metrics_.groupby("metric_name").std()
scib_metrics_ = scib_metrics_.assign(
    metric_v_mean=lambda x: x.metric_name.map(means_.metric_value),
    metric_v_std=lambda x: x.metric_name.map(stds_.metric_value),
    metric_v_col=lambda x: (x.metric_value - x.metric_v_mean) / x.metric_v_std,
)

# %%
plot_df = (
    scib_metrics_.loc[lambda x: x.latent == "u"]
)
# scib_metrics_ = scib_metrics_.loc[lambda x: x.latent == "u", :]
fig = (
    p9.ggplot(plot_df, p9.aes(x="Model", y="metric_name", fill="metric_v_col"))
    + p9.geom_tile()
    + p9.geom_text(p9.aes(label="metric_v"), size=6)
    # + p9.geom_point(stroke=0, size=3)
    # + p9.facet_grid("latent~metric_name", scales="free")
    + p9.coord_flip()
    + p9.labs(
        x="",
        y="",
    )
    + p9.theme_classic()
    + SHARED_THEME
    + p9.theme(
        aspect_ratio=1.0,
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
        axis_text_x=p9.element_text(angle=-45),
        legend_position="none",
    )
)
fig.save(os.path.join(FIGURE_DIR, "haniffa.u.svg"))
fig

# %%
plot_df = (
    scib_metrics_.loc[
        lambda x: (x.latent == "z") | (~x.Model.str.startswith("scviv2"))
    ]
)
fig = (
    p9.ggplot(plot_df, p9.aes(x="Model", y="metric_name", fill="metric_v_col"))
    + p9.geom_tile()
    + p9.geom_text(p9.aes(label="metric_v"), size=6)
    # + p9.geom_point(stroke=0, size=3)
    # + p9.facet_grid("latent~metric_name", scales="free")
    + p9.coord_flip()
    + p9.labs(
        x="",
        y="",
    )
    + p9.theme_classic()
    + SHARED_THEME
    + p9.theme(
        aspect_ratio=1.0,
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
        axis_text_x=p9.element_text(angle=-45),
        legend_position="none",
    )
)
fig.save(os.path.join(FIGURE_DIR, "haniffa.z.svg"))
fig
# %%
scib_metrics_
# %%
donor_info = adata_.obs.drop_duplicates("patient_id").set_index("patient_id")
color_covid = donor_info["Status"].map({"Covid": "#9E1800", "Healthy": "#019E5D"})
color_sex = donor_info["Sex"].map({"Male": "#4791FF", "Female": "#EBA315"})
color_outcome = donor_info["Outcome"].map({"Home": "#466EB8", "Death": "#B80033", "unknown": "#718085"})
color_site = donor_info["Site"].map({"Ncl": "#eb4034", "Cambridge": "#3452eb"})
color_age = donor_info["Age_interval"].map(
    {
    '(20, 29]': "#f7fcf5",
    '(30, 39]': "#dbf1d6",
    '(40, 49]': "#aedea7",
    '(50, 59]': "#73c476",
    '(60, 69]': "#37a055",
    '(70, 79]': "#0b7734",
    '(80, 89]': "#00441b"
    }
)
donor_info["color_age"] = color_age


colors = pd.concat(
    [
        color_site, color_age, color_covid
    ], 1
)
# %%
dmat_files = glob.glob(
    "../results/aws_pipeline/distance_matrices/haniffa.*.nc"
)
dmat_files

# %%
dmat_file = "../results/aws_pipeline/distance_matrices/haniffa.scviv2_attention.distance_matrices.nc"
d = xr.open_dataset(dmat_file)
# %%
# selected_ct = "CD16"
selected_ct = "CD4"

d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
d1 = d1.loc[dict(sample_x=donor_info.index)].loc[dict(sample_y=donor_info.index)]
Z = hierarchical_clustering(d1.values, method="complete", return_ete=False)
clusters = fcluster(Z, t=3, criterion="maxclust")
donor_info.loc[:, "cluster"] = clusters
# cluster_colors = donor_info["cluster"].map({1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5"})
colors.loc[:, "cluster"] = donor_info["cluster"].map({1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5"}).values
sns.clustermap(d1.to_pandas(), row_linkage=Z, col_linkage=Z, row_colors=colors)

# %%
mask_samples = donor_info.loc[lambda x: x.Site == "Ncl"].index
d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
d1 = d1.loc[dict(sample_x=mask_samples)].loc[dict(sample_y=mask_samples)]
Z = hierarchical_clustering(d1.values, method="complete", return_ete=False)
clusters = fcluster(Z, t=3, criterion="maxclust")
colors_ = colors.loc[d1.sample_x.values]
colors_.loc[:, "cluster"] = clusters
colors_.loc[:, "cluster"] = colors_.cluster.map(
    {1: "#eb4034", 2: "#3452eb", 3: "#f7fcf5", 4: "#FF8000"}
).values

sns.clustermap(d1.to_pandas(), row_linkage=Z, col_linkage=Z, row_colors=colors_)



# %%
mask_samples = donor_info.loc[lambda x: x.Site == "Cambridge"].index
d1 = d.loc[dict(initial_clustering_name=selected_ct)]["initial_clustering"]
d1 = d1.loc[dict(sample_x=mask_samples)].loc[dict(sample_y=mask_samples)]
Z = hierarchical_clustering(d1.values, method="complete", return_ete=False)
sns.clustermap(d1.to_pandas(), row_linkage=Z, col_linkage=Z, row_colors=colors)

# %%
adata_log = adata.copy()
sc.pp.normalize_total(adata_log, target_sum=1e4)
sc.pp.log1p(adata_log)
pop = adata_log[(adata_log.obs.initial_clustering == selected_ct)].copy()
pop.obs.loc[:, "sample_group"] = "Group " + pop.obs["patient_id"].map(donor_info["cluster"]).astype(str)
pop.obs.loc[:, "sample_group"] = pop.obs["sample_group"].astype("category")

sc.tl.rank_genes_groups(pop, "sample_group", method="wilcoxon", n_genes=1000)

# %%
sc.pl.rank_genes_groups_heatmap(
    pop,
    n_genes=10,
    save=f"haniffa.{selected_ct}.clustered.svg",
)


# %%
pop_ncl = pop[pop.obs.Site == "Ncl"].copy()
sc.tl.rank_genes_groups(pop_ncl, "sample_group", method="wilcoxon", n_genes=1000)
sc.pl.rank_genes_groups(
    pop_ncl,
    n_genes=25,
)

# %%
pop_cam = pop[pop.obs.Site == "Cambridge"].copy()
sc.tl.rank_genes_groups(pop_cam, "sample_group", method="wilcoxon", n_genes=1000)
sc.pl.rank_genes_groups_dotplot(
    pop_cam,
    n_genes=25,
)

# %%
