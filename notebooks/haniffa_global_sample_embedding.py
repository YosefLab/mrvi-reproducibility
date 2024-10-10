# %%
from mrvi import MrVI
import scanpy as sc
import pandas as pd
import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import scipy

# %%
adata_path = "../results/aws_pipeline/data/haniffa2.preprocessed.h5ad"
model_path = "../results/aws_pipeline/models/haniffa2.mrvi_attention_mog/"

adata = sc.read_h5ad(adata_path)

# %%
model = MrVI.load(model_path, adata=adata)

# %%
# do cluster A, B, global, global emb
# do subset of healthy, early, late patients for easy viz. fix the ordering
cluster_a_cts = [
    "CD14",
    "CD16",
    "DCs",
]
cluster_b_cts = [
    "CD4",
    "NK_16hi",
    "NK_56hi",
    "CD8",
    "gdT",
]
patient_subset = pd.concat(
    [
        model.donor_info[model.donor_info["Worst_Clinical_Status"] == "Healthy"][:6],
        model.donor_info[
            model.donor_info["Worst_Clinical_Status"] == "Critical "
        ].sort_values("Days_from_onset"),
    ]
)["sample_id"]
covid_legend = {"Covid": "#9E1800", "Healthy": "#019E5D"}
patient_covid_colors = patient_subset.map(
    lambda x: (
        "Covid"
        if x
        in model.donor_info[model.donor_info["Status"] == "Covid"]["sample_id"].values
        else "Healthy"
    )
).map(covid_legend)
# %%
# get sample embeddings
global_sample_embed_subset = jax.device_get(
    model.module.params["qz"]["Embed_0"]["embedding"]
)[patient_subset.index]
global_sample_embed_distances = euclidean_distances(global_sample_embed_subset)
# %%
# set keys to group by
adata.obs["cluster_id"] = ""
adata.obs["cluster_id"].loc[adata.obs["initial_clustering"].isin(cluster_a_cts)] = "A"
adata.obs["cluster_id"].loc[adata.obs["initial_clustering"].isin(cluster_b_cts)] = "B"
adata.obs["dummy_var"] = 1
# %%
# get globally averaged sample dist matrix
global_avg_sample_distances = model.get_local_sample_distances(
    adata,
    keep_cell=False,
    groupby="dummy_var",
)
global_avg_sample_distances = global_avg_sample_distances["dummy_var"].values[0]
global_avg_sample_distances = global_avg_sample_distances[patient_subset.index][
    :, patient_subset.index
]
# %%
# get local sample dist matrices from the paper
local_sample_dist_matrices = model.get_local_sample_distances(
    adata, keep_cell=False, groupby="cluster_id"
)
# %%
cluster_a_sample_dist_matrices = (
    local_sample_dist_matrices["cluster_id"].sel(cluster_id_name="A").values
)
cluster_a_sample_dist_matrices = cluster_a_sample_dist_matrices[patient_subset.index][
    :, patient_subset.index
]
cluster_b_sample_dist_matrices = (
    local_sample_dist_matrices["cluster_id"].sel(cluster_id_name="B").values
)
cluster_b_sample_dist_matrices = cluster_b_sample_dist_matrices[patient_subset.index][
    :, patient_subset.index
]
# %%
# plot dist matrices together
# Create a figure with 4 subplots
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# # Flatten the axs array for easier indexing
# axs = axs.flatten()

# List of matrices to plot
matrices = [
    global_sample_embed_distances,
    global_avg_sample_distances,
    cluster_a_sample_dist_matrices,
    cluster_b_sample_dist_matrices,
]

# Titles for each subplot
titles = [
    "Global Sample Embedding Distances",
    "Global Averaged Local Sample Distances",
    "Cluster A Averaged Local Sample Distances",
    "Cluster B Averaged Local Sample Distances",
]

# Plot each matrix as a heatmap
for i, (matrix, title) in enumerate(zip(matrices, titles)):

    # compute KS test statistic between distribution of within status and cross status distances
    within_covid_distances = matrix[6:, 6:][np.where(~np.eye(6, dtype=bool))].flatten()
    cross_status_distances = np.concatenate(
        [matrix[:6, 6:].flatten(), matrix[6:, :6].flatten()]
    )
    ks_statistic, ks_pvalue = scipy.stats.kstest(
        within_covid_distances, cross_status_distances
    )
    print(f"KS statistic: {ks_statistic}, p-value: {ks_pvalue}")
    g = sns.clustermap(
        pd.DataFrame(matrix, columns=patient_subset.index, index=patient_subset.index),
        cmap="viridis",
        square=True,
        row_cluster=False,
        row_colors=patient_covid_colors,
        col_cluster=False,
    )
    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Samples")
    g.ax_heatmap.set_xticklabels(patient_subset.values, rotation=90)
    g.ax_heatmap.set_yticklabels(patient_subset.values, rotation=0)

    # Create a legend for the row colors
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in covid_legend.values()
    ]
    labels = list(covid_legend.keys())

    # Add the legend to the plot
    g.ax_row_dendrogram.legend(
        handles,
        labels,
        title="COVID Status",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    # Adjust the figure size to accommodate the legend
    g.fig.set_size_inches(8, 6)

    # Set the title
    g.fig.suptitle(title, fontsize=16)
    # Add subtitle with KS statistic
    ks_stat_rounded = f"{ks_statistic:.2g}"
    g.fig.text(
        0.5,
        0.92,
        f"KS within-COVID vs. cross-status distances p-val: {ks_pvalue:.2g}",
        ha="center",
        fontsize=12,
    )

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f"./figures/sample_embedding_experiment_{title}.png")
    plt.show()

    plt.close()

# %%
