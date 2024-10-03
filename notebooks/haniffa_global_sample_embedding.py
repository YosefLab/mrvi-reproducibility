# %%
from mrvi import MrVI
import scanpy as sc
import pandas as pd
import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

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
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axs array for easier indexing
axs = axs.flatten()

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
    sns.heatmap(matrix, ax=axs[i], cmap="viridis", square=True)
    axs[i].set_title(title)
    axs[i].set_xlabel("Samples")
    axs[i].set_ylabel("Samples")

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the figure
plt.savefig("./figures/sample_emedding_experiment.png")
plt.close()

# %%
