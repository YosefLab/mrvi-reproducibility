"""
Script for processing signatures from L1000 bulk data into a reference similarity matrix between chemical compounds.

Takes signatures for chemical perturbations found in the Sciplex dataset and produces a similarity matrix
using the Jaccard similarity between DEGs of the perturbations. This similarity matrix can then be used to
assess how well a model can recapitulate the similarity between perturbations.
"""
# %%
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fuzzywuzzy import process

from statsmodels.stats.multitest import multipletests
from bioinfokit import visuz
import leidenalg as la
import igraph as ig

# %%
# Collected from http://www.ilincs.org/ilincs/signatures/search/MCF7
full_mcf7_df_list = []
for i in range(1, 6):
    df = pd.read_csv(f"../data/l1000_signatures/MCF7_sig_batch_{i}.xls", sep="\t")
    full_mcf7_df_list.append(df)
full_mcf7_df = pd.concat(full_mcf7_df_list)
# %%
mcf7_sig_meta_df = pd.read_csv(f"../data/l1000_signatures/MCF7_sig_meta.xls", sep="\t")

# %%
# Join the meta data with the signatures
full_mcf7_df_w_meta = full_mcf7_df.merge(
    mcf7_sig_meta_df, left_on="signatureID", right_on="SignatureId"
)
# %%
# Read the full mcf7 sciplex data
mcf7_adata = sc.read("../data/sciplex_MCF7_simple_filtered_all_phases.h5ad")
# %%
sciplex_product_names = set(mcf7_adata.obs["product_name"].cat.categories)

# %%
# Filter on
sig_product_names = set(mcf7_sig_meta_df["Perturbagen"])

# %%
sciplex_product_names.intersection(sig_product_names)

# %%
# remove incorrect one
sig_product_names.remove("Estradiol")

valid_mapping = {}
for pn in sciplex_product_names:
    match, score = process.extractOne(pn, sig_product_names)
    if score >= 90:
        print(f"{pn} match: {match}, score: {score}")
        valid_mapping[match] = pn

# %%
# Filter df on valid mapping
filtered_mcf7_df = full_mcf7_df_w_meta[
    full_mcf7_df_w_meta["Perturbagen"].isin(valid_mapping.keys())
]
# %%
# Map name to new column
filtered_mcf7_df.loc[:, "product_name"] = filtered_mcf7_df["Perturbagen"].map(
    valid_mapping
)

# %%
# Write filtered sig df to csv
filtered_mcf7_df.to_csv(f"../data/l1000_signatures/MCF7_sig_filtered.csv", sep=",")

# %%
# Read back filtered sig df
filtered_mcf7_df = pd.read_csv(
    f"../data/l1000_signatures/MCF7_sig_filtered.csv", sep=","
)

# %%
# Filter on one signature for every chemical at random (for simplicity)
# Should use exemplar if available
simple_filtered_mcf7_df_lst = []
for prod in filtered_mcf7_df["product_name"].unique():
    prod_df = filtered_mcf7_df[filtered_mcf7_df["product_name"] == prod]
    if (prod_df["is_exemplar"] == 1).any():
        sig_id = prod_df[prod_df["is_exemplar"] == 1].iloc[0]["signatureID"]
    else:
        sig_id = prod_df["signatureID"].unique()[0]
    simple_filtered_mcf7_df_lst.append(
        filtered_mcf7_df[filtered_mcf7_df["signatureID"] == sig_id]
    )
simple_filtered_mcf7_df = pd.concat(simple_filtered_mcf7_df_lst)
simple_filtered_mcf7_df.to_csv(
    f"../data/l1000_signatures/MCF7_sig_filtered_simple.csv", sep=","
)

# %%
# Read back filtered sig df
simple_filtered_mcf7_df = pd.read_csv(
    f"../data/l1000_signatures/MCF7_sig_filtered_simple.csv", sep=","
)

# %%
simple_filtered_mcf7_df["product_name"].value_counts()

# %%
# First correct for multiple testing
for prod in simple_filtered_mcf7_df["product_name"].unique():
    prod_mask = simple_filtered_mcf7_df["product_name"] == prod
    simple_filtered_mcf7_df.loc[prod_mask, "corr_p_val"] = multipletests(
        simple_filtered_mcf7_df.loc[prod_mask]["Significance_pvalue"],
        alpha=0.05,
        method="fdr_bh",
    )[1]
simple_filtered_mcf7_df[["Significance_pvalue", "corr_p_val"]]

# %%
# Volcano plot of the filtered mcf7 signatures
visuz.GeneExpression.volcano(
    simple_filtered_mcf7_df,
    lfc="Value_LogDiffExp",
    pv="corr_p_val",
    lfc_thr=(-0.5, 0.5),
    show=True,
)

# %%
# Threshold for DEGs
prod_deg_map = {}
for prod in simple_filtered_mcf7_df["product_name"].unique():
    prod_df = simple_filtered_mcf7_df[simple_filtered_mcf7_df["product_name"] == prod]
    prod_deg_map[prod] = set(
        [
            (gene, "+")
            for gene in prod_df[
                (prod_df["Value_LogDiffExp"] >= 0.5) & (prod_df["corr_p_val"] <= 0.05)
            ]["Name_GeneSymbol"]
            .unique()
            .tolist()
        ]
    )
    prod_deg_map[prod] = prod_deg_map[prod].union(
        set(
            [
                (gene, "-")
                for gene in prod_df[
                    (prod_df["Value_LogDiffExp"] <= -0.5)
                    & (prod_df["corr_p_val"] <= 0.05)
                ]["Name_GeneSymbol"]
                .unique()
                .tolist()
            ]
        )
    )

# %%
# Construct matrix describing jaccard sim of sets of DEGs between drugs (and in same direction)
deg_sim_mtx = np.zeros((len(prod_deg_map), len(prod_deg_map)))
prod_order = list(prod_deg_map.keys())
for i, prodi in enumerate(prod_order):
    for j, prodj in enumerate(prod_order):
        if i <= j:
            # jaccard similarity
            intersection = len(
                list(set(prod_deg_map[prodi]).intersection(prod_deg_map[prodj]))
            )
            union = (len(prod_deg_map[prodi]) + len(prod_deg_map[prodj])) - intersection
            deg_sim_mtx[i, j] = float(intersection) / union if union > 0 else 0
            deg_sim_mtx[j, i] = deg_sim_mtx[i, j]
deg_sim_mtx

# %%
# Filter out zero diagonal elements (no degs)
has_deg_idxs = np.where(np.diag(deg_sim_mtx) != 0)[0]
deg_sim_mtx = deg_sim_mtx[has_deg_idxs, :][:, has_deg_idxs]
prod_order = list(np.array(prod_order)[has_deg_idxs])

# %%
# Viz matrix
deg_sim_df = pd.DataFrame(deg_sim_mtx, index=prod_order, columns=prod_order)

sns.clustermap(deg_sim_df, cmap="YlGnBu")
plt.show()

# %%
# Save matrix
deg_sim_df.to_csv(f"../data/l1000_signatures/MCF7_deg_sim.csv", sep=",")

# %%
# Load matrix
deg_sim_df = pd.read_csv(
    f"../data/l1000_signatures/MCF7_deg_sim.csv", sep=",", index_col=0
)

# %%
# clustermap
g = sns.clustermap(
    deg_sim_df, cmap="YlGnBu", xticklabels=True, yticklabels=True, row_cluster=False
)
plt.clf()
col_cluster_ord = g.dendrogram_col.reordered_ind
g2 = sns.clustermap(
    deg_sim_df.iloc[col_cluster_ord, col_cluster_ord],
    cmap="YlGnBu",
    xticklabels=True,
    yticklabels=True,
    col_cluster=False,
    row_cluster=False,
)
plt.show()


# %%
# Silhouette plots to check clustering

from sklearn.metrics import silhouette_samples, silhouette_score


def silhouette_plot(dist_mtx, cluster_labels, title=""):
    silhouette_avg = silhouette_samples(dist_mtx, cluster_labels)

    fig, ax = plt.subplots()
    y_lower = 10

    for i in range(np.max(cluster_labels) + 1):
        ith_cluster_silhouette_values = silhouette_avg[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        if size_cluster_i == 0:
            continue
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / np.max(cluster_labels))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score
    avg_silhouette_score = silhouette_score(dist_mtx, cluster_labels)
    ax.axvline(x=avg_silhouette_score, color="red", linestyle="--")
    ax.set_title(title)

    plt.show()


# %%
for res in np.arange(0.1, 1.05, 0.05):
    clipped_deg_sim = deg_sim_df.values.copy()
    clipped_deg_sim[clipped_deg_sim < 0.1] = 0
    g_adj = ig.Graph.Weighted_Adjacency(clipped_deg_sim, mode="undirected")
    partition = la.RBConfigurationVertexPartition(g_adj, resolution_parameter=res)
    optimizer = la.Optimiser()
    optimizer.optimise_partition(partition)

    # Get the cluster labels
    cluster_labels = np.array(partition.membership)
    filtered_cluster_labels = []
    cluster_label_index = []
    keep_clusters = [
        cluster_label
        for cluster_label, ct in zip(*np.unique(cluster_labels, return_counts=True))
        if ct > 1
    ]
    if len(keep_clusters) == 1:
        continue
    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label in keep_clusters:
            filtered_cluster_labels.append(cluster_label)
            cluster_label_index.append(deg_sim_df.index[i])
    subset_deg_sim_df = deg_sim_df.loc[cluster_label_index, :].loc[
        :, cluster_label_index
    ]
    silhouette_plot(
        1 - subset_deg_sim_df.values, np.array(filtered_cluster_labels), title=f"{res}, kept {len(filtered_cluster_labels)}"
    )

# %%
# Define the parameters for the Leiden algorithm
clipped_deg_sim = deg_sim_df.values.copy()
clipped_deg_sim[clipped_deg_sim < 0.1] = 0

g_adj = ig.Graph.Weighted_Adjacency(clipped_deg_sim, mode="undirected")
partition = la.RBConfigurationVertexPartition(g_adj, resolution_parameter=0.9)
optimizer = la.Optimiser()
optimizer.optimise_partition(partition)

# Get the cluster labels
cluster_labels = np.array(partition.membership)

print(cluster_labels)

# %%
color_list = [
    "red",
    "blue",
    "green",
    "cyan",
    "pink",
    "orange",
    "grey",
    "yellow",
    "white",
    "black",
    "purple",
    "olive",
    "goldenrod",
    "magenta",
    "teal",
]
ig.plot(
    g_adj,
    vertex_color=[color_list[k] for k in cluster_labels],
    vertex_label=deg_sim_df.index,
)

# %%
# Filter out clusters with only one element
filtered_cluster_labels = []
cluster_label_index = []
keep_clusters = [
    cluster_label
    for cluster_label, ct in zip(*np.unique(cluster_labels, return_counts=True))
    if ct > 1
]
for i, cluster_label in enumerate(cluster_labels):
    if cluster_label in keep_clusters:
        filtered_cluster_labels.append(cluster_label)
        cluster_label_index.append(deg_sim_df.index[i])

# %%
# Save the cluster labels
cluster_labels_df = pd.DataFrame(filtered_cluster_labels, index=cluster_label_index)
print(cluster_labels_df)
cluster_labels_df.to_csv(f"../data/l1000_signatures/MCF7_cluster_labels.csv", sep=",")
# %%
