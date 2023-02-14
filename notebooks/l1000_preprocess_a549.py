# %%
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fuzzywuzzy import process

from statsmodels.stats.multitest import multipletests
from bioinfokit import visuz

# %%
full_a549_df_list = []
for i in range(1, 7):
    df = pd.read_csv(f"../data/l1000_signatures/A549_sig_batch_{i}.xls", sep="\t")
    full_a549_df_list.append(df)
full_a549_df = pd.concat(full_a549_df_list)
# %%
full_a549_sig_meta_df_list = []
for i in range(1, 7):
    df = pd.read_csv(f"../data/l1000_signatures/A549_sig_meta_batch_{i}.xls", sep="\t")
    full_a549_sig_meta_df_list.append(df)
a549_sig_meta_df = pd.concat(full_a549_sig_meta_df_list)

# %%
# Join the meta data with the signatures
full_a549_df_w_meta = full_a549_df.merge(
    a549_sig_meta_df, left_on="signatureID", right_on="SignatureId"
)
# %%
# Read the full a549 sciplex data
a549_adata = sc.read("../data/sciplex_A549_significant_all_phases.h5ad")
# %%
sciplex_product_names = set(a549_adata.obs["product_name"].cat.categories)

# %%
# Filter on
sig_product_names = set(a549_sig_meta_df["Perturbagen"])

# %%
sciplex_product_names.intersection(sig_product_names)

# %%
valid_mapping = {}
for pn in sciplex_product_names:
    match, score = process.extractOne(pn, sig_product_names)
    if score >= 90:
        print(f"{pn} match: {match}, score: {score}")
        valid_mapping[match] = pn

# remove incorrect one
del valid_mapping["Estradiol"]
# %%
# Filter df on valid mapping
filtered_a549_df = full_a549_df_w_meta[
    full_a549_df_w_meta["Perturbagen"].isin(valid_mapping.keys())
]
# %%
# Map name to new column
filtered_a549_df.loc[:, "product_name"] = filtered_a549_df["Perturbagen"].map(
    valid_mapping
)

# %%
# Write filtered sig df to csv
filtered_a549_df.to_csv(f"../data/l1000_signatures/A549_sig_filtered.csv", sep=",")

# %%
# Read back filtered sig df
filtered_a549_df = pd.read_csv(
    f"../data/l1000_signatures/A549_sig_filtered.csv", sep=","
)

# %%
# Filter on one signature for every chemical at random (for simplicity)
# Should use exemplar if available
simple_filtered_a549_df_lst = []
for prod in filtered_a549_df["product_name"].unique():
    prod_df = filtered_a549_df[filtered_a549_df["product_name"] == prod]
    if (prod_df["is_exemplar"] == 1).any():
        sig_id = prod_df[prod_df["is_exemplar"] == 1].iloc[0]["signatureID"]
    else:
        sig_id = prod_df[
            "signatureID"
        ].unique()[0]
    simple_filtered_a549_df_lst.append(
        filtered_a549_df[filtered_a549_df["signatureID"] == sig_id]
    )
simple_filtered_a549_df = pd.concat(simple_filtered_a549_df_lst)
simple_filtered_a549_df.to_csv(
    f"../data/l1000_signatures/A549_sig_filtered_simple.csv", sep=","
)

# %%
# Read back filtered sig df
simple_filtered_a549_df = pd.read_csv(
    f"../data/l1000_signatures/A549_sig_filtered_simple.csv", sep=","
)

# %%
simple_filtered_a549_df["product_name"].value_counts()

# %%
# First correct for multiple testing
for prod in simple_filtered_a549_df["product_name"].unique():
    prod_mask = simple_filtered_a549_df["product_name"] == prod
    simple_filtered_a549_df.loc[prod_mask, "corr_p_val"] = multipletests(
        simple_filtered_a549_df.loc[prod_mask]["Significance_pvalue"],
        alpha=0.05,
        method="fdr_bh",
    )[1]
simple_filtered_a549_df[["Significance_pvalue", "corr_p_val"]]

# %%
# Volcano plot of the filtered a549 signatures
visuz.GeneExpression.volcano(
    simple_filtered_a549_df,
    lfc="Value_LogDiffExp",
    pv="corr_p_val",
    lfc_thr=(-0.5, 0.5),
    show=True,
)

# %%
# Threshold for DEGs
prod_deg_map = {}
for prod in simple_filtered_a549_df["product_name"].unique():
    prod_df = simple_filtered_a549_df[simple_filtered_a549_df["product_name"] == prod]
    prod_deg_map[prod] = set(
        [(gene, "+") for gene in prod_df[
            (prod_df["Value_LogDiffExp"] >= 0.5) & (prod_df["corr_p_val"] <= 0.05)
        ]["Name_GeneSymbol"]
        .unique()
        .tolist()]
    )
    prod_deg_map[prod] = prod_deg_map[prod].union(set(
        [(gene, "-") for gene in prod_df[
            (prod_df["Value_LogDiffExp"] <= -0.5) & (prod_df["corr_p_val"] <= 0.05)
        ]["Name_GeneSymbol"]
        .unique()
        .tolist()]
    ))

# %%
# Construct matrix describing jaccard sim of sets of DEGs between drugs (and in same direction)
deg_sim_mtx = np.zeros((len(prod_deg_map), len(prod_deg_map)))
prod_order = list(prod_deg_map.keys())
for i, prodi in enumerate(prod_order):
    for j, prodj in enumerate(prod_order):
        if i <= j:
            # jaccard similarity
            intersection = len(list(set(prod_deg_map[prodi]).intersection(prod_deg_map[prodj])))
            union = (len(prod_deg_map[prodi]) + len(prod_deg_map[prodj])) - intersection
            deg_sim_mtx[i, j] = float(intersection) / (union + 1)
            deg_sim_mtx[j, i] = deg_sim_mtx[i, j]
deg_sim_mtx

# %%
# Viz matrix
deg_sim_df = pd.DataFrame(deg_sim_mtx, index=prod_order, columns=prod_order)

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(deg_sim_df, cmap="YlGnBu", ax=ax)
plt.show()

# %%
# clustermap
g = sns.clustermap(deg_sim_df, cmap="YlGnBu", xticklabels=True, yticklabels=True, row_cluster=False)
col_cluster_ord = g.dendrogram_col.reordered_ind
g2 = sns.clustermap(deg_sim_df.iloc[col_cluster_ord, col_cluster_ord], cmap="YlGnBu", xticklabels=True, yticklabels=True, col_cluster=False, row_cluster=False)
plt.show()

# %%
# Save matrix
deg_sim_df.to_csv(f"../data/l1000_signatures/A549_deg_sim.csv", sep=",")

# %%
