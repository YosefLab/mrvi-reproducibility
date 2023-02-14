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
mcf7_adata = sc.read("../data/sciplex_MCF7_significant_all_phases.h5ad")
# %%
sciplex_product_names = set(mcf7_adata.obs["product_name"].cat.categories)

# %%
# Filter on
sig_product_names = set(mcf7_sig_meta_df["Perturbagen"])

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
simple_filtered_mcf7_df = []
for prod in filtered_mcf7_df["product_name"].unique():
    first_sig_id = filtered_mcf7_df[filtered_mcf7_df["product_name"] == prod][
        "signatureID"
    ].unique()[0]
    simple_filtered_mcf7_df.append(
        filtered_mcf7_df[filtered_mcf7_df["signatureID"] == first_sig_id]
    )
simple_filtered_mcf7_df = pd.concat(simple_filtered_mcf7_df)
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
    simple_filtered_mcf7_df.loc[prod_mask]["corr_p_val"] = multipletests(
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
            deg_sim_mtx[i, j] = float(intersection) / union
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
deg_sim_df.to_csv(f"../data/l1000_signatures/MCF7_deg_sim.csv", sep=",")

# %%
