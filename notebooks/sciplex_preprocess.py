# %%
import warnings
from collections import defaultdict

import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt

# %%
adata = sc.read(
    "/home/justin/ghrepos/mrvi-reproducibility/data/archive/sciplex_raw.h5ad"
)
adata
# %%
plate_meta = pd.read_csv(
    "/home/justin/ghrepos/mrvi-reproducibility/data/archive/aax6234-srivatsan-table-s3.txt",
    header=1,
    sep="\t",
)
plate_meta

# %%
# Fix two drug names in plate meta unaligned w adata
plate_meta.loc[plate_meta["name"] == "Glesatinib(MGCD265)", "name"] = (
    "Glesatinib?(MGCD265)"
)
plate_meta.loc[plate_meta["name"] == "SNS-314 Mesylate", "name"] = "SNS-314"
plate_meta.loc[plate_meta["name"].isna(), "name"] = "Vehicle"
set(plate_meta["name"].unique()).symmetric_difference(
    set(adata.obs["product_name"].unique())
)

# %%
# Assign plate oligo to each cell observation
# This will be used as batch
plate_oligo = np.empty(adata.X.shape[0], dtype=object)
for _, row in plate_meta.iterrows():
    plate = row["plate_oligo"]
    cell_type, dose, product, replicate = (
        row["cell_type"],
        row["dose"],
        row["name"],
        row["replicate"],
    )
    match_idxs = (
        (adata.obs["cell_type"] == cell_type)
        & (adata.obs["dose"] == dose)
        & (adata.obs["product_name"] == product)
        & (adata.obs["replicate"] == replicate)
    )
    plate_oligo[match_idxs] = plate
adata.obs["plate_oligo"] = plate_oligo


# %%
# label cells by phase
# From https://github.com/scverse/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
cell_cycle_genes = [
    x.strip() for x in open("../data/archive/regev_lab_cell_cycle_genes.txt")
]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

# %%
cell_lines = list(adata.obs["cell_type"].cat.categories)
cell_lines
# %%
use_simple_deg_filter = True
if use_simple_deg_filter:
    warnings.filterwarnings("ignore")
    adata.layers["log1p"] = sc.pp.log1p(adata, copy=True).X
    adata.uns["log1p"] = {"base": None}
    n_deg_cutoff = 3000
    per_cl_deg_products = defaultdict(list)
    per_cl_deg_product_doses = defaultdict(list)
    for cl in cell_lines:
        cl_adata = adata[adata.obs["cell_type"] == cl]
        sc.tl.rank_genes_groups(
            cl_adata,
            "product_dose",
            layer="log1p",
            reference="Vehicle_0",
            method="t-test",
            corr_method="benjamini-hochberg",
        )

        flat_n_deg_dict = {}
        n_deg_dict = defaultdict(dict)
        for prod_dose in cl_adata.obs["product_dose"].cat.categories:
            if prod_dose == "Vehicle_0":
                continue
            sig_idxs = cl_adata.uns["rank_genes_groups"]["pvals_adj"][prod_dose] <= 0.05
            suff_lfc_idxs = (
                np.abs(cl_adata.uns["rank_genes_groups"]["logfoldchanges"][prod_dose])
                >= 0.5
            )
            product_name, dose = prod_dose.split("_")
            n_deg_dict[product_name][dose] = np.sum(sig_idxs & suff_lfc_idxs)
            flat_n_deg_dict[prod_dose] = np.sum(sig_idxs & suff_lfc_idxs)

        # save flat_deg_dict to csv
        flat_deg_df = pd.DataFrame.from_dict(flat_n_deg_dict, orient="index")
        flat_deg_df.to_csv(
            "output/{}_flat_deg_dict.csv".format(cl), index_label="product_dose"
        )

        n_deg_list = []
        for prod in n_deg_dict:
            for dose in n_deg_dict[prod]:
                n_deg_list.append(n_deg_dict[prod][dose])

        plt.hist(n_deg_list, bins=100)
        plt.xlim(0, 10000)
        plt.axvline(x=n_deg_cutoff, color="r", linestyle="--")
        plt.show()

        # Keep products with at least one dose past the cutoff
        for prod in n_deg_dict:
            for dose in n_deg_dict[prod]:
                if n_deg_dict[prod][dose] >= n_deg_cutoff:
                    per_cl_deg_products[cl].append(prod)
                    per_cl_deg_product_doses[cl].append(f"{prod}_{dose}")

    for cl in per_cl_deg_products:
        print(cl, len(set(per_cl_deg_products[cl])))
    # Len of union of cl deg products
    union_deg_products = set.union(
        *[set(per_cl_deg_products[cl]) for cl in per_cl_deg_products]
    )
    print(f"Union of all: {len(union_deg_products)}")

    filtered_adata = adata[adata.obs["product_name"].isin(union_deg_products)].copy()
    for cl in per_cl_deg_products:
        filtered_adata.obs[f"{cl}_deg_product"] = filtered_adata.obs[
            "product_name"
        ].isin(per_cl_deg_products[cl])
        filtered_adata.obs[f"{cl}_deg_product"] = (
            filtered_adata.obs[f"{cl}_deg_product"].astype(str).astype("category")
        )

        filtered_adata.obs[f"{cl}_deg_product_dose"] = filtered_adata.obs[
            "product_dose"
        ].isin(per_cl_deg_product_doses[cl])
        filtered_adata.obs[f"{cl}_deg_product_dose"] = (
            filtered_adata.obs[f"{cl}_deg_product_dose"].astype(str).astype("category")
        )
else:
    # Requires running sciplex_get_significant_product_dose.R first
    all_sig_prods = set()
    per_cell_line_prods = {}
    for cl in cell_lines:
        sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
        sig_prod_doses.columns = ["product_dose"]
        sig_prod_doses["product"] = sig_prod_doses["product_dose"].apply(
            lambda x: x.split("_")[0]
        )
        all_sig_prods = all_sig_prods.union(sig_prod_doses["product"].unique())
        per_cell_line_prods[cl] = set(sig_prod_doses["product"].unique())
        print(f"{cl}: {len(per_cell_line_prods[cl])}")
    print(len(all_sig_prods))

    # filter to all significant products across all cell lines
    filtered_adata = adata[adata.obs["product_name"].isin(all_sig_prods)].copy()
    for cl in cell_lines:
        # Indicate which are significant products for this cell line
        filtered_adata.obs[f"{cl}_sig_prod"] = filtered_adata.obs["product_name"].isin(
            per_cell_line_prods[cl]
        )

        # indicate which doses are significant for this cell line
        sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
        sig_prod_doses.columns = ["product_dose"]
        filtered_adata.obs[f"{cl}_sig_prod_dose"] = filtered_adata.obs[
            "product_dose"
        ].isin(sig_prod_doses["product_dose"].values)


# %%
# Add back control
filtered_adata = filtered_adata.concatenate(
    adata[adata.obs["product_name"] == "Vehicle"]
)

# %%
# Add dummy variable for unwanted covariates
filtered_adata.obs["_dummy"] = 1
filtered_adata.obs["_dummy"] = filtered_adata.obs["_dummy"].astype("category")


# %%
# Top HVGs
hvgs = sc.pp.highly_variable_genes(
    filtered_adata,
    flavor="seurat_v3",
    batch_key="cell_type",
    n_top_genes=5000,
    subset=True,
)

# %%
# For keeping all phases
sub_adatas = []
for cl in cell_lines:
    sub_adata = filtered_adata[filtered_adata.obs["cell_type"] == cl].copy()

    # label phases (Too much mem to do this before filtering doses)
    sub_adata.layers["counts"] = sub_adata.X.copy()
    sc.pp.filter_cells(sub_adata, min_genes=50)
    sc.pp.normalize_per_cell(sub_adata, counts_per_cell_after=1e5)

    sc.pp.log1p(sub_adata)
    sc.pp.scale(sub_adata)
    sc.tl.score_genes_cell_cycle(sub_adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # Subsample cells
    # sc.pp.subsample(sub_adata, n_obs=min(25000, sub_adata.shape[0]))

    # Revert counts to X
    sub_adata.X = sub_adata.layers["counts"]

    # Unwrap obs names from index
    sub_adata.obs_names = sub_adata.obs_names.values

    print(sub_adata)
    if use_simple_deg_filter:
        sub_adata.write(f"../data/sciplex_{cl}_simple_filtered_all_phases.h5ad")
    else:
        sub_adata.write(f"../data/sciplex_{cl}_significant_all_phases.h5ad")
    sub_adatas.append(sub_adata)

# %%
# Save full adata too
if use_simple_deg_filter:
    full_adata = anndata.concat(sub_adatas)
    full_adata.write(f"../data/sciplex_simple_filtered.h5ad")

# %%
