# %%
import pandas as pd
import scanpy as sc
import numpy as np

# %%
adata = sc.read(
    "/home/justin/ghrepos/scvi-v2-reproducibility/data/archive/sciplex_raw.h5ad"
)
adata
# %%
plate_meta = pd.read_csv(
    "/home/justin/ghrepos/scvi-v2-reproducibility/data/archive/aax6234-srivatsan-table-s3.txt",
    header=1,
    sep="\t",
)
plate_meta

# %%
# Fix two drug names in plate meta unaligned w adata
plate_meta.loc[
    plate_meta["name"] == "Glesatinib(MGCD265)", "name"
] = "Glesatinib?(MGCD265)"
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
use_sciplex_filter = True
if use_sciplex_filter:
    # filter with vehicle similar products from sciplex_filter.py
    vehicle_nonsim_prods_path = "output/vehicle_nonsim_prods.txt"
    with open(vehicle_nonsim_prods_path, "r") as f:
        vehicle_nonsim_prods = f.read().splitlines()
    filtered_adata = adata[adata.obs["product_name"].isin(vehicle_nonsim_prods)].copy()
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


# %%
# Add back control
filtered_adata = filtered_adata.concatenate(
    adata[adata.obs["product_name"] == "Vehicle"]
)

# %%
# Top 10k HVGs
hvgs = sc.pp.highly_variable_genes(
    filtered_adata, flavor="seurat_v3", n_top_genes=10000, subset=True
)

# %%
# For keeping all phases
for cl in cell_lines:
    sub_adata = filtered_adata[filtered_adata.obs["cell_type"] == cl].copy()

    if not use_sciplex_filter:
        # Indicate which are significant products for this cell line
        sub_adata.obs["sig_prod_cell_line"] = sub_adata.obs["product_name"].isin(
            per_cell_line_prods[cl]
        )

        # indicate which doses are significant for this cell line
        sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
        sig_prod_doses.columns = ["product_dose"]
        sub_adata.obs["sig_prod_dose_cell_line"] = sub_adata.obs["product_dose"].isin(
            sig_prod_doses["product_dose"].values
        )

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
    if use_sciplex_filter:
        sub_adata.write(f"../data/sciplex_{cl}_significant_filtered_all_phases.h5ad")
    else:
        sub_adata.write(f"../data/sciplex_{cl}_significant_all_phases.h5ad")
    del sub_adata

# %%
# Subsample the filtered datasets to 100 cells (filter out lower)
subsample_size = 100
for cl in cell_lines:
    sub_adata = sc.read(f"../data/sciplex_{cl}_significant_filtered_all_phases.h5ad")
    num_cells_per_sample = sub_adata.obs["product_dose"].value_counts()
    keep_idxs = np.zeros(sub_adata.X.shape[0], dtype=bool)
    for sample in num_cells_per_sample.index:
        if num_cells_per_sample[sample] >= subsample_size:
            sample_idxs = sub_adata.obs["product_dose"] == sample
            idx = np.flatnonzero(sample_idxs)
            r = np.random.choice(idx, 100, replace=False)
            keep_idxs[r] = True
    sub_adata = sub_adata[keep_idxs]
    sub_adata.write(f"../data/sciplex_{cl}_significant_subsampled_all_phases.h5ad")
    del sub_adata

# %%
