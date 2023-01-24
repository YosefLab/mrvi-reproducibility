# %%
import pandas as pd
import scanpy as sc
import numpy as np

# %%
adata = sc.read("/home/justin/ghrepos/scvi-v2-reproducibility/data/archive/sciplex_raw.h5ad")
adata
# %%
plate_meta = pd.read_csv("/home/justin/ghrepos/scvi-v2-reproducibility/data/archive/aax6234-srivatsan-table-s3.txt", header=1, sep="\t")
plate_meta

# %%
# Fix two drug names in plate meta unaligned w adata
plate_meta.loc[plate_meta["name"] == "Glesatinib(MGCD265)", "name"] = "Glesatinib?(MGCD265)"
plate_meta.loc[plate_meta["name"] == "SNS-314 Mesylate", "name"] = "SNS-314"
plate_meta.loc[plate_meta["name"].isna(), "name"] = "Vehicle"
set(plate_meta["name"].unique()).symmetric_difference(set(adata.obs["product_name"].unique()))

# %%
# Assign plate oligo to each cell observation
# This will be used as batch
plate_oligo = np.empty(adata.X.shape[0], dtype=object)
for _, row in plate_meta.iterrows():
    plate = row["plate_oligo"]
    cell_type, dose, product, replicate = row["cell_type"], row["dose"], row["name"], row["replicate"]
    match_idxs = ((adata.obs["cell_type"] == cell_type) &
            (adata.obs["dose"] == dose) &
            (adata.obs["product_name"] == product) &
            (adata.obs["replicate"] == replicate))
    plate_oligo[match_idxs] = plate
print(plate_oligo)
adata.obs["plate_oligo"] = plate_oligo
print(adata.obs["plate_oligo"].value_counts())


# %%
# label cells by phase
# From https://github.com/scverse/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
cell_cycle_genes = [x.strip() for x in open('../data/archive/regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

# %%
cell_lines = list(adata.obs["cell_type"].cat.categories)
cell_lines
# %%
all_sig_prods = set()
per_cell_line_prods = {}
for cl in cell_lines:
    sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
    sig_prod_doses.columns = ["product_dose"]
    sig_prod_doses["product"] = sig_prod_doses["product_dose"].apply(lambda x: x.split("_")[0])
    all_sig_prods = all_sig_prods.union(sig_prod_doses["product"].unique())
    per_cell_line_prods[cl] = set(sig_prod_doses["product"].unique())
    print(f"{cl}: {len(per_cell_line_prods[cl])}")
print(len(all_sig_prods))

# %%
# for cl in cell_lines:
#     sub_adata = adata[adata.obs["cell_type"] == cl]

#     # Filter down to significant product doses
#     sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
#     sig_prod_doses.columns = ["product_dose"]

#     sub_adata = sub_adata[
#         sub_adata.obs["product_dose"].isin(sig_prod_doses["product_dose"].values)
#     ]
#     # Add back control
#     sub_adata = sub_adata.concatenate(adata[(adata.obs["product_dose"] == "Vehicle_0") & (adata.obs["cell_type"] == cl)])

#     # Filter down to S phase (Too much mem to do this before filtering doses)
#     sub_adata.layers["counts"] = sub_adata.X.copy()
#     sc.pp.filter_cells(sub_adata, min_genes=200)
#     sc.pp.filter_genes(sub_adata, min_cells=3)
#     sc.pp.normalize_per_cell(sub_adata, counts_per_cell_after=1e5)

#     sc.pp.log1p(sub_adata)
#     sc.pp.scale(sub_adata)
#     sc.tl.score_genes_cell_cycle(sub_adata, s_genes=s_genes, g2m_genes=g2m_genes)
#     sub_adata = sub_adata[sub_adata.obs['phase'] == 'S']

#     # Filter on product doses with more than 30 observations
#     prod_dose_cell_thresh = 30
#     prod_dose_w_enough = sub_adata.obs.product_dose.value_counts()[sub_adata.obs.product_dose.value_counts() > 30].index.tolist()
#     sub_adata = sub_adata[sub_adata.obs["product_dose"].isin(prod_dose_w_enough)]

#     # Create dummy batch column
#     sub_adata.obs["dummy_batch"] = "0"

#     # Revert counts to X
#     sub_adata.X = sub_adata.layers["counts"]

#     # Unwrap obs names from index
#     sub_adata.obs_names = sub_adata.obs_names.values

#     print(sub_adata)
#     sub_adata.write(f"../data/sciplex_{cl}_significant.h5ad")
#     del sub_adata

# %%
# For keeping all phases
for cl in cell_lines:
    sub_adata = adata[adata.obs["cell_type"] == cl].copy()

    # filter to all significant products across all cell lines
    sub_adata = sub_adata[
        sub_adata.obs["product_name"].isin(all_sig_prods)
    ].copy()

    # Filter out lowest doses (for mem reasons)
    sub_adata = sub_adata[sub_adata.obs["dose"] >= 1000]

    # Add back control
    sub_adata = sub_adata.concatenate(adata[(adata.obs["product_name"] == "Vehicle") & (adata.obs["cell_type"] == cl)])

    # Indicate which are significant products for this cell line
    sub_adata.obs["sig_prod_cell_line"] = sub_adata.obs["product_name"].isin(per_cell_line_prods[cl])

    # indicate which doses are significant for this cell line
    sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
    sig_prod_doses.columns = ["product_dose"]
    sub_adata.obs["sig_prod_dose_cell_line"] = sub_adata.obs["product_dose"].isin(sig_prod_doses["product_dose"].values)

    # Subsample to 30k cells
    sc.pp.subsample(sub_adata, n_obs=min(30000, sub_adata.shape[0]))

    # label phases (Too much mem to do this before filtering doses)
    sub_adata.layers["counts"] = sub_adata.X.copy()
    sc.pp.filter_cells(sub_adata, min_genes=200)
    sc.pp.filter_genes(sub_adata, min_cells=20)
    sc.pp.normalize_per_cell(sub_adata, counts_per_cell_after=1e5)

    sc.pp.log1p(sub_adata)
    sc.pp.scale(sub_adata)
    sc.tl.score_genes_cell_cycle(sub_adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # # Filter on product doses with more than 30 observations
    # prod_dose_cell_thresh = 30
    # prod_dose_w_enough = sub_adata.obs.product_dose.value_counts()[sub_adata.obs.product_dose.value_counts() > 30].index.tolist()
    # sub_adata = sub_adata[sub_adata.obs["product_dose"].isin(prod_dose_w_enough)]

    # Revert counts to X
    sub_adata.X = sub_adata.layers["counts"]

    # Unwrap obs names from index
    sub_adata.obs_names = sub_adata.obs_names.values

    print(sub_adata)
    sub_adata.write(f"../data/sciplex_{cl}_significant_all_phases.h5ad")
    del sub_adata

# %%
