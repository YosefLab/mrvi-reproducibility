# %%
import pandas as pd
import scanpy as sc

# %%
adata = sc.read("/home/justin/ghrepos/scvi-v2-reproducibility/data/archive/sciplex_raw.h5ad")
# %%
adata

# %%
# Filter for cells in S phase
# From https://github.com/scverse/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
cell_cycle_genes = [x.strip() for x in open('../data/archive/regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

# %%
cell_lines = list(adata.obs["cell_type"].cat.categories)
cell_lines
# %%
for cl in cell_lines:
    sub_adata = adata[adata.obs["cell_type"] == cl]

    # Filter down to significant product doses
    sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
    sig_prod_doses.columns = ["product_dose"]

    sub_adata = sub_adata[
        sub_adata.obs["product_dose"].isin(sig_prod_doses["product_dose"].values)
    ]
    # Add back control
    sub_adata = sub_adata.concatenate(adata[(adata.obs["product_dose"] == "Vehicle_0") & (adata.obs["cell_type"] == cl)])

    # Filter down to S phase (Too much mem to do this before filtering doses)
    sub_adata.layers["counts"] = sub_adata.X.copy()
    sc.pp.filter_cells(sub_adata, min_genes=200)
    sc.pp.filter_genes(sub_adata, min_cells=3)
    sc.pp.normalize_per_cell(sub_adata, counts_per_cell_after=1e5)

    sc.pp.log1p(sub_adata)
    sc.pp.scale(sub_adata)
    sc.tl.score_genes_cell_cycle(sub_adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sub_adata = sub_adata[sub_adata.obs['phase'] == 'S']

    # Filter on product doses with more than 30 observations
    prod_dose_cell_thresh = 30
    prod_dose_w_enough = sub_adata.obs.product_dose.value_counts()[sub_adata.obs.product_dose.value_counts() > 30].index.tolist()
    sub_adata = sub_adata[sub_adata.obs["product_dose"].isin(prod_dose_w_enough)]

    # Create dummy batch column
    sub_adata.obs["dummy_batch"] = "0"

    # Revert counts to X
    sub_adata.X = sub_adata.layers["counts"]

    # Unwrap obs names from index
    sub_adata.obs_names = sub_adata.obs_names.values

    print(sub_adata)
    sub_adata.write(f"../data/sciplex_{cl}_significant.h5ad")
    del sub_adata

# %%
# For keeping all phases
for cl in cell_lines:
    sub_adata = adata[adata.obs["cell_type"] == cl]

    # Filter down to significant product doses
    sig_prod_doses = pd.read_csv(f"output/{cl}.csv", header=None)
    sig_prod_doses.columns = ["product_dose"]

    sub_adata = sub_adata[
        sub_adata.obs["product_dose"].isin(sig_prod_doses["product_dose"].values)
    ]
    # Add back control
    sub_adata = sub_adata.concatenate(adata[(adata.obs["product_dose"] == "Vehicle_0") & (adata.obs["cell_type"] == cl)])

    # label phases (Too much mem to do this before filtering doses)
    sub_adata.layers["counts"] = sub_adata.X.copy()
    sc.pp.filter_cells(sub_adata, min_genes=200)
    sc.pp.filter_genes(sub_adata, min_cells=3)
    sc.pp.normalize_per_cell(sub_adata, counts_per_cell_after=1e5)

    sc.pp.log1p(sub_adata)
    sc.pp.scale(sub_adata)
    sc.tl.score_genes_cell_cycle(sub_adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # Filter on product doses with more than 30 observations
    prod_dose_cell_thresh = 30
    prod_dose_w_enough = sub_adata.obs.product_dose.value_counts()[sub_adata.obs.product_dose.value_counts() > 30].index.tolist()
    sub_adata = sub_adata[sub_adata.obs["product_dose"].isin(prod_dose_w_enough)]

    # Create dummy batch column
    sub_adata.obs["dummy_batch"] = "0"

    # Revert counts to X
    sub_adata.X = sub_adata.layers["counts"]

    # Unwrap obs names from index
    sub_adata.obs_names = sub_adata.obs_names.values

    print(sub_adata)
    sub_adata.write(f"../data/sciplex_{cl}_significant_all_phases.h5ad")
    del sub_adata

# %%
