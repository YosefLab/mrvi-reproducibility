# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import jax
import jax.numpy as jnp
import faiss

import scvi_v2
import scvi
import scanpy as sc
import seaborn as sns
import scipy
import xarray as xr

# %%
method_name = "scviv2"
cell_line = "A549"
data_variant = "significant_filtered"

base_dir_path = "/home/justin/ghrepos/scvi-v2-reproducibility"
results_dir_path = os.path.join(base_dir_path, "results/3_8_deg_filter_sciplex_pipeline")
adata_path = os.path.join(
    results_dir_path,
    f"data/sciplex_{cell_line}_{data_variant}_all_phases.preprocessed.h5ad",
)


# %%
adata = sc.read_h5ad(adata_path)
# %%
product_names = adata.obs["product_name"].unique().to_numpy()

for product_name in product_names:
    print(f"{product_name};")

# %%
voi = "target"
for pname in product_names:
    print(f'{adata[adata.obs["product_name"] == pname][0].obs[voi].to_numpy()[0]};')
# %%
