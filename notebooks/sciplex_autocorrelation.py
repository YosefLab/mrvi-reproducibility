# %%
import argparse
import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import scanpy as sc
import plotnine as p9

import scvi_v2
import scanpy as sc

# %%

adata = sc.read_h5ad(adata_in)
model = scvi_v2.MrVI.load(model_in, adata=adata)

cell_dists = model.get_local_sample_distances(
    adata, keep_cell=True,
)