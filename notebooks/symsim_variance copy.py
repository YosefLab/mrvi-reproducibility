# %%
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

import scanpy as sc
import anndata
import scvi_v2
import scvi

import matplotlib.pyplot as plt
import seaborn as sns


# %%
adata = sc.read("../results/aws_pipeline/symsim_new.preprocessed.h5ad")

# %%
# Use run models env for this
scvi_v2.MrVI.setup_anndata(adata, batch_key="batch", sample_key="donor")
model_kwargs = {
    "qz_nn_flavor": "linear",
}
model = scvi_v2.MrVI(adata, **model_kwargs)

# %%
# model.train(plan_kwargs={"lr":1e-4})
model.train()

# %%
# Check loss curves
plt.plot(model.history["elbo_train"], label="ELBO")
plt.plot(model.history["reconstruction_loss_train"], label="Reconstruction Loss")
plt.plot(model.history["kl_local_train"], label="KL Local")
plt.legend()

# %%
plt.plot(model.history["train_loss_epoch"])

# %%
norm_dists = model.get_local_sample_distances(adata, normalize_distances=True)
dists = model.get_local_sample_distances(adata, normalize_distances=False)


# %%
scdl = model._make_data_loader(adata=adata, batch_size=1000, iter_ndarray=True, shuffle=False)
first_batch = next(iter(scdl), None)
means, vars = model._compute_local_baseline_dists(first_batch)
avg_mean, avg_var = np.mean(means), np.mean(vars)
# %%
# alternate computation
model.can_compute_normalized_dists = False
g_means, g_vars = model._compute_local_baseline_dists(first_batch)
g_avg_mean, g_avg_var = np.mean(g_means), np.mean(g_vars)
model.can_compute_normalized_dists = True

# %%
plt.hist(dists["cell"].data.flatten(), label="dists", bins=20, alpha=0.5, density=True)
# plt.hist(norm_dists["cell"].data.flatten(), label="norm_dists", bins=20, alpha=0.5, density=True)

# Define the range of the x-axis
x = np.linspace(avg_mean - 4 * np.sqrt(avg_var), avg_mean + 4 * np.sqrt(avg_var), 1000)

# Calculate the PDF values
pdf = (1 / np.sqrt(2 * np.pi * avg_var)) * np.exp(
    -((x - avg_mean) ** 2) / (2 * avg_var)
)

# Plot the PDF using a dotted line
plt.plot(x, pdf, linestyle="--")

g_x = np.linspace(g_avg_mean - 4 * np.sqrt(g_avg_var), g_avg_mean + 4 * np.sqrt(g_avg_var), 1000)
# Calculate the PDF values
g_pdf = (1 / np.sqrt(2 * np.pi * g_avg_var)) * np.exp(
    -((g_x - g_avg_mean) ** 2) / (2 * g_avg_var)
)

# Plot the PDF using a dotted line
plt.plot(g_x, g_pdf, linestyle="--")
# %%
# plot heatmap of avg dist matrix
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    dists["cell"].data.mean(axis=0).T,
    ax=ax,
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    norm_dists["cell"].data.mean(axis=0).T,
    ax=ax,
)

# %%
