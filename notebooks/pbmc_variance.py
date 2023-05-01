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

from sklearn.decomposition import PCA

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

# %%
# Get qeps for each cell
def get_qeps(
    model,
    adata=None,
    indices=None,
    batch_size=None,
    cf_sample=None,
) -> np.ndarray:
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    scdl = model._make_data_loader(
        adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
    )

    qeps_means = []
    qeps_scales = []
    all_z = []
    for array_dict in tqdm(scdl):
        cf_sample_idx = (
            None
            if cf_sample is None
            else np.broadcast_to(
                np.array(model.sample_order.tolist().index(cf_sample)),
                (array_dict[scvi.REGISTRY_KEYS.X_KEY].shape[0], 1),
            )
        )
        inference_kwargs = None if cf_sample is None else {"cf_sample": cf_sample_idx}
        jit_inference_fn = model.module.get_jit_inference_fn(
            inference_kwargs=inference_kwargs
        )
        outputs = jit_inference_fn(model.module.rngs, array_dict)
        qeps = outputs["qeps"]
        if qeps is not None:
            qeps_means.append(qeps.loc)
            qeps_scales.append(qeps.scale)
        else:
            all_z.append(outputs["z"])
    if len(qeps_means) > 0:
        qeps_mean = np.array(jax.device_get(jnp.concatenate(qeps_means, axis=0)))
        qeps_scale = np.array(jax.device_get(jnp.concatenate(qeps_scales, axis=0)))
        return qeps_mean, qeps_scale
    else:
        return np.array(jax.device_get(jnp.concatenate(all_z, axis=0)))


# %%
full_adata = sc.read("../data/pbmcs68k.h5ad")

# %%
# leiden cluster cells in scVI space
scvi.model.JaxSCVI.setup_anndata(
    full_adata,
)
m = scvi.model.JaxSCVI(full_adata)
m.train(
    max_epochs=400,
    batch_size=1024,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="reconstruction_loss_validation",
)
latent = m.get_latent_representation()
full_adata.obsm["X_scvi"] = latent
sc.pp.neighbors(full_adata, use_rep="X_scvi")
sc.tl.leiden(full_adata, resolution=1, key_added="leiden")


# %%
full_adata.write("../data/pbmcs68k_scvi.h5ad")

# %%
full_adata = sc.read("../data/pbmcs68k_scvi.h5ad")

# %%
# generate uniform samples and subsample one cluster in one sample
n_samples = 10
subsample_fraction = 0.1
full_adata.obs["donor"] = np.random.choice(
    n_samples, size=full_adata.n_obs, replace=True
)
largest_cluster_idx = full_adata.obs["leiden"].value_counts().idxmax()
to_subsample_idxs = np.where(
    (full_adata.obs["leiden"] == largest_cluster_idx) & (full_adata.obs["donor"] == 0)
)[0]
subsample_remove_idxs = np.random.choice(
    to_subsample_idxs,
    size=int((1 - subsample_fraction) * len(to_subsample_idxs)),
    replace=False,
)
subset_idxs = np.setdiff1d(np.arange(full_adata.n_obs), subsample_remove_idxs)
adata = full_adata[subset_idxs, :].copy()

# %%
# Use run models env for this
scvi_v2.MrVI.setup_anndata(adata, batch_key=None, sample_key="donor")
model_kwargs = {
    "n_latent": 10,
    "qz_nn_flavor": "linear",
    # "qz_kwargs": {},
    "z_u_prior_scale": 1.0,
}
model = scvi_v2.MrVI(adata, **model_kwargs)

# %%
model.train()

# %%
model.save("models/pbmcs68k_linear_01", save_anndata=True)

# %%
model = scvi_v2.MrVI.load("models/pbmcs68k_linear_01")
adata = model.adata

# %%
# Check loss curves
plt.plot(model.history["elbo_train"], label="ELBO")
plt.plot(model.history["reconstruction_loss_train"], label="Reconstruction Loss")
plt.plot(model.history["kl_local_train"], label="KL Local")
plt.legend()

# %%
plt.plot(model.history["train_loss_epoch"])

# %%
# Compute MDE of latent space
latent_z = model.get_latent_representation(give_z=True)
latent_u = model.get_latent_representation(give_z=False)

# %%
# PCA of latents
pca_z = PCA(n_components=2).fit_transform(latent_z)
pca_u = PCA(n_components=2).fit_transform(latent_u)

adata.obsm["pca_z"] = pca_z
adata.obsm["pca_u"] = pca_u
adata.obs["donor"] = adata.obs["donor"].astype("category")

# %%
sc.pl.embedding(
    adata,
    basis="pca_z",
    color=["leiden", "donor"],
    ncols=2,
)

# %%
sc.pl.embedding(
    adata,
    basis="pca_u",
    color=["leiden", "donor"],
    ncols=2,
)

# # %%
# mde_z = scvi.model.utils.mde(latent_z)
# mde_u = scvi.model.utils.mde(latent_u)

# # %%
# # Plot MDE of U
# adata.obsm["mde_u"] = mde_u
# sc.pl.embedding(
#     adata,
#     basis="mde_u",
#     color=["celltype", "donor"],
#     ncols=2,
# )

# # %%
# # Plot MDE of Z
# adata.obsm["mde_z"] = mde_z
# sc.pl.embedding(
#     adata,
#     basis="mde_z",
#     color=["celltype", "donor"],
#     ncols=2,
# )

# %%
# get qepses for subsampled
# all_z0_loc, all_z0_scale = get_qeps(model, cf_sample=0)
# all_z1_loc, all_z1_scale = get_qeps(model, cf_sample=1)
# all_z2_loc, all_z2_scale = get_qeps(model, cf_sample=2)
all_z0_loc = get_qeps(model, cf_sample=0)
all_z1_loc = get_qeps(model, cf_sample=1)
all_z2_loc = get_qeps(model, cf_sample=2)

# %%
# all_z0_scale = np.abs(all_z0_scale)
# all_z1_scale = np.abs(all_z1_scale)
# all_z2_scale = np.abs(all_z2_scale)
# %%
adata.obs["sample_0_1_l2"] = np.linalg.norm(all_z0_loc - all_z1_loc, axis=1)
adata.obs["sample_1_2_l2"] = np.linalg.norm(all_z1_loc - all_z2_loc, axis=1)

# %%
# adata.obs["sample_0_scale_l2"] = np.linalg.norm(all_z0_scale, axis=1)
# adata.obs["sample_1_scale_l2"] = np.linalg.norm(all_z1_scale, axis=1)

# %%
largest_cluster_idx = full_adata.obs["leiden"].value_counts().idxmax()
adata.obs["perturbed_cluster"] = (adata.obs["leiden"] == largest_cluster_idx).astype(
    int
)

# %%
shuffled_idxs = np.random.permutation(adata.obs.index)
sc.pl.embedding(
    adata[shuffled_idxs],
    basis="pca_u",
    color=[
        "sample_0_1_l2",
        "sample_1_2_l2",
        # "sample_0_scale_l2",
        # "sample_1_scale_l2",
        "perturbed_cluster",
    ],
    ncols=2,
)

# %%
# plot overlaying histograms of scales from each sample
all_l2 = adata.obs["sample_0_1_l2"]
in_cluster_l2 = all_l2[adata.obs["perturbed_cluster"] == 1]
neg_control_l2 = all_l2[adata.obs["perturbed_cluster"] == 0]
all_control_l2 = adata.obs["sample_1_2_l2"]
control_in_cluster_l2 = all_control_l2[adata.obs["perturbed_cluster"] == 1]
control_neg_cluster_l2 = all_control_l2[adata.obs["perturbed_cluster"] == 0]
plt.hist(in_cluster_l2, bins=50, label="In cluster", alpha=0.5, density=True)
plt.hist(neg_control_l2, bins=50, label="Negative Control", alpha=0.5, density=True)
plt.hist(
    control_in_cluster_l2, bins=100, label="Control in cluster", alpha=0.5, density=True
)
plt.hist(
    control_neg_cluster_l2,
    bins=100,
    label="Control negative cluster",
    alpha=0.5,
    density=True,
)
plt.legend()
plt.xlim((0, 1))
plt.title("linear")

# %%
