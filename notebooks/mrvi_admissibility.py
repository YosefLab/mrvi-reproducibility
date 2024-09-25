# %%
from mrvi import MrVI
import flax.linen as nn
import scanpy as sc
import numpy as np
import jax.numpy as jnp
from scvi import REGISTRY_KEYS
from tqdm import tqdm
import pandas as pd
import jax

# %%
retrain = False

# %%
holdout_sample = "newcastle49"

if retrain:
    adata = sc.read_h5ad("../results/aws_pipeline/data/haniffa2.preprocessed.h5ad")
    adata

    # holdout a portion of one sample of the data
    holdout_frac = 0.5  # hold out high fraction to get varying admissibility scores
    sample_indices = np.where(adata.obs["patient_id"] == holdout_sample)[0]
    holdout_indices = np.random.choice(
        sample_indices, size=int(len(sample_indices) * holdout_frac), replace=False
    )
    train_indices = np.setdiff1d(np.arange(adata.n_obs), holdout_indices)
    adata_train = adata[train_indices].copy()
    adata_holdout = adata[holdout_indices].copy()

    adata_train.write_h5ad(
        "../results/aws_pipeline/data/haniffa2.admissibility.train.h5ad"
    )
    adata_holdout.write_h5ad(
        "../results/aws_pipeline/data/haniffa2.admissibility.holdout.h5ad"
    )

    MrVI.setup_anndata(
        adata_train,
        batch_key="Site",
        sample_key="patient_id",
    )
    model_kwargs = {
        "n_latent": 30,
        "n_latent_u": 5,
        "qz_nn_flavor": "attention",
        "px_nn_flavor": "attention",
        "qz_kwargs": {
            "use_map": True,
            "stop_gradients": False,
            "stop_gradients_mlp": True,
            "dropout_rate": 0.03,
        },
        "px_kwargs": {
            "stop_gradients": False,
            "stop_gradients_mlp": True,
            "h_activation": nn.softmax,
            "low_dim_batch": True,
            "dropout_rate": 0.03,
        },
        "learn_z_u_prior_scale": False,
        "z_u_prior": True,
        "u_prior_mixture": True,
        "u_prior_mixture_k": 20,
    }

    model = MrVI(adata_train, **model_kwargs)
    train_kwargs = {
        "max_epochs": 400,
        "batch_size": 1024,
        "early_stopping": True,
        "early_stopping_patience": 30,
        "check_val_every_n_epoch": 1,
        "early_stopping_monitor": "elbo_validation",
        "plan_kwargs": {"n_epochs_kl_warmup": 50, "lr": 3e-3},
    }
    model.train(**train_kwargs)
    model.save("../results/aws_pipeline/models/mrvi.admissibility.model")

else:
    adata_train = sc.read_h5ad(
        "../results/aws_pipeline/data/haniffa2.admissibility.train.h5ad"
    )
    adata_holdout = sc.read_h5ad(
        "../results/aws_pipeline/data/haniffa2.admissibility.holdout.h5ad"
    )
    model = MrVI.load(
        "../results/aws_pipeline/models/mrvi.admissibility.model", adata=adata_train
    )


# %%
# compute ball admissibility scores for heldout data
# to get admissibility scores of a heldout sample wrt only training idxs,
# need to do them one at a time
def get_admissibility_scores(
    self,
    adata,
    adata_holdout,
    sample_names,
    quantile_threshold: float = 0.05,
    admissibility_threshold: float = 0.0,
    minibatch_size: int = 256,
):
    adata = self._validate_anndata(adata)
    adata_holdout = self._validate_anndata(adata_holdout)

    # Compute u reps
    us_holdout = self.get_latent_representation(
        adata_holdout, use_mean=True, give_z=False
    )
    adata_holdout.obsm["U"] = us_holdout

    log_probs = []
    for sample_name in tqdm(sample_names):
        sample_idxs = np.where(adata.obs[self.sample_key] == sample_name)[0]
        ap = self.get_aggregated_posterior(adata=adata, indices=sample_idxs)

        log_probs_ = []
        n_splits = adata_holdout.n_obs // minibatch_size
        for u_rep in np.array_split(adata_holdout.obsm["U"], n_splits):
            log_probs_.append(
                jax.device_get(
                    ap.component_distribution.log_prob(
                        np.expand_dims(u_rep, ap.mixture_dim)
                    )
                    .sum(axis=1)
                    .max(axis=1, keepdims=True)
                )
            )

        log_probs_ = np.concatenate(log_probs_, axis=0)  # (n_cells, 1)
        log_probs.append(np.array(log_probs_))

    threshs_all = np.concatenate(log_probs)
    global_thresh = np.quantile(threshs_all, q=quantile_threshold)
    threshs = len(log_probs) * [global_thresh]
    threshs = np.array(threshs)
    log_ratios = (log_probs - threshs)[0]

    log_probs = np.concatenate(log_probs, 1)
    log_probs_df = pd.DataFrame(
        log_probs, index=adata_holdout.obs_names, columns=sample_names
    )
    log_probs_df.index.name = "cell_name"
    log_probs_df = log_probs_df.reset_index().melt(
        id_vars="cell_name", var_name="sample", value_name="log_probs"
    )
    admissibility_df = pd.DataFrame(
        log_ratios > admissibility_threshold,
        index=adata_holdout.obs_names,
        columns=sample_names,
    )
    admissibility_df.index.name = "cell_name"
    admissibility_df = admissibility_df.reset_index().melt(
        id_vars="cell_name", var_name="sample", value_name="is_admissible"
    )
    return log_probs_df, admissibility_df


# %%
adata_holdout.obs["_indices"] = np.arange(adata_holdout.n_obs)
adm_scores, admissibility_df = get_admissibility_scores(
    model,
    adata_train,
    adata_holdout,
    sample_names=[holdout_sample],
    minibatch_size=256,
)
print(adm_scores.head())
print(admissibility_df.head())
# %%
adm_scores = adm_scores.set_index("cell_name")
adm_scores

admissibility_df = admissibility_df.set_index("cell_name")
admissibility_df
# %%
# save outputs
adm_scores.to_csv(
    f"../results/aws_pipeline/admissibility/{holdout_sample}_admissibility_scores.csv",
    index=list(adata_holdout.obs_names),
)
admissibility_df.to_csv(
    f"../results/aws_pipeline/admissibility/{holdout_sample}_is_admissible.csv",
    index=list(adata_holdout.obs_names),
)


# %%
# compute reconstruction loss for heldout data
def compute_px_from_x(
    self,
    x,
    sample_index,
    batch_index,
    cf_sample=None,
    continuous_covs=None,
    label_index=None,
    mc_samples=10,
):
    """Compute normalized gene expression from observations"""
    log_library = 7.0 * jnp.ones_like(
        sample_index
    )  # placeholder, will be replaced by observed library sizes.
    inference_outputs = self.inference(
        x, sample_index, mc_samples=mc_samples, cf_sample=cf_sample, use_mean=True
    )
    generative_inputs = {
        "z": inference_outputs["z"],
        "library": log_library,
        "batch_index": batch_index,
        "continuous_covs": continuous_covs,
        "label_index": label_index,
    }
    generative_outputs = self.generative(**generative_inputs)
    return generative_outputs["px"], inference_outputs["u"], log_library


def compute_sample_cf_reconstruction_scores(
    self,
    sample_idx,
    adata=None,
    indices=None,
    batch_size=256,
):
    self._check_if_trained(warn=False)
    adata = self._validate_anndata(adata)
    sample_name = self.sample_order[sample_idx]

    scdl = self._make_data_loader(
        adata=adata, batch_size=batch_size, indices=indices, iter_ndarray=True
    )

    def _get_all_inputs(
        inputs,
    ):
        x = jnp.array(inputs[REGISTRY_KEYS.X_KEY])
        sample_index = jnp.array(inputs["sample"])
        batch_index = jnp.array(inputs[REGISTRY_KEYS.BATCH_KEY])
        continuous_covs = inputs.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        label_index = inputs.get(REGISTRY_KEYS.LABELS_KEY, None)
        if continuous_covs is not None:
            continuous_covs = jnp.array(continuous_covs)
        return {
            "x": x,
            "sample_index": sample_index,
            "batch_index": batch_index,
            "continuous_covs": continuous_covs,
            "label_index": label_index,
        }

    scores = []
    for array_dict in tqdm(scdl):
        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs

        inputs = _get_all_inputs(array_dict)
        px, u, log_library_placeholder = self.module.apply(
            vars_in,
            rngs=rngs,
            method=compute_px_from_x,
            x=inputs["x"],
            sample_index=inputs["sample_index"],
            batch_index=inputs["batch_index"],
            cf_sample=np.ones(inputs["x"].shape[0]) * sample_idx,
            continuous_covs=inputs["continuous_covs"],
            label_index=inputs["label_index"],
        )

        scores.append(px.log_prob(inputs["x"]).sum(axis=1))

    all_scores = np.concatenate(scores, axis=0)
    adata_index = adata[indices] if indices is not None else adata
    return pd.DataFrame(
        all_scores,
        index=list(adata_index.obs_names),
        columns=[f"{sample_name}_score"],
    )


# %%
sample_idx = np.where(model.sample_order == holdout_sample)[0][0]
holdout_reconstruction_scores = compute_sample_cf_reconstruction_scores(
    model, sample_idx=sample_idx, adata=adata_holdout
)
holdout_reconstruction_scores.head()

# %%
# save outputs
holdout_reconstruction_scores.to_csv(
    f"../results/aws_pipeline/admissibility/{holdout_sample}_reconstruction_scores.csv",
    index=list(adata_holdout.obs_names),
)
# %%
# plot results
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame with both scores
plot_data = pd.DataFrame(
    {
        "Reconstruction Scores": holdout_reconstruction_scores[
            f"{holdout_sample}_score"
        ],
        "Admissibility Scores": adm_scores["log_probs"],
        "Is Admissible": admissibility_df["is_admissible"].astype(str),
    }
)

# Create the scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=plot_data,
    x="Admissibility Scores",
    y="Reconstruction Scores",
    hue="Is Admissible",
)
plt.title(
    f"Admissibility Scores vs Reconstruction Log Probs for Heldout {holdout_sample} Cells"
)

# # Add a trend line
# sns.regplot(
#     data=plot_data,
#     x="Admissibility Scores",
#     y="Reconstruction Scores",
#     scatter=False,
#     color="red",
#     line_kws={"linestyle": "--"},
# )

plt.xlabel("Admissibility Scores")
plt.ylabel("Heldout Reconstruction Log Probability")

plt.tight_layout()


# Save the plot
plt.savefig(
    f"../results/aws_pipeline/admissibility/{holdout_sample}_admissibility_vs_reconstruction.png"
)
plt.savefig(
    f"../results/aws_pipeline/admissibility/{holdout_sample}_admissibility_vs_reconstruction.svg"
)
plt.show()
plt.close()

# Print correlation coefficient
correlation = plot_data["Reconstruction Scores"].corr(plot_data["Admissibility Scores"])
print(
    f"Correlation coefficient between Reconstruction and Admissibility Scores: {correlation:.4f}"
)


# %%
