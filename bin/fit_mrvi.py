import jax.numpy as jnp
import flax.linen as nn
import scanpy as sc
import mrvi
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_mrvi(
    *,
    adata_in: str,
    config_in: str,
    model_out: str,
    use_reference: str = "true",
    use_linear_uz: str = "false",
    use_mlp_uz: str = "false",
    use_same_dim_uz: str = "false",
    use_encoder_regularnorm: str = "false",
    use_iso_prior: str = "false",
) -> mrvi.MrVI:
    """
    Train a MrVI model.

    Parameters
    ----------
    adata_in
        Path to the preprocessed AnnData object.
    config_in
        Path to the dataset configuration file.
    model_out
        Path to write the trained MrVI model.
    """

    use_reference = use_reference.lower() == "true"
    use_linear_uz = use_linear_uz.lower() == "true"
    use_mlp_uz = use_mlp_uz.lower() == "true"
    use_same_dim_uz = use_same_dim_uz.lower() == "true"
    use_encoder_regularnorm = use_encoder_regularnorm.lower() == "true"
    use_iso_prior = use_iso_prior.lower() == "true"

    config = load_config(config_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    model_kwargs = config.get("mrvi_model_kwargs", {})
    train_kwargs = config.get("mrvi_train_kwargs", {})
    adata = sc.read(adata_in)

    mrvi.MrVI.setup_anndata(
        adata,
        batch_key=batch_key,
        sample_key=sample_key,
    )
    model_kwargs.update(
        {
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
    )

    if use_reference:
        pass

    if use_linear_uz:
        model_kwargs.update(
            {
                "qz_nn_flavor": "linear",
                "px_nn_flavor": "linear",
                "px_kwargs": {},
                "qz_kwargs": {},
            }
        )
    if use_mlp_uz:
        model_kwargs.update(
            {
                "qz_nn_flavor": "mlp",
                "px_nn_flavor": "linear",
                "px_kwargs": {},
                "qz_kwargs": {},
            }
        )
    if use_same_dim_uz:
        model_kwargs.update(
            {
                "n_latent": 30,
                "n_latent_u": 30,
            }
        )

    if use_encoder_regularnorm:
        model_kwargs.update(
            {
                "qu_kwargs": {
                    "use_conditional": False,
                }
            }
        )

    if use_iso_prior:
        model_kwargs.update(
            {
                "u_prior_mixture": False,
            }
        )

    model = mrvi.MrVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    make_parents(model_out)
    model.save(dir_path=model_out, overwrite=True, save_anndata=False)
    return model


if __name__ == "__main__":
    fit_mrvi()
