import jax.numpy as jnp
import flax.linen as nn
import scanpy as sc
import scvi_v2
from utils import load_config, make_parents, wrap_kwargs

BASE_MOG_DICT = {
    "qz_nn_flavor": "attention",
    "px_nn_flavor": "attention",
    "qz_kwargs": {
        "use_map": True,
        "stop_gradients": False,
        "stop_gradients_mlp": True,
    },
    "px_kwargs": {
        "stop_gradients": False,
        "stop_gradients_mlp": True,
        "h_activation": nn.softmax,
        "low_dim_batch": True,
    },
    "learn_z_u_prior_scale": False,
    "z_u_prior": False,
    "u_prior_mixture": False,
    "u_prior_mixture_k": 20,
}


@wrap_kwargs
def fit_scviv2(
    *,
    adata_in: str,
    config_in: str,
    model_out: str,
    use_attention_noprior: str = "false",
    use_attention_no_prior_mog: str = "false",
    z30: str = "false",
    z20_u5: str = "false",
    z50_u5: str = "false",
    z30_u5: str = "false",
    z100_u5: str = "false",
) -> scvi_v2.MrVI:
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
    use_attention_noprior = use_attention_noprior.lower() == "true"
    use_attention_no_prior_mog = use_attention_no_prior_mog.lower() == "true"
    z30 = z30.lower() == "true"
    z20_u5 = z20_u5.lower() == "true"
    z50_u5 = z50_u5.lower() == "true"
    z30_u5 = z30_u5.lower() == "true"
    z100_u5 = z100_u5.lower() == "true"

    config = load_config(config_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    model_kwargs = config.get("scviv2_model_kwargs", {})
    train_kwargs = config.get("scviv2_train_kwargs", {})
    adata = sc.read(adata_in)

    scvi_v2.MrVI.setup_anndata(
        adata,
        batch_key=batch_key,
        sample_key=sample_key,
    )
    if use_attention_noprior:
        model_kwargs.update(
            {
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
                    "dropout_rate": 0.03,
                    "low_dim_batch": True,
                },
                "learn_z_u_prior_scale": False,
                "z_u_prior": True,
                "u_prior_mixture": False,
            }
        )
    if use_attention_no_prior_mog:
        model_kwargs.update(BASE_MOG_DICT)
    if z30:
        model_kwargs.update({**BASE_MOG_DICT, "n_latent": 30})
    if z20_u5:
        model_kwargs.update({**BASE_MOG_DICT, "n_latent": 20, "n_latent_u": 5})
    if z50_u5:
        model_kwargs.update({**BASE_MOG_DICT, "n_latent": 50, "n_latent_u": 5})
    if z30_u5:
        model_kwargs.update({**BASE_MOG_DICT, "n_latent": 30, "n_latent_u": 5})
    if z100_u5:
        model_kwargs.update({**BASE_MOG_DICT, "n_latent": 100, "n_latent_u": 5})

    model = scvi_v2.MrVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    make_parents(model_out)
    model.save(dir_path=model_out, overwrite=True, save_anndata=False)
    return model


if __name__ == "__main__":
    fit_scviv2()
