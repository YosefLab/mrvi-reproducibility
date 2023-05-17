import jax.numpy as jnp
import flax.linen as nn
import scanpy as sc
import scvi_v2
from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_scviv2(
    *,
    adata_in: str,
    config_in: str,
    model_out: str,
    use_mlp: str = "false",
    use_mlp_smallu: str = "false",
    use_attention: str = "false",
    use_attention_smallu: str = "false",
    use_attention_noprior: str = "false",
    use_attention_no_prior_mog: str = "false",
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
    use_mlp = use_mlp.lower() == "true"
    use_mlp_smallu = use_mlp_smallu.lower() == "true"
    use_attention = use_attention.lower() == "true"
    use_attention_smallu = use_attention_smallu.lower() == "true"
    use_attention_noprior = use_attention_noprior.lower() == "true"
    use_attention_no_prior_mog = use_attention_no_prior_mog.lower() == "true"

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
    if use_mlp:
        model_kwargs.update(
            {
                "qz_nn_flavor": "mlp",
                "qz_kwargs": {"use_map": False, "stop_gradients": True},
            }
        )
    if use_mlp_smallu:
        n_latent = model_kwargs.get("n_latent", 20)
        model_kwargs.update(
            {
                "qz_nn_flavor": "mlp",
                "qz_kwargs": {"use_map": False, "stop_gradients": True},
                "n_latent_u": n_latent // 2,
            }
        )
    if use_attention:
        model_kwargs.update(
            {
                "qz_nn_flavor": "attention",
                "px_nn_flavor": "attention",
                "qz_kwargs": {
                    "use_map": False,
                    "stop_gradients": False,
                    "stop_gradients_mlp": True,
                },
                "px_kwargs": {
                    "stop_gradients": False,
                    "stop_gradients_mlp": True,
                    "h_activation": nn.softmax,
                },
                "learn_z_u_prior_scale": False,
                "z_u_prior": True,
                "z_u_prior_scale": 2,
                "u_prior_scale": 2,
            }
        )
    if use_attention_smallu:
        n_latent = model_kwargs.get("n_latent", 20)
        model_kwargs.update(
            {
                "qz_nn_flavor": "attention",
                "px_nn_flavor": "attention",
                "qz_kwargs": {
                    "use_map": False,
                    "stop_gradients": False,
                    "stop_gradients_mlp": True,
                },
                "px_kwargs": {
                    "stop_gradients": False,
                    "stop_gradients_mlp": True,
                    "h_activation": nn.softmax,
                },
                "learn_z_u_prior_scale": False,
                "z_u_prior": True,
                "z_u_prior_scale": 2,
                "qz_nn_flavor": "attention",
                "n_latent_u": n_latent // 2,
            }
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
                },
                "px_kwargs": {
                    "stop_gradients": False,
                    "stop_gradients_mlp": True,
                    "h_activation": jnp.exp,
                },
                "learn_z_u_prior_scale": False,
                "z_u_prior": False,
                "u_prior_mixture": True,
                "u_prior_mixture_k": 20,
            }
        )
    if use_attention_no_prior_mog:
        model_kwargs.update(
            {
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
                    "h_activation": jnp.exp,
                },
                "learn_z_u_prior_scale": False,
                "z_u_prior": False,
                "u_prior_mixture": False,
            }
        )
    model = scvi_v2.MrVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    make_parents(model_out)
    model.save(dir_path=model_out, overwrite=True, save_anndata=False)
    return model


if __name__ == "__main__":
    fit_scviv2()
