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
    use_attention: str = "false",
    use_attention_ld: str = "false",
    use_attention_hd: str = "false",
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
    use_attention = use_attention.lower() == "true"
    use_attention_ld = use_attention_ld.lower() == "true"
    use_attention_hd = use_attention_hd.lower() == "true"


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
    if use_attention:
        model_kwargs.update(
            {
                "qz_nn_flavor": "attention",
                "qz_kwargs": {"use_map": False},
            }
        )
    if use_attention_ld:
        model_kwargs.update(
            {
                "qz_nn_flavor": "attention",
                "px_nn_flavor": "attention",
                "qz_kwargs": {"use_map": False, "stop_gradients": True},
                "px_kwargs": {"stop_gradients": True, "low_dim_batch": True},
                "learn_z_u_prior_scale": True,
            }
        )
    if use_attention_hd:
        model_kwargs.update(
            {
                "qz_nn_flavor": "attention",
                "px_nn_flavor": "attention",
                "qz_kwargs": {"use_map": False, "stop_gradients": True},
                "px_kwargs": {"stop_gradients": True, "low_dim_batch": False},
                "learn_z_u_prior_scale": True,
            }
        )
    model = scvi_v2.MrVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    make_parents(model_out)
    model.save(dir_path=model_out, overwrite=True, save_anndata=False)
    return model


if __name__ == "__main__":
    fit_scviv2()
