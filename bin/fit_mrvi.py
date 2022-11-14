import mrvi
import scanpy as sc

from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def fit_mrvi(
    *,
    adata_in: str,
    config_in: str,
    model_out: str,
) -> mrvi.MrVI:
    """
    Train a MrVI model.
    
    Parameters
    ----------
    adata_in
        Path to the preprocessed AnnData object.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the latent AnnData object.
    model_out
        Path to write the trained MrVI model.
    """
    config = load_config(config_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    train_kwargs = config.get("train_kwargs", {})
    adata = sc.read(adata_in)

    mrvi.MrVI.setup_anndata(
        adata, categorical_nuisance_keys=[batch_key], sample_key=sample_key,
    )
    model = mrvi.MrVI(adata)
    model.train(**train_kwargs)

    make_parents(model_out)
    model.save(dir_path=model_out, overwrite=True, save_anndata=False)
    return model


if __name__ == "__main__":
    fit_mrvi()
