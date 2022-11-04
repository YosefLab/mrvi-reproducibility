import mrvi
import scanpy as sc

from utils import load_config, make_parents, wrap_kwargs


def fit_mrvi(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
    model_out: str,
):
    """Train a MrVI model."""
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

    make_parents([adata_out, model_out])
    model.save(model_out, save_anndata=False)
    adata.write(filename=adata_out)


if __name__ == "__main__":
    wrap_kwargs(fit_mrvi)()
