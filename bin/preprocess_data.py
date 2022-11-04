import scanpy as sc
from anndata import AnnData

from utils import load_config, make_parents, wrap_kwargs


def _hvg(adata: AnnData, **kwargs) -> None:
    """Select highly-variable genes in-place."""
    kwargs.update({"subset": True})
    sc.pp.highly_variable_genes(adata, **kwargs)


def preprocess_data(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
):
    """
    Preprocess an input AnnData object and saves it to a new file.

    Performs the following steps:

    1. Highly variable genes selection

    TODO: Add more preprocessing steps as necessary

    Parameters
    ----------
    adata_in
        Input AnnData path
    config_in
        Input dataset configuration path
    adata_out
        Output preprocessed AnnData path
    """
    config = load_config(config_in)
    adata = sc.read(adata_in)
    hvg_kwargs = config.get("hvg_kwargs", {})
    
    _hvg(adata, **hvg_kwargs)
    
    make_parents(adata_out)
    adata.write(filename=adata_out)


if __name__ == "__main__":
    wrap_kwargs(preprocess_data)()
