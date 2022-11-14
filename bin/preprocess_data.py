import scanpy as sc
from anndata import AnnData

from utils import load_config, make_parents, wrap_kwargs


def _hvg(adata: AnnData, **kwargs) -> AnnData:
    """Select highly-variable genes in-place."""
    kwargs.update({"subset": True})
    sc.pp.highly_variable_genes(adata, **kwargs)
    return adata


@wrap_kwargs
def preprocess_data(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
) -> AnnData:
    """
    Preprocess an input AnnData object and saves it to a new file.

    Performs the following steps:

    1. Highly variable genes selection

    TODO: Add more preprocessing steps as necessary

    Parameters
    ----------
    adata_in
        Path to the input AnnData object.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the preprocessed AnnData object.
    """
    config = load_config(config_in)
    adata = sc.read(adata_in)
    hvg_kwargs = config.get("hvg_kwargs", None)
    
    if hvg_kwargs is not None:
        adata = _hvg(adata, **hvg_kwargs)
    
    make_parents(adata_out)
    adata.write(filename=adata_out)
    return adata


if __name__ == "__main__":
    preprocess_data()
