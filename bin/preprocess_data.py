import argparse
import json
import pathlib

import scanpy as sc
from anndata import AnnData


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def _hvg(adata: AnnData, **kwargs) -> None:
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
    """
    config = load_config(config_in)
    hvg_kwargs = config.get("hvg_kwargs", {})
    adata = sc.read(adata_in)
    
    _hvg(adata, **hvg_kwargs)
    
    path = pathlib.Path(adata_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(filename=adata_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset.")

    parser.add_argument(
        "--adata_in",
        dest="adata_in",
        type=str,
        help="Input raw AnnData path",
    )
    parser.add_argument(
        "--config_in",
        dest="config_in",
        type=str,
        help="Input dataset configuration path",
    )
    parser.add_argument(
        "--adata_out",
        dest="adata_out",
        type=str,
        help="Output preprocessed AnnData path",
    )
    
    args = parser.parse_args()
    preprocess_data(
        adata_in=args.adata_in,
        config_in=args.config_in,
        adata_out=args.adata_out,
    )
