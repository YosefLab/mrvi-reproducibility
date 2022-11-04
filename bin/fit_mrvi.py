import argparse
import json
import pathlib

import mrvi
import scanpy as sc


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


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
    
    model_path = pathlib.Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out, save_anndata=False)
    adata_path = pathlib.Path(adata_out)
    adata_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(filename=adata_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate scRNA-seq data with MrVI")

    parser.add_argument(
        "--adata_in",
        dest="adata_in",
        type=str,
        help="Input preprocessed AnnData path",
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
        help="Output AnnData path",
    )
    parser.add_argument(
        "--model_out",
        dest="model_out",
        type=str,
        help="Output trained MrVI model path",
    )
    args = parser.parse_args()
    fit_mrvi(
        adata_in=args.adata_in,
        config=args.config_in,
        adata_out=args.adata_out,
        model_out=args.model_out,
    )
