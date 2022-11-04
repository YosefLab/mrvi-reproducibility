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


def get_latent_mrvi(
    *,
    adata_in: str,
    model_in: str,
    config_in: str,
    adata_out: str,
) -> None:
    """Get latent space from MrVI model."""
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)
    model = mrvi.MrVI.load(model_in)
    latent_key = config.get("latent_key", "X_mrvi_u")

    adata[latent_key] = model.get_latent_representation(adata, give_z=False)

    path = pathlib.Path(adata_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(filename=adata_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get latent space from a MrVI model")

    parser.add_argument(
        "--adata_in",
        dest="adata_in",
        type=str,
        help="Input preprocessed AnnData attached to MrVI path",
    )
    parser.add_argument(
        "--model_in",
        dest="model_in",
        type=str,
        help="Input trained MrVI model path",
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
        help="Output AnnData with latent space path",
    )
    args = parser.parse_args()
    get_latent_mrvi(
        adata_in=args.adata_in,
        model_in=args.model_in,
        config_in=args.config_in,
        adata_out=args.adata_out,
    )