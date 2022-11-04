import argparse
import json

import scanpy as sc


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def compute_vendi(
    *,
    adata_in: str,
    config_in: str,
    table_out: str,
) -> None:
    """Compute integration metrics."""
    config = load_config(config_in)

    adata = sc.read_h5ad(adata_in)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for a dataset")

    parser.add_argument(
        "--adata_in",
        dest="adata_in",
        type=str,
        help="Input AnnData path",
    )
    parser.add_argument(
        "--config_in",
        dest="config_in",
        type=str,
        help="Input dataset configuration path",
    )
    parser.add_argument(
        "--table_out",
        dest="table_out",
        type=str,
        help="Output results table path",
    )
    args = parser.parse_args()
    compute_vendi(
        adata_in=args.adata_in,
        config_in=args.config_in,
        adata_out=args.table_out,
    )