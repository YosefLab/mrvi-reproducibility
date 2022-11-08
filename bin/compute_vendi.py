import pandas as pd
import scanpy as sc

from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def compute_vendi(
    *,
    adata_in: str,
    config_in: str,
    table_out: str,
) -> pd.DataFrame:
    """
    Compute Vendi score.

    Parameters
    ----------
    adata_in
        Path to input AnnData object with integrated data.
    config_in
        Path to the dataset configuration file.
    table_out
        Path to write output CSV table with Vendi score.
    """
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)

    metrics = {"example_metric": [0.0]}
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])

    make_parents(table_out)
    df.to_csv(table_out)
    return df


if __name__ == "__main__":
    compute_vendi()
