import scanpy as sc
import scib_metrics as metrics

from utils import load_config, make_parents, wrap_kwargs


def compute_scib_metrics(
    *,
    adata_in: str,
    config_in: str,
    table_out: str,
) -> None:
    """
    Compute integration metrics.
    
    Parameters
    ----------
    adata_in
        Path to input AnnData object with integrated data.
    config_in
        Path to the dataset configuration file.
    table_out
        Path to write output CSV table with integration metrics.
    """
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)
    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    labels_key = config.get("labels_key", None)
    latent_key = config.get("latent_key", None)

    X_latent = adata.obsm[latent_key]
    labels = 


if __name__ == "__main__":
    wrap_kwargs(compute_scib_metrics)()
