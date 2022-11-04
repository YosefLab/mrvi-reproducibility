import scanpy as sc
import scib_metrics as metrics

from utils import load_config, make_parents, wrap_kwargs


def compute_scib_metrics(
    *,
    adata_in: str,
    config_in: str,
    table_out: str,
) -> None:
    """Compute integration metrics."""
    config = load_config(config_in)
    adata = sc.read_h5ad(adata_in)


if __name__ == "__main__":
    wrap_kwargs(compute_scib_metrics)()
