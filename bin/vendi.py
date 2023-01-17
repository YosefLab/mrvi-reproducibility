import pandas as pd
import scanpy as sc
from vendi_score import vendi
import xarray as xr

from utils import load_config, make_parents, wrap_kwargs


@wrap_kwargs
def compute_vendi(
    *,
    distance_matrix_in: str,
    config_in: str,
    table_out: str,
) -> pd.DataFrame:
    """
    Compute Vendi score.

    Parameters
    ----------
    distance_matrix_in
        Path to input AnnData object with integrated data.
    config_in
        Path to the dataset configuration file.
    table_out
        Path to write output CSV table with Vendi score.
    """
    config = load_config(config_in)
    distance_matrix = xr.open_dataset(distance_matrix_in)

    local_sample_dists_key = adata.uns["local_sample_dists_key"]
    local_sample_dists = adata.obsm[local_sample_dists_key]
    local_sample_similarities = 1 / (1 + local_sample_dists)

    vendi_scores = []
    for i in range(local_sample_similarities.shape[0]):
        vendi_scores.append(vendi.score_K(local_sample_similarities[i]))

    vendi_dict = {"obs_id": adata.obs.index, "vendi_score": vendi_scores}
    df = pd.DataFrame.from_dict(vendi_dict)

    make_parents(table_out)
    df.to_csv(table_out)
    return df


if __name__ == "__main__":
    compute_vendi()
