import pandas as pd
import scanpy as sc
from vendi_score import vendi
import xarray as xr
import numpy as np

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
    celltype_key = config.get("labels_key", None)
    if celltype_key is None:
        # Empty dataframe
        df = pd.DataFrame()
        make_parents(table_out)
        df.to_csv(table_out)
        return df

    try:
        distance_matrix = xr.open_dataarray(distance_matrix_in)
    except ValueError:
        distance_matrix = xr.open_dataset(distance_matrix_in)[celltype_key]
        distance_matrix = distance_matrix.rename("distance")

    vmax = np.percentile(distance_matrix.values, 95)
    local_sample_similarities = (vmax - distance_matrix) / vmax

    clusters = local_sample_similarities.coords[f"{celltype_key}_name"].values
    vendi_scores = []
    for cluster in clusters:
        cluster_sims = local_sample_similarities.loc[cluster].values
        vendi_scores.append(vendi.score_K(cluster_sims))

    vendi_dict = {"cluster_name": clusters, "vendi_score": vendi_scores}
    df = pd.DataFrame.from_dict(vendi_dict)

    make_parents(table_out)
    df.to_csv(table_out)
    return df


if __name__ == "__main__":
    compute_vendi()
