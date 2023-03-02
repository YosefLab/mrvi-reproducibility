import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import silhouette_score
import xarray as xr
from utils import (
    determine_if_file_empty,
    load_config,
    make_parents,
    wrap_kwargs,
)


@wrap_kwargs
def compute_sciplex_metrics(
    *,
    distance_matrices_in,
    gt_clusters_in,
    config_in,
    table_out,
):
    """Main fn for computing sciplex specific metrics.

    Parameters
    ----------
    distance_matrices_in:
        path to inferred distance matrices
    gt_clusters_in:
        paths to approximate ground truth clusters retrieved from bulk data
    config_in :
        path to config file
    table_out :
        desired output table file
    """
    basename = os.path.basename(distance_matrices_in)
    cell_line = basename.split("_")[1]
    model_name = basename.split(".")[1]
    distance_type = basename.split(".")[2]
    config = load_config(config_in)
    celltype_key = config["labels_key"]
    dim_name = f"{celltype_key}_name"

    try:
        inferred_mats = xr.open_dataarray(distance_matrices_in)
    except ValueError:
        inferred_mats = xr.open_dataset(distance_matrices_in)[celltype_key]
    inferred_mats = inferred_mats.rename("distance")
    phases = inferred_mats[dim_name].data

    make_parents(table_out)

    metrics_dict = {}

    gt_cluster_labels_df = None
    gt_clusters_in = gt_clusters_in.split(",")
    for gt_clusters_in_path in gt_clusters_in:
        gt_cell_line = os.path.basename(gt_clusters_in_path).split("_")[0]
        if gt_cell_line == cell_line:
            if determine_if_file_empty(gt_clusters_in_path):
                break

            gt_cluster_labels_df = pd.read_csv(gt_clusters_in_path, index_col=0)
            # Assign them all at 10000 nM dose
            new_sample_idx = [prod + "_10000" for prod in list(gt_cluster_labels_df.index)]
            gt_cluster_labels_df.index = new_sample_idx
            # Filter on samples in the distance matrix
            gt_cluster_labels_df = gt_cluster_labels_df.loc[np.intersect1d(inferred_mats.coords["sample_x"].data, gt_cluster_labels_df.index.array)]
            break

    if gt_cluster_labels_df is None:
        Path(table_out).touch()

    if gt_cluster_labels_df is not None:
        gt_silhouette_scores = []

        for phase in phases:
            dist_inferred = (
                inferred_mats.loc[phase]
                .sel(
                    sample_x=gt_cluster_labels_df.index,
                    sample_y=gt_cluster_labels_df.index,
                )
                .values
            )
            np.fill_diagonal(dist_inferred, 0)
            asw = silhouette_score(
                dist_inferred, gt_cluster_labels_df.values, metric="precomputed"
            )
            asw = (asw + 1) / 2
            gt_silhouette_scores.append(asw)

        metrics_dict["gt_silhouette_score"] = gt_silhouette_scores

    # Compute product dose metrics
    all_products = set()
    all_doses = set()
    for sample_name in inferred_mats.sample_x.data:
        product_name, dose = sample_name.split("_")
        if product_name != "Vehicle":
            all_products.add(product_name)
        if dose != "0":
            all_doses.add(dose)

    in_product_all_dist_avg_percentile = []
    in_product_top_2_dist_avg_percentile = []
    top_two_doses = ["1000", "10000"]
    for phase in phases:
        phase_dists = inferred_mats.sel(phase_name=phase)
        phase_dists_arr = phase_dists.data
        non_diag_mask = (
            np.ones(shape=phase_dists_arr.shape) - np.identity(phase_dists_arr.shape[0])
        ).astype(bool)
        in_prod_mask = np.zeros(shape=phase_dists_arr.shape, dtype=bool)
        in_prod_top_two_mask = np.zeros(shape=phase_dists_arr.shape, dtype=bool)
        for product_name in all_products:
            for dosex in all_doses:
                for dosey in all_doses:
                    if dosex == dosey:
                        continue
                    dosex_idx = np.where(
                        phase_dists.sample_x.data == f"{product_name}_{dosex}"
                    )[0]
                    if len(dosex_idx) == 0:
                        continue
                    dosey_idx = np.where(
                        phase_dists.sample_y.data == f"{product_name}_{dosey}"
                    )[0]
                    if len(dosey_idx) == 0:
                        continue
                    in_prod_mask[dosex_idx[0], dosey_idx[0]] = True

                    if dosex in top_two_doses and dosey in top_two_doses:
                        in_prod_top_two_mask[dosex_idx[0], dosey_idx[0]] = True
        # Get
        adjusted_ranks = (
            scipy.stats.rankdata(phase_dists_arr).reshape(phase_dists_arr.shape)
            - phase_dists_arr.shape[0]
        )
        in_prod_all_dist_avg_percentile = (
            adjusted_ranks[in_prod_mask].mean() / non_diag_mask.sum()
        )
        in_prod_top_two_dist_avg_percentile = (
            adjusted_ranks[in_prod_top_two_mask].mean() / non_diag_mask.sum()
        )
        in_product_all_dist_avg_percentile.append(in_prod_all_dist_avg_percentile)
        in_product_top_2_dist_avg_percentile.append(in_prod_top_two_dist_avg_percentile)
    metrics_dict[
        "in_product_all_dist_avg_percentile"
    ] = in_product_all_dist_avg_percentile
    metrics_dict[
        "in_product_top_2_dist_avg_percentile"
    ] = in_product_top_2_dist_avg_percentile

    metrics = pd.DataFrame(
        {"distance_type": distance_type, "phase": phases, **metrics_dict}
    ).assign(model_name=model_name)
    metrics.to_csv(table_out, index=False)


if __name__ == "__main__":
    compute_sciplex_metrics()
