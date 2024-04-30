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
    gt_deg_sim_in,
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
    gt_deg_sim_in:
        paths to ground truth differential expression similarity scores retrieved from bulk data
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

    try:
        inferred_mats = xr.open_dataarray(distance_matrices_in)
    except ValueError:
        inferred_mats = xr.open_dataset(distance_matrices_in)[celltype_key]
    inferred_mats = inferred_mats.rename("distance")
    dim_name = inferred_mats.dims[0]
    clusters = inferred_mats[dim_name].data

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
            new_sample_idx = [
                prod + "_10000" for prod in list(gt_cluster_labels_df.index)
            ]
            gt_cluster_labels_df.index = new_sample_idx
            # Filter on samples in the distance matrix
            gt_cluster_labels_df = gt_cluster_labels_df.loc[
                np.intersect1d(
                    inferred_mats.coords["sample_x"].data,
                    gt_cluster_labels_df.index.array,
                )
            ]
            break

    gt_deg_sim_df = None
    gt_deg_sim_in = gt_deg_sim_in.split(",")
    for gt_deg_sim_in_path in gt_deg_sim_in:
        gt_cell_line = os.path.basename(gt_deg_sim_in_path).split("_")[0]
        if gt_cell_line == cell_line:
            if determine_if_file_empty(gt_deg_sim_in_path):
                break

            gt_deg_sim_df = pd.read_csv(gt_deg_sim_in_path, index_col=0)
            # Assign them all at 10000 nM dose
            new_sample_idx = [prod + "_10000" for prod in list(gt_deg_sim_df.index)]
            gt_deg_sim_df.index = new_sample_idx
            # Filter on samples in the distance matrix
            gt_deg_sim_df = gt_deg_sim_df.loc[
                np.intersect1d(
                    inferred_mats.coords["sample_x"].data,
                    gt_deg_sim_df.index.array,
                )
            ]
            break

    if gt_cluster_labels_df is None:
        Path(table_out).touch()
    else:
        gt_silhouette_scores = []
        gt_correlation_scores = []
        gt_correlation_pvals = []

        for cluster in clusters:
            dist_inferred = (
                inferred_mats.loc[cluster]
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

            dist_inferred_deg_sim = (
                inferred_mats.loc[cluster]
                .sel(
                    sample_x=gt_deg_sim_df.index,
                    sample_y=gt_deg_sim_df.index,
                )
                .values
            )
            off_diag_idx = np.invert(np.eye(dist_inferred_deg_sim.shape[0], dtype=bool))
            res = scipy.stats.spearmanr(
                dist_inferred_deg_sim[off_diag_idx],
                gt_deg_sim_df.values[off_diag_idx],
                alternative="greater",
            )
            gt_correlation_scores.append(res.statistic)
            gt_correlation_pvals.append(res.pvalue)

        metrics_dict["gt_silhouette_score"] = gt_silhouette_scores
        metrics_dict["gt_correlation_score"] = gt_correlation_scores

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
    for cluster in clusters:
        cluster_dists = inferred_mats.loc[cluster]
        cluster_dists_arr = cluster_dists.data
        non_diag_mask = (
            np.ones(shape=cluster_dists_arr.shape)
            - np.identity(cluster_dists_arr.shape[0])
        ).astype(bool)
        in_prod_mask = np.zeros(shape=cluster_dists_arr.shape, dtype=bool)
        in_prod_top_two_mask = np.zeros(shape=cluster_dists_arr.shape, dtype=bool)
        for product_name in all_products:
            for dosex in all_doses:
                for dosey in all_doses:
                    if dosex == dosey:
                        continue
                    dosex_idx = np.where(
                        cluster_dists.sample_x.data == f"{product_name}_{dosex}"
                    )[0]
                    if len(dosex_idx) == 0:
                        continue
                    dosey_idx = np.where(
                        cluster_dists.sample_y.data == f"{product_name}_{dosey}"
                    )[0]
                    if len(dosey_idx) == 0:
                        continue
                    in_prod_mask[dosex_idx[0], dosey_idx[0]] = True

                    if dosex in top_two_doses and dosey in top_two_doses:
                        in_prod_top_two_mask[dosex_idx[0], dosey_idx[0]] = True
        # Get
        adjusted_ranks = (
            scipy.stats.rankdata(cluster_dists_arr).reshape(cluster_dists_arr.shape)
            - cluster_dists_arr.shape[0]
        )
        in_prod_all_dist_avg_percentile = (
            adjusted_ranks[in_prod_mask].mean() / non_diag_mask.sum()
        )
        in_prod_top_two_dist_avg_percentile = (
            adjusted_ranks[in_prod_top_two_mask].mean() / non_diag_mask.sum()
        )
        in_product_all_dist_avg_percentile.append(in_prod_all_dist_avg_percentile)
        in_product_top_2_dist_avg_percentile.append(in_prod_top_two_dist_avg_percentile)
    metrics_dict["in_product_all_dist_avg_percentile"] = (
        in_product_all_dist_avg_percentile
    )
    metrics_dict["in_product_top_2_dist_avg_percentile"] = (
        in_product_top_2_dist_avg_percentile
    )

    metrics = pd.DataFrame(
        {"distance_type": distance_type, dim_name: clusters, **metrics_dict}
    ).assign(model_name=model_name)
    metrics.to_csv(table_out, index=False)


if __name__ == "__main__":
    compute_sciplex_metrics()
