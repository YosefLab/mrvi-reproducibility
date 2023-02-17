# %%
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import plotnine as p9
import seaborn as sns
import xarray as xr
from utils import INCH_TO_CM, load_results

# Change to False if you want to run this script directly
RUN_WITH_PARSER = True


def parser():
    """Parse paths to results files (used by nextflow)"""
    parser = argparse.ArgumentParser(description="Analyze results of symsim_new")
    parser.add_argument("--results_paths", "--list", nargs="+")
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


# %%
if RUN_WITH_PARSER:
    args = parser()
    results_paths = args.results_paths
    output_dir = args.output_dir
else:
    output_dir = "../results/symsim_pipeline/figures"
    results_paths = set(
        glob.glob("../results/symsim_pipeline/*/*.csv")
        + glob.glob("../results/symsim_pipeline/*/*.h5ad")
    )
results_path_root = os.path.join(output_dir, "..")
os.makedirs(output_dir, exist_ok=True)
# %%
all_results = load_results(results_paths)
# %%
# Latent visualization

# %%
# Metrics comparison
plot_df = all_results["vendi_metrics"]
fig = (
    p9.ggplot(plot_df, p9.aes(x="model_name", y="vendi_score", fill="model_name"))
    + p9.geom_boxplot()
    + p9.theme_classic()
    + p9.coord_flip()
    + p9.theme(
        legend_position="none",
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + p9.labs(x="", y="VENDI score")
)
fig.save(os.path.join(output_dir, "symsim_new.vendi_score.svg"))
fig
# %%
plot_df = all_results["scib_metrics"]
# COMPLETE
plot_df

# %%
plot_df = all_results["rf_metrics"].query(
    "cell_type == 'CT1:1'"
)  # only select active cell-type
fig = (
    p9.ggplot(plot_df, p9.aes(x="model_name", y="rf_dist", fill="model_name"))
    + p9.geom_bar(stat="identity")
    + p9.theme_classic()
    + p9.coord_flip()
    + p9.theme(
        legend_position="none",
        figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
    )
    + p9.labs(x="", y="RF distance")
)
fig.save(os.path.join(output_dir, "symsim_new.rf.svg"))
fig
# %%
plot_df
# %%
model_name = "scviv2"
# model_name = "scviv2_nonlinear"
# Distance matrix comparison
scviv2_dists_path = os.path.join(
    results_path_root, f"distance_matrices/symsim_new.{model_name}.distance_matrices.nc"
)
scviv2_normalized_dists_path = os.path.join(
    results_path_root,
    f"distance_matrices/symsim_new.{model_name}.normalized_distance_matrices.nc",
)
dists = xr.open_dataset(scviv2_dists_path).celltype
normalized_dists = xr.open_dataset(scviv2_normalized_dists_path).celltype
# %%
normalized_vmax = np.percentile(normalized_dists.data, 95)
vmax = np.percentile(dists.data, 95)
for ct in normalized_dists.celltype_name.values:
    sns.heatmap(
        normalized_dists.sel(celltype_name=ct),
        cmap="YlGnBu",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=normalized_vmax,
    )
    plt.savefig(
        os.path.join(
            output_dir, f"symsim_new.normalized_distance_matrix.{model_name}.{ct}.svg"
        )
    )
    plt.clf()

    sns.heatmap(
        dists.sel(celltype_name=ct),
        cmap="YlGnBu",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=vmax,
    )
    plt.savefig(
        os.path.join(output_dir, f"symsim_new.distance_matrix.{model_name}.{ct}.svg")
    )
    plt.clf()
# %%
# Labeled histogram of distances vs normalized distances
plt.hist(dists.data.flatten(), bins=100, alpha=0.5, label="distances")
plt.hist(
    normalized_dists.data.flatten(), bins=100, alpha=0.5, label="normalized distances"
)
plt.legend()
plt.savefig(
    os.path.join(
        output_dir, f"symsim_new.normalized_distance_matrix_hist.{model_name}.svg"
    )
)
plt.clf()
# %%
binwidth = 0.1
bins = np.arange(0, vmax + binwidth, binwidth)
for ct in normalized_dists.celltype_name.values:
    plt.hist(
        dists.sel(celltype_name=ct).data.flatten(),
        bins=bins,
        alpha=0.5,
        label="distances",
    )
    plt.hist(
        normalized_dists.sel(celltype_name=ct).data.flatten(),
        bins=bins,
        alpha=0.5,
        label="normalized distances",
    )
    plt.title(ct)
    plt.legend()
    plt.xlim(-0.5, vmax + 0.5)
    plt.savefig(
        os.path.join(
            output_dir, f"symsim_new.compare_distance_matrix_hist.{ct}.{model_name}.svg"
        )
    )
    plt.clf()

# %%
