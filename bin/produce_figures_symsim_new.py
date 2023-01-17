# %%
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
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
    results_paths = glob.glob("../results/symsim_pipeline/*/*.csv") + glob.glob(
        "../results/symsim_pipeline/*/*.h5ad"
    )
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
# Distance matrix comparison
dists = xr.open_dataarray(
    "/home/justin/ghrepos/scvi-v2-reproducibility/results/symsim_pipeline/distance_matrices/symsim_new.scviv2.distance_matrices.nc"
)
normalized_dists = xr.open_dataarray(
    "/home/justin/ghrepos/scvi-v2-reproducibility/results/symsim_pipeline/distance_matrices/symsim_new.scviv2.normalized_distance_matrices.nc"
)

# %%
vmax = np.percentile(normalized_dists.values, 95)
for ct in normalized_dists.celltype:
    sns.heatmap(
        normalized_dists.sel(celltype=ct),
        cmap="YlGnBu",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=vmax,
    )
    plt.show()
    plt.clf()
# %%
# Labeled histogram of distances vs normalized distances
plt.hist(dists.values.flatten(), bins=100, alpha=0.5, label="distances")
plt.hist(
    normalized_dists.values.flatten(), bins=100, alpha=0.5, label="normalized distances"
)
plt.legend()
# %%
binwidth = 0.1
bins = np.arange(0, vmax + binwidth, binwidth)
for ct in normalized_dists.celltype.values:
    plt.hist(
        dists.sel(celltype=ct).values.flatten(), bins=bins, alpha=0.5, label="distances"
    )
    plt.hist(
        normalized_dists.sel(celltype=ct).values.flatten(),
        bins=bins,
        alpha=0.5,
        label="normalized distances",
    )
    plt.title(ct)
    plt.legend()
    plt.xlim(-0.5, vmax + 0.5)
    plt.show()
    plt.clf()
