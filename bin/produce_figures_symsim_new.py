# %%
import argparse
import os

import plotnine as p9
from utils import INCH_TO_CM, load_results


def parser():
    """Parse paths to results files (used by nextflow)"""
    parser = argparse.ArgumentParser(description="Analyze results of symsim_new")
    parser.add_argument("--results_paths", "--list", nargs="+")
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


# %%
args = parser()
results_paths = args.results_paths
output_dir = args.output_dir
# %% # Uncomment this cell to run the script interactively
output_dir = "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/figures"
results_paths = [
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/data/symsim_new.mrvi.h5ad",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/symsim_new.scviv2.h5ad",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/data/symsim_new.composition_scvi.h5ad",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/data/symsim_new.composition_pca.h5ad",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.scviv2.distance_matrices.rf.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.mrvi.distance_matrices.rf.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.composition_scvi.scib.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.composition_pca.scib.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.mrvi.scib.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.scviv2.scib.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.composition_pca.vendi.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.composition_scvi.vendi.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.mrvi.vendi.csv",
    "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline/metrics/symsim_new.scviv2.vendi.csv",
]
# %%
os.makedirs(output_dir, exist_ok=True)

# %%
all_results = load_results(results_paths)

# %%

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
plot_df
# %%
# Distance matrix comparison
