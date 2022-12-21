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
os.makedirs(output_dir, exist_ok=True)
# %% # Uncomment this cell to run the script interactively
# results_paths = [
#     # COMPLETE
# ]

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
