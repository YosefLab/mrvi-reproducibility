# %%
import argparse
import os

import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd
from utils import load_results

# Change to False if you want to run this script directly
RUN_WITH_PARSER = True
SHARED_THEME = (
    p9.theme_classic()
    + p9.theme(
        strip_background=p9.element_blank(),
    )
)
plt.rcParams["svg.fonttype"] = "none"

# %%
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
    os.makedirs(output_dir, exist_ok=True)

    pd.Series(results_paths).to_csv(os.path.join(output_dir, "path_to_intermediary_files.txt"), index=False)
else:
    output_dir = "PATH/TO/OUTPUT_DIR"
    results_paths = pd.read_csv(os.path.join(output_dir, "path_to_intermediary_files.txt")).squeeze().values.flatten()
# %%
def save_figures(fig, filename):
    fig.save(os.path.join(output_dir, filename + ".svg"))
    fig.save(os.path.join(output_dir, filename + ".png"), dpi=300)


all_results = load_results(results_paths)

# %%
if all_results["vendi_metrics"].size >= 1:
    fig = (
        p9.ggplot(all_results["vendi_metrics"], p9.aes(x="model_name", y="vendi_score", fill="model_name"))
        + p9.geom_boxplot()
        + p9.geom_point()
        + p9.coord_flip()
        + SHARED_THEME
        + p9.theme(
            legend_position="none",
        )
        + p9.labs(
            x="",
            y="VENDI score",
            fill="",
        )
    )
    save_figures(fig, "vendi_scores")

# %%
if all_results["scib_metrics"].size >= 1:
    # Assumption: we deal with u individually, and z with the rest
    where_u = all_results["scib_metrics"]["latent_key"].str.endswith("_u")
    u_plots = all_results["scib_metrics"][where_u]
    ufig = (
        p9.ggplot(u_plots, p9.aes(x="model_name", y="metric_value", fill="model_name"))
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("~metric_name", scales="free")
        + SHARED_THEME
        + p9.theme(
            subplots_adjust={"wspace": 0.5, "hspace": 0.5},
            legend_position="none",
        )
        + p9.labs(
            x="",
            y="",
        )
    )
    save_figures(ufig, "u_scib_scores")

    z_plots = all_results["scib_metrics"][~where_u]
    zfig = (
        p9.ggplot(z_plots, p9.aes(x="model_name", y="metric_value", fill="model_name"))
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("~metric_name", scales="free")
        + SHARED_THEME
        + p9.theme(
            subplots_adjust={"wspace": 0.5, "hspace": 0.5},
            legend_position="none",
        )
        + p9.labs(
            x="",
            y="",
        )
    )
    save_figures(zfig, "z_scib_scores")


# %%
# if all_results["rf_metrics"].size >= 1:
#     fig = (
#         p9.ggplot(all_results["rf_metrics"], p9.aes(x="model_name", y="rf_score", fill="model_name"))
#         + p9.geom_boxplot()
#         + p9.geom_point()
#         + p9.coord_flip()
#         + SHARED_THEME
#         + p9.theme(
#             # figure_size=(4 * INCH_TO_CM, 4 * INCH_TO_CM),
#             legend_position="none",
#         )
#         + p9.labs(
#             x="",
#             y="RF score",
#             fill="",
#         )
#     )
    # save_figures(fig, "rf_scores")
