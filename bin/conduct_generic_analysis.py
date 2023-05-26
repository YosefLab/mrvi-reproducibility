# %%
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
from utils import load_config, load_results, save_figures

# Change to False if you want to run this script directly
RUN_WITH_PARSER = False
SHARED_THEME = p9.theme_classic() + p9.theme(
    strip_background=p9.element_blank(),
)
plt.rcParams["svg.fonttype"] = "none"


def parser():
    """Parse paths to results files (used by nextflow)"""
    parser = argparse.ArgumentParser(description="Analyze results of symsim_new")
    parser.add_argument("--results_paths", "--list", nargs="+")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config_in", type=str)
    return parser.parse_args()


def extract_colors_from_config_file(config):
    """Infers which colors to use for plotting latent representations."""
    colors = config.get("latent_color_by", None)
    if colors is None:
        colors = [config["batch_key"], config["labels_key"]]
    return colors


# %%
if RUN_WITH_PARSER:
    args = parser()
    results_paths = args.results_paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config_path = args.config_in
    pd.Series(results_paths).to_csv(
        os.path.join(output_dir, "path_to_intermediary_files.txt"), index=False
    )

else:
    dataset_name = "sciplex_A549_simple_filtered_all_phases"
    output_dir = f"../results/sciplex_pipeline/figures/{dataset_name}"
    # good_filenames = (
    #     pd.read_csv(os.path.join(output_dir, "path_to_intermediary_files.txt"))
    #     .squeeze()
    #     .values.flatten()
    # )
    basedir = Path(output_dir).parent.parent.absolute()
    all_results_files = glob.glob(os.path.join(basedir, "**"), recursive=True)
    results_paths = [
        # x for x in all_results_files if os.path.basename(x) in good_filenames
        x
        for x in all_results_files
        if x.startswith(
            f"/home/justin/ghrepos/scvi-v2-reproducibility/bin/../results/sciplex_pipeline/data/{dataset_name}"
        )
        and x.endswith(".final.h5ad")
    ]
    # assert len(results_paths) == len(good_filenames)
    config_path = f"../conf/datasets/{dataset_name}.json"


config = load_config(config_path)
colors = extract_colors_from_config_file(config)
colors.append("pathway")
all_results = load_results(results_paths)

# %%
if all_results["vendi_metrics"].size >= 1:
    vendi_fig = (
        p9.ggplot(
            all_results["vendi_metrics"],
            p9.aes(x="model_name", y="vendi_score", fill="model_name"),
        )
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
    save_figures(vendi_fig, output_dir, "metrics/vendi_scores")

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
    save_figures(ufig, output_dir, "metrics/u_scib_scores")

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
    save_figures(zfig, output_dir, "metrics/z_scib_scores")


# %%
if all_results["rf_metrics"].size >= 1:
    rf_fig = (
        p9.ggplot(
            all_results["rf_metrics"],
            p9.aes(x="model_name", y="rf_dist", fill="model_name"),
        )
        + p9.geom_boxplot()
        + p9.geom_point()
        + p9.coord_flip()
        + SHARED_THEME
        + p9.theme(
            legend_position="none",
        )
        + p9.labs(
            x="",
            y="RF score",
            fill="",
        )
    )
    save_figures(rf_fig, output_dir, "metrics/rf_scores")


# %%
if all_results["representations"].size >= 1:
    mde_reps = all_results["representations"].query("representation_type == 'MDE'")
    if mde_reps.size >= 1:
        unique_reps = mde_reps.representation_name.unique()
        for rep in unique_reps:
            for color_by in colors:
                rep_plots = mde_reps.query(f"representation_name == '{rep}' and A549_deg_product_dose == 'True'")
                rep_fig = (
                    p9.ggplot(rep_plots, p9.aes(x="x", y="y", fill=color_by))
                    + p9.geom_point(stroke=0, size=0.5)
                    + SHARED_THEME
                    + p9.theme(
                        axis_text=p9.element_blank(),
                        axis_ticks=p9.element_blank(),
                    )
                    + p9.labs(
                        x="MDE1",
                        y="MDE2",
                        title=rep,
                    )
                )
                save_figures(
                    rep_fig,
                    output_dir,
                    f"representations/{rep}_{color_by}",
                    save_svg=False,
                )
                plt.clf()

# %%
