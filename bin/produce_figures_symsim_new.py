# %%
import argparse

import pandas as pd


def parser():
    """Parse paths to results files (used by nextflow)"""
    parser = argparse.ArgumentParser(description="Analyze results of symsim_new")
    parser.add_argument("--results_paths", "--list", nargs="+")
    return parser.parse_args()


# %%
args = parser()
results_paths = args.results_paths
# %%
# Uncomment this cell to run the script interactively
# results_dir = "/home/pierre/scvi-v2-reproducibility/results/symsim_pipeline"

# metrics_dir = os.path.join(results_dir, "metrics")
# metric_files = glob.glob(os.path.join(metrics_dir, "*.csv"))

# results_paths = metric_files
# %%
# Right now, we save model outputs that either correspond to
# 1. metrics, for all, or a subset of, the models
# 2. latent representations for some models
# 3. distance matrices for some models

vendi_metrics = pd.DataFrame()
scib_metrics = pd.DataFrame()
rf_metrics = pd.DataFrame()
losses_metrics = pd.DataFrame()
umaps_metrics = pd.DataFrame()
distances_metrics = pd.DataFrame()
all_results = {}

# %%

# %%
# Latent visualization

# %%
# Metrics comparison

# %%
# Distance matrix comparison
