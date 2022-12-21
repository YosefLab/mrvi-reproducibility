import json
import os
import pathlib
import pickle
from inspect import signature
from typing import Callable

import click
import pandas as pd

INCH_TO_CM = 1 / 2.54


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def wrap_kwargs(fn: Callable) -> Callable:
    """Wrap a function to accept keyword arguments from the command line."""
    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def save_pickle(obj, path):
    """Save a Python object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_results(results_paths):
    """Load and sort all results from a list of paths.

    Parameters
    ----------
    results_paths :
        List of paths to results files.
    """
    all_results = {
        "vendi_metrics": pd.DataFrame(),
        "scib_metrics": pd.DataFrame(),
        "rf_metrics": pd.DataFrame(),
        "losses_metrics": pd.DataFrame(),
        "umaps_metrics": pd.DataFrame(),
        "distances_metrics": pd.DataFrame(),
    }
    for file in results_paths:
        if file.endswith(".h5ad"):
            continue
        df = pd.read_csv(file)
        basename = os.path.basename(file)
        model_name = basename.split(".")[1]
        df.loc[:, "model_name"] = model_name
        if file.endswith("vendi.csv"):
            all_results["vendi_metrics"] = all_results["vendi_metrics"].append(df)
        elif file.endswith("scib.csv"):
            all_results["scib_metrics"] = all_results["scib_metrics"].append(df)
        elif file.endswith("rf.csv"):
            all_results["rf_metrics"] = all_results["rf_metrics"].append(df)
        elif file.endswith("losses.csv"):
            all_results["losses_metrics"] = all_results["losses_metrics"].append(df)
        elif file.endswith("umap.csv"):
            all_results["umaps_metrics"] = all_results["umaps_metrics"].append(df)
        elif file.endswith("distances.csv"):
            all_results["distances_metrics"] = all_results["distances_metrics"].append(
                df
            )
    return all_results
