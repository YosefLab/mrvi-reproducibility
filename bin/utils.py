import json
import os
import pathlib
import pickle
from inspect import signature
from pathlib import Path
from typing import Callable

import click
import pandas as pd
from remote_pdb import RemotePdb

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


def determine_if_file_empty(file_path):
    """Determine if file is empty."""
    return Path(file_path).stat().st_size == 0


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
        if determine_if_file_empty(file):
            continue
        if file.endswith(".h5ad") or file.endswith(".nc"):
            continue
        df = pd.read_csv(file)
        basename = os.path.basename(file)
        model_name = basename.split(".")[1]
        df.loc[:, "model_name"] = model_name
        if file.endswith(".distance_matrices.vendi.csv"):
            all_results["vendi_metrics"] = all_results["vendi_metrics"].append(df)
        elif file.endswith(".distance_matrices.scib.csv"):
            all_results["scib_metrics"] = all_results["scib_metrics"].append(df)
        elif file.endswith(".distance_matrices.rf.csv"):
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


def set_breakpoint(host: str = "127.0.0.1", port: int = 4444):
    """Set a breakpoint for debugging.

    The interactive debugger can be accessed by running locally
    `telnet 127.0.0.1 4444` in a separate terminal.
    To move up and down the callstack, type in `up` or `down`.
    To exit, use `exit` or ctrl + c.
    """
    RemotePdb(host, port).set_trace()
