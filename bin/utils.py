import json
import os
import pathlib
import pickle
import gc
from inspect import signature
from pathlib import Path
from typing import Callable

import click
import pandas as pd
import scanpy as sc
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

    def append_representations(
        adata, uns_latent_key, representation_name, dataset_name
    ):
        """
        Retrieve latent representations from some adata.

        Parameters
        ----------
        adata :
            Anndata object.
        uns_latent_key :
            Key in adata.uns containing the list of cell representations to extract.
        representation_name :
            Name of the representation type.
        dataset_name :
            Name of the dataset.
        """
        if uns_latent_key in adata.uns.keys():
            obs = pd.DataFrame()
            for latent_key in adata.uns[uns_latent_key]:
                obs_ = adata.obs.copy().reset_index()
                obs_.loc[:, ["x", "y"]] = adata.obsm[latent_key]
                obs_.loc[:, "representation_name"] = latent_key
                obs_.loc[:, "representation_type"] = representation_name
                obs_.loc[:, "dataset_name"] = dataset_name
                obs = obs.append(obs_)
            return obs
        return None

    all_results = {
        "vendi_metrics": [],
        "scib_metrics": [],
        "rf_metrics": [],
        "losses_metrics": [],
        "umaps_metrics": [],
        "distances_metrics": [],
        "representations": [],
        "sciplex_metrics": [],
    }
    for file in results_paths:
        if determine_if_file_empty(file):
            continue
        if file.endswith(".nc"):
            continue
        basename = os.path.basename(file)
        dataset_name = basename.split(".")[0]
        model_name = basename.split(".")[1]
        if file.endswith("csv"):
            df = pd.read_csv(file)
            df.loc[:, "dataset_name"] = dataset_name
            df.loc[:, "model_name"] = model_name
            if file.endswith(".distance_matrices.vendi.csv"):
                all_results["vendi_metrics"].append(df)
            elif file.endswith(".scib.csv"):
                all_results["scib_metrics"].append(df)
            elif file.endswith(".distance_matrices.rf.csv"):
                all_results["rf_metrics"].append(df)
            elif file.endswith("losses.csv"):
                all_results["losses_metrics"].append(df)
            elif file.endswith("umap.csv"):
                all_results["umaps_metrics"].append(df)
            elif file.endswith("distances.csv"):
                all_results["distances_metrics"].append(df)
            elif file.endswith("sciplex_metrics.csv"):
                all_results["sciplex_metrics"].append(df)
            pass
        elif file.endswith(".h5ad"):
            adata = sc.read_h5ad(file, backed="r")
            for rep_type in ["MDE", "PCA", "UMAP"]:
                uns_key = f"latent_{rep_type.lower()}_keys"
                if uns_key in adata.uns.keys():
                    all_results["representations"].append(
                        append_representations(adata, uns_key, rep_type, dataset_name)
                    )
            del adata
            gc.collect()  # Adata uns creates weak reference that must be manually gc.
    for key in all_results.keys():
        if len(all_results[key]) > 0:
            all_results[key] = pd.concat(all_results[key])
        else:
            all_results[key] = pd.DataFrame()
    return all_results


def set_breakpoint(host: str = "127.0.0.1", port: int = 4444):
    """Set a breakpoint for debugging.

    The interactive debugger can be accessed by running locally
    `telnet 127.0.0.1 4444` in a separate terminal.
    To move up and down the callstack, type in `up` or `down`.
    To exit, use `exit` or ctrl + c.
    """
    RemotePdb(host, port).set_trace()


def save_figures(fig, output_dir, filename, save_svg=True):
    """Save a figure to disk.

    Parameters
    ----------
    fig :
        Plotnine figure.
    output_dir :
        Directory to save the figure to.
    filename :
        Filename to save the figure to, without extension.
    save_svg :
        Whether to save the figure as an SVG file in addition to a PNG file.
    """
    basename = os.path.join(output_dir, filename)
    basedir = os.path.dirname(basename)
    os.makedirs(basedir, exist_ok=True)
    fig.save(basename + ".png", dpi=300)
    if save_svg:
        fig.save(basename + ".svg")
