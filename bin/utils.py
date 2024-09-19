import gc
import json
import os
import pathlib
import pickle
import time
import warnings
from inspect import signature
from pathlib import Path
from typing import Callable

import click
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from remote_pdb import RemotePdb


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


def compute_n_degs(adata, group_key, ref_group):
    """Utility function to compute the number of DEGs per group compared to a fixed reference."""
    warnings.filterwarnings("ignore")
    adata.layers["log1p"] = sc.pp.log1p(adata, copy=True).X
    adata.uns["log1p"] = {"base": None}

    sc.tl.rank_genes_groups(
        adata,
        group_key,
        layer="log1p",
        reference=ref_group,
        method="t-test",
        corr_method="benjamini-hochberg",
    )

    n_deg_dict = {}
    for group in adata.obs[group_key].cat.categories:
        if group == ref_group:
            continue
        sig_idxs = adata.uns["rank_genes_groups"]["pvals_adj"][group] <= 0.05
        suff_lfc_idxs = (
            np.abs(adata.uns["rank_genes_groups"]["logfoldchanges"][group]) >= 0.5
        )
        n_deg_dict[group] = np.sum(sig_idxs & suff_lfc_idxs)
    return pd.Series(n_deg_dict)


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


def perform_gsea(
    genes: list,
    gene_sets: list = None,
    organism: str = "human",
    n_trials_max: int = 20,
    plot: bool = False,
    plot_sortby: str = "Adjusted P-value",
    plot_ntop: int = 5,
    use_server: bool = True,
):
    """
    Perform GSEA using Enrichr.

    Parameters
    ----------
    genes :
        List of gene symbols to perform GSEA on.
    gene_sets :
        List of gene sets to use for GSEA.
        An exhaustive list of gene sets can be found using `gp.get_library_name()`
    organism :
        Considered organism.
    n_trials_max :
        Maximum number of trials to perform GSEA. Consider increasing if GSEA fails.
    plot :
        Whether to plot the results.
    plot_sortby :
        Key to sort the results by, only used if `plot` is True.
    plot_ntop :
        Number of top results to plot, only used if `plot` is True.
    use_server :
        Whether to use the web server.

    Returns
    -------
    A pandas DataFrame containing the GSEA results.
    If `plot` is True, also returns a plotnine figure.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError(
            "GSEApy is not installed. Please install it via pip or conda."
        )

    if gene_sets is None:
        gene_sets = [
            "MSigDB_Hallmark_2020",
            "WikiPathway_2021_Human",
            "KEGG_2021_Human",
            "Reactome_2022",
            "GO_Biological_Process_2023",
            "GO_Cellular_Component_2023",
            "GO_Molecular_Function_2023",
        ]
        if not use_server:
            gene_set_dicts = [
                gp.parser.download_library(gene_set_name, "human")
                for gene_set_name in gene_sets
            ]
            gene_set_names = gene_sets
            gene_sets = gene_set_dicts

    if use_server:
        is_done = False
        for _ in tqdm(range(n_trials_max)):
            if is_done:
                break

            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=gene_sets,
                    organism=organism,
                    outdir=None,
                    verbose=False,
                )
                is_done = True
            except:
                print("GSEA failed; retrying...")
                time.sleep(1)
                continue
        if not is_done:
            raise ValueError(
                "GSEA failed; please consider increasing `n_trials_max` or try running enrichr manually."
            )
    else:
        enr = gp.enrich(
            gene_list=genes,
            gene_sets=gene_sets,
            outdir=None,
            verbose=False,
        )
    enr_results = enr.results.copy().sort_values("Adjusted P-value")
    enr_results.loc[:, "Significance score"] = -np.log10(
        enr_results.loc[:, "Adjusted P-value"]
    )
    if not use_server:
        gene_set_mapping = {
            f"gs_ind_{i}": gene_set_name
            for i, gene_set_name in enumerate(gene_set_names)
        }
        enr_results.loc[:, "Gene_set"] = enr_results.loc[:, "Gene_set"].map(
            gene_set_mapping
        )
    if not plot:
        return enr_results

    try:
        import plotnine as p9
    except ImportError:
        raise ImportError(
            "Plotnine is not installed. Please install it via pip or conda."
        )
    plot_df = (
        enr_results.loc[lambda x: x["Adjusted P-value"] < 0.1, :]
        .sort_values(plot_sortby)
        .head(plot_ntop)
        .sort_values("Gene_set")
    )
    fig = (
        p9.ggplot(plot_df, p9.aes(x="Term", y="Significance score", fill="Gene_set"))
        + p9.geom_col()
        + p9.scale_x_discrete(limits=plot_df.Term.tolist())
        + p9.labs(
            x="",
        )
        + p9.theme_classic()
        + p9.theme(
            strip_background=p9.element_blank(),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            axis_text=p9.element_text(family="sans-serif", size=5),
            axis_title=p9.element_text(family="sans-serif", size=6),
        )
    )
    return enr_results, fig
