import argparse
import itertools
import string
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def _create_joint_obs_key(adata: AnnData, keys: List[str]) -> str:
    """Combine multiple obs column into a single joint obs key in-place."""
    joint_key_name = "_".join(keys)
    adata.obs[joint_key_name] = (
        adata.obs[keys].astype(str).apply(lambda x: "_".join(x), axis=1)
    )
    return joint_key_name


def _make_categorical(adata: AnnData, key: str) -> None:
    """Make an obs column categorical in-place."""
    adata.obs[key] = adata.obs[key].astype("category")


def _create_obs_mapper(adata: AnnData, dataset_name: str,) -> None:
    """Create a mapper dict to store obs keys."""
    dataset_config = None
    
    donor_key = dataset_config["donor_key"]
    if isinstance(donor_key, list):
        donor_key = _create_joint_obs_key(adata, donor_key)
    _make_categorical(adata, donor_key)

    cell_type_key = dataset_config["cell_type_key"]
    _make_categorical(adata, cell_type_key)

    nuisance_keys = dataset_config["nuisance_keys"]
    for key in nuisance_keys:
        _make_categorical(adata, key)
    
    adata.uns["mapper"] = {
        "donor_key": donor_key,
        "cell_type_key": cell_type_key,
        "categorical_nuisance_keys": nuisance_keys,
    }


def _process_dataset(dataset_name: str, config: str) -> None:
    input_h5ad = None
    adata = sc.read_h5ad(input_h5ad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for MrVI.")

    parser.add_argument(
        "dataset_name",
        type=str,
    )
    parser.add_argument(
        "config",
        type=str,
    )
    args = parser.parse_args()
