import string
from itertools import product

import numpy as np
import scanpy as sc
from anndata import AnnData

from utils import load_config, make_parents, wrap_kwargs


def _filter_genes(adata: AnnData, **kwargs) -> AnnData:
    """Filter genes in-place."""
    kwargs.update({"inplace": True}})
    sc.pp.filter_genes(adata, **kwargs)
    return adata


def _subsample(adata: AnnData, **kwargs) -> AnnData:
    """Subsample an AnnData object in-place."""
    kwargs.update({"copy": False}})
    sc.pp.subsample(adata, **kwargs)
    return adata


def _hvg(adata: AnnData, **kwargs) -> AnnData:
    """Select highly-variable genes in-place."""
    kwargs.update({"subset": True})
    sc.pp.highly_variable_genes(adata, **kwargs)
    return adata


def assign_symsim_donors(adata):
    np.random.seed(1)
    dataset_config = snakemake.config["synthetic"]["keyMapping"]
    donor_key = dataset_config["donorKey"]
    batch_key = dataset_config["nuisanceKeys"][0]

    n_donors = 32
    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]
    make_categorical(adata, batch_key)
    batches = adata.obs[batch_key].cat.categories.tolist()
    n_batch = len(batches)

    meta_combos = list(itertools.product([0, 1], repeat=n_meta))
    donors_per_meta_batch_combo = n_donors // len(meta_combos) // n_batch

    # Assign donors uniformly at random for cells with matching metadata.
    donor_assignment = np.empty(adata.n_obs, dtype=object)
    for batch in batches:
        batch_donors = []
        for meta_combo in meta_combos:
            match_cats = [f"CT{meta_combo[i]+1}:1" for i in range(n_meta)]
            eligible_cell_idxs = (
                (
                    np.all(
                        adata.obs[meta_keys].values == match_cats,
                        axis=1,
                    )
                    & (adata.obs[batch_key] == batch)
                )
                .to_numpy()
                .nonzero()[0]
            )
            meta_donors = [
                f"donor_meta{meta_combo}_batch{batch}_{ch}"
                for ch in string.ascii_lowercase[:donors_per_meta_batch_combo]
            ]
            donor_assignment[eligible_cell_idxs] = np.random.choice(
                meta_donors, replace=True, size=len(eligible_cell_idxs)
            )
            batch_donors += meta_donors

    adata.obs[donor_key] = donor_assignment

    donor_meta = adata.obs[donor_key].str.extractall(
        r"donor_meta\(([0-1]), ([0-1]), ([0-1])\)_batch[0-9]_[a-z]"
    )
    for match_idx, meta_key in enumerate(meta_keys):
        adata.obs[f"donor_{meta_key}"] = donor_meta[match_idx].astype(int).tolist()


def assign_symsim_donors(
    adata: AnnData, batch_key: str, sample_key: str, seed: int = 1
) -> AnnData:
    np.random.seed(seed)
    n_donors = 32
    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]




@wrap_kwargs
def preprocess_data(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
) -> AnnData:
    """
    Preprocess an input AnnData object and saves it to a new file.

    Performs the following steps:

    1. Highly variable genes selection

    TODO: Add more preprocessing steps as necessary

    Parameters
    ----------
    adata_in
        Path to the input AnnData object.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the preprocessed AnnData object.
    """
    config = load_config(config_in)
    adata = sc.read(adata_in)
    sample_key = config.get("sample_key", None)
    batch_key = config.get("batch_key", None)
    filter_genes_kwargs = config.get("filter_genes_kwargs", None)
    subsample_kwargs = config.get("subsample_kwargs", None)
    hvg_kwargs = config.get("hvg_kwargs", None)
    symsim_donors = config.get("symsim_donors", False)
    tree_semisynth = config.get("tree_semisynth", False)
    snrna = config.get("snrna", False)
    
    if filter_genes_kwargs is not None:
        adata = _filter_genes(adata, **filter_genes_kwargs)
    if subsample_kwargs is not None:
        adata = _subsample(adata, **subsample_kwargs)
    if hvg_kwargs is not None:
        adata = _hvg(adata, **hvg_kwargs)

    if symsim_donors:
        pass
    elif tree_semisynth:
        pass
    elif snrna:
        pass

    make_parents(adata_out)
    adata.write(filename=adata_out)
    return adata


if __name__ == "__main__":
    preprocess_data()
