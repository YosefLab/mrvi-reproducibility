import string
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from utils import load_config, make_parents, wrap_kwargs


def _make_categorical(adata: AnnData, obs_key: str) -> AnnData:
    """Make an obs column categorical in-place."""
    adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    return adata


def _subsample(adata: AnnData, **kwargs) -> AnnData:
    """Subsample cells in-place."""
    kwargs.update({"copy": False})
    sc.pp.subsample(adata, **kwargs)
    return adata


def _filter_genes(adata: AnnData, **kwargs) -> AnnData:
    """Filter genes in-place."""
    kwargs.update({"inplace": True})
    sc.pp.filter_genes(adata, **kwargs)
    return adata


def _hvg(adata: AnnData, **kwargs) -> AnnData:
    """Select highly-variable genes in-place."""
    kwargs.update({"subset": True})
    sc.pp.highly_variable_genes(adata, **kwargs)
    return adata


def _assign_symsim_donors(
    adata: AnnData,
    *,
    batch_key: Optional[str] = None,
    sample_key: Optional[str] = None,
    n_donors: int = 32,
    seed: int = 1, 
) -> AnnData:
    np.random.seed(seed)

    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]
    batches = adata.obs[batch_key].cat.categories.tolist()
    n_batch = len(batches)

    meta_combos = list(product([0, 1], repeat=n_meta))
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

    adata.obs[sample_key] = donor_assignment
    donor_meta = adata.obs[sample_key].str.extractall(
        r"donor_meta\(([0-1]), ([0-1]), ([0-1])\)_batch[0-9]_[a-z]"
    )
    for match_idx, meta_key in enumerate(meta_keys):
        adata.obs[f"donor_{meta_key}"] = donor_meta[match_idx].astype(int).tolist()
    return adata 


def _construct_tree_semisynth(adata: AnnData, depth_tree: int = 4, seed: int = 0) -> AnnData:
    """
    Modifies gene expression in two cell subpopulations according to a controlled
    donor tree structure.
    """
    # construct donors
    n_donors = int(2**depth_tree)
    np.random.seed(seed)
    random_donors = np.random.randint(0, n_donors, adata.n_obs)
    n_modules = sum([int(2**k) for k in range(1, depth_tree + 1)])

    # ct_key = snakemake.config[dataset_name]["keyMapping"]["cellTypeKey"]
    # cts = adata.obs.groupby(ct_key).size().sort_values(ascending=False)[:2]
    # ct1, ct2 = cts.index.values
    ct_key = "leiden"
    ct1, ct2 = "0", "1"

    # construct donor trees
    leaves_id = np.array(
        [format(i, "0{}b".format(depth_tree)) for i in range(n_donors)]
    )  # ids of leaves

    all_node_ids = []
    for dep in range(1, depth_tree + 1):
        node_ids = [format(i, "0{}b".format(dep)) for i in range(2**dep)]
        all_node_ids += node_ids  # ids of all nodes in the tree

    def perturb_gene_exp(ct, X_perturbed, all_node_ids, leaves_id):
        leaves_id1 = leaves_id.copy()
        np.random.shuffle(leaves_id1)
        genes = np.arange(adata.n_vars)
        np.random.shuffle(genes)
        gene_modules = np.array_split(genes, n_modules)
        gene_modules = {
            node_id: gene_modules[i] for i, node_id in enumerate(all_node_ids)
        }
        gene_modules = {
            node_id: np.isin(np.arange(adata.n_vars), gene_modules[node_id])
            for node_id in all_node_ids
        }
        # convert to one hots to make life easier for saving

        # modifying gene expression
        # e.g., 001 has perturbed modules 0, 00, and 001
        subpop = adata.obs.loc[:, ct_key].values == ct
        print("perturbing {}".format(ct))
        for donor_id in range(n_donors):
            selected_pop = subpop & (random_donors == donor_id)
            leaf_id = leaves_id1[donor_id]
            perturbed_mod_ids = [leaf_id[:i] for i in range(1, depth_tree + 1)]
            perturbed_modules = np.zeros(adata.n_vars, dtype=bool)
            for id in perturbed_mod_ids:
                perturbed_modules = perturbed_modules | gene_modules[id]

            Xmat = X_perturbed[selected_pop].copy()
            print(
                "Perturbing {} genes in {} cells".format(
                    perturbed_modules.sum(), selected_pop.sum()
                )
            )
            print(
                "Non-zero values in the relevant subpopulation and modules:   ",
                (Xmat[:, perturbed_modules] != 0).sum(),
            )
            Xmat[:, perturbed_modules] = Xmat[:, perturbed_modules] * 2

            X_perturbed[selected_pop] = Xmat
        return X_perturbed, gene_modules, leaves_id1

    X_pert = adata.X.copy()
    X_pert, gene_mod1, leaves1 = perturb_gene_exp(ct1, X_pert, all_node_ids, leaves_id)
    X_pert, gene_mod2, leaves2 = perturb_gene_exp(ct2, X_pert, all_node_ids, leaves_id)

    gene_modules1 = pd.DataFrame(gene_mod1)
    gene_modules2 = pd.DataFrame(gene_mod2)
    donor_metadata = pd.DataFrame(
        dict(
            donor_id=np.arange(n_donors),
            tree_id1=leaves1,
            affected_ct1=ct1,
            tree_id2=leaves2,
            affected_ct2=ct2,
        )
    )

    meta_id1 = pd.DataFrame([list(x) for x in donor_metadata.tree_id1.values]).astype(
        int
    )
    n_features1 = meta_id1.shape[1]
    new_cols1 = ["tree_id1_{}".format(i) for i in range(n_features1)]
    donor_metadata.loc[:, new_cols1] = meta_id1.values
    meta_id2 = pd.DataFrame([list(x) for x in donor_metadata.tree_id2.values]).astype(
        int
    )
    n_features2 = meta_id2.shape[1]
    new_cols2 = ["tree_id2_{}".format(i) for i in range(n_features2)]
    donor_metadata.loc[:, new_cols2] = meta_id2.values

    donor_metadata.loc[:, new_cols1 + new_cols2] = donor_metadata.loc[
        :, new_cols1 + new_cols2
    ].astype("category")

    adata.obs.loc[:, "batch"] = random_donors
    adata.obs.loc[:, "Site"] = 1
    original_index = adata.obs.index.copy()
    adata.obs = adata.obs.merge(
        donor_metadata, left_on="batch", right_on="donor_id", how="left"
    )
    adata.obs.index = original_index
    adata.uns["gene_modules1"] = gene_modules1
    adata.uns["gene_modules2"] = gene_modules2
    adata.uns["donor_metadata"] = donor_metadata
    adata = sc.AnnData(X_pert, obs=adata.obs, var=adata.var, uns=adata.uns)
    return adata


def _process_snrna(adata: AnnData, seed: int = 0) -> AnnData:
    """Process single-nuclei RNA-seq data."""
    # We first remove samples processed with specific protocols that
    # are not used in other samples
    select_cell = ~(adata.obs["sample"].isin(["C41_CST", "C41_NST"]))
    adata = adata[select_cell].copy()

    # We also artificially subsample
    # one of the samples into two samples
    # as a negative control
    subplit_sample = "C72_RESEQ"
    mask_selected_lib = (adata.obs["sample"] == subplit_sample).values
    np.random.seed(seed)
    mask_split = (
        np.random.randint(0, 2, size=mask_selected_lib.shape[0]).astype(bool)
        * mask_selected_lib
    )
    libraries = adata.obs["library_uuid"].astype(str)
    libraries.loc[mask_split] = libraries.loc[mask_split] + "_split"
    assert libraries.unique().shape[0] == 9
    adata.obs.loc[:, "library_uuid"] = libraries.astype("category")
    return adata


@wrap_kwargs
def preprocess(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
) -> AnnData:
    """
    Preprocess an input AnnData object and saves it to a new file.

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

    batch_key = config.get("batch_key", None)
    sample_key = config.get("sample_key", None)
    subsample_kwargs = config.get("subsample_kwargs", None)
    filter_genes_kwargs = config.get("filter_genes_kwargs", None)
    hvg_kwargs = config.get("hvg_kwargs", None)
    synthetic = config.get("synthetic", False)
    semisynthetic = config.get("semisynthetic", False)
    snrna = config.get("snrna", False)


    adata = _make_categorical(adata, batch_key)
    adata = _make_categorical(adata, sample_key)
    if isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.tocsr()
    
    if subsample_kwargs is not None:
        adata = _subsample(adata, **subsample_kwargs)
    if filter_genes_kwargs is not None:
        adata = _filter_genes(adata, **filter_genes_kwargs)
    if hvg_kwargs is not None:
        adata = _hvg(adata, **hvg_kwargs)

    if synthetic:
        adata = _assign_symsim_donors(
            adata, batch_key=batch_key, sample_key=sample_key
        )
    elif semisynthetic:
        adata = _construct_tree_semisynth(adata)
    elif snrna:
        adata = _process_snrna(adata)

    
    make_parents(adata_out)
    adata.write(filename=adata_out)
    return adata


if __name__ == "__main__":
    preprocess()
