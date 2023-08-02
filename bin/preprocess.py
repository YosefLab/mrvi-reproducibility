import itertools
import string
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
import xarray as xr
from sklearn.decomposition import PCA, SparsePCA
from anndata import AnnData
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from tree_utils import linkage_to_ete
from utils import load_config, make_parents, wrap_kwargs, set_breakpoint


@wrap_kwargs
def preprocess(
    *,
    adata_in: str,
    config_in: str,
    adata_out: str,
    distance_matrices_out=None,
) -> AnnData:
    """
    Preprocess an input AnnData object and saves it to a new file.

    Performs the following steps:

    1. Highly variable genes selection
    2. Dataset-specific preprocessing

    Parameters
    ----------
    adata_in
        Path to the input AnnData object.
    config_in
        Path to the dataset configuration file.
    adata_out
        Path to write the preprocessed AnnData object.
    distance_matrices_out
        Path to write the distance matrices, if applicable.
    """
    config = load_config(config_in)
    adata = sc.read(adata_in)
    hvg_kwargs = config.get("hvg_kwargs", None)
    min_obs_per_sample = config.get("min_obs_per_sample", None)
    requires_celltype_subset = "subset_celltypes" in config

    cell_type_key = config.get("labels_key")
    adata.obs.index.name = None  # ensuring that index is not named, which could cause problem when resetting index
    if cell_type_key not in adata.obs.keys():
        adata.obs.loc[:, cell_type_key] = "0"
        adata.obs.loc[:, cell_type_key] = adata.obs.loc[:, cell_type_key].astype(
            "category"
        )
    if requires_celltype_subset:
        adata = _subset_celltypes(adata, config)
    if min_obs_per_sample is not None:
        adata = _subset_samples(adata, config)
    if hvg_kwargs is not None:
        adata = _hvg(adata, **hvg_kwargs)
    adata, distance_matrices = _run_dataset_specific_preprocessing(
        adata, adata_in, config
    )
    make_parents(adata_out)
    adata.write(filename=adata_out)
    make_parents(distance_matrices_out)
    if distance_matrices is not None:
        distance_matrices.to_netcdf(distance_matrices_out)
    else:
        Path(distance_matrices_out).touch()
    return adata


def _hvg(adata: AnnData, **kwargs) -> AnnData:
    """Select highly-variable genes in-place."""
    kwargs.update({"subset": True})
    sc.pp.highly_variable_genes(adata, **kwargs)
    return adata


def _run_dataset_specific_preprocessing(
    adata: AnnData, adata_in: str, config: dict
) -> AnnData:
    distance_matrices = None
    if adata_in == "symsim_new.h5ad":
        adata, distance_matrices = _assign_symsim_donors(adata, config)
    if adata_in == "scvi_pbmcs.h5ad":
        adata = _construct_tree_semisynth(
            adata,
            config,
            depth_tree=4,
        )
    if adata_in == "nucleus.h5ad":
        adata = _process_snrna(adata, config)
    if adata_in == "pbmcs68k.h5ad":
        adata = _process_semisynth2(
            adata,
            config,
        )
    if adata_in == "pbmcs68k_for_subsample.h5ad":
        adata = _process_semisynth2(
            adata,
            config,
            subsample=True,
        )
    if adata_in in ["haniffa.h5ad", "haniffasubsample.h5ad", "haniffa2.h5ad"]:
        adata = _process_haniffa(adata, config)
    return adata, distance_matrices


def _assign_symsim_donors(adata, config):
    np.random.seed(1)
    batch_key = config["batch_key"]
    sample_key = config["sample_key"]
    n_donors = 32
    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
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

    adata.obs[sample_key] = donor_assignment

    donor_meta = adata.obs[sample_key].str.extractall(
        r"donor_meta\(([0-1]), ([0-1]), ([0-1])\)_batch[0-9]_[a-z]"
    )

    # construct GT trees and distance matrices
    meta_corr = [0, 0.5, 0.9]
    donor_combos = list(itertools.product(*[[0, 1] for _ in range(len(meta_corr))]))
    # E[||x - y||_2^2]
    meta_dist = 2 - 2 * np.array(meta_corr)
    dist_mtx = np.zeros((len(donor_combos), len(donor_combos)))
    for i in range(len(donor_combos)):
        for j in range(i):
            donor_i, donor_j = donor_combos[i], donor_combos[j]
            donor_diff = abs(np.array(donor_i) - np.array(donor_j))
            dist_mtx[i, j] = dist_mtx[j, i] = np.sum(donor_diff * meta_dist)
    dist_mtx = np.sqrt(dist_mtx)  # E[||x - y||_2]

    # build GT donor metadata
    gt_donor_meta = (
        adata.obs.loc[lambda x: ~x[sample_key].duplicated(keep="first")]
        .set_index(sample_key)
        .sort_index()
        .assign(donor_archetype=lambda x: x.index.str[11:18])
    )
    adata.uns["sample_metadata"] = gt_donor_meta

    # Map archetype (e.g., (0, 1, 0))
    # to index in dist_mtx
    donor_archetype_to_idx = {}
    for idx, donor in enumerate(donor_combos):
        donor_archetype_to_idx[donor] = idx

    def _get_donor_archetype(donor):
        """Convert string pattern to tuple of ints"""
        split = donor.split(",")
        return tuple(int(s) for s in split)

    n_donors_final = gt_donor_meta.shape[0]
    gt_dist_mtx = np.zeros((n_donors_final, n_donors_final))
    for idx_a, vals_a in gt_donor_meta.reset_index().iterrows():
        for idx_b, vals_b in gt_donor_meta.reset_index().iterrows():
            arch_a = _get_donor_archetype(vals_a["donor_archetype"])
            arch_b = _get_donor_archetype(vals_b["donor_archetype"])

            index_a = donor_archetype_to_idx[arch_a]
            index_b = donor_archetype_to_idx[arch_b]
            gt_dist_mtx[idx_a, idx_b] = gt_dist_mtx[idx_b, idx_a] = dist_mtx[
                index_a, index_b
            ]
    gt_control_dist_mtx = 1 - np.eye(len(gt_donor_meta.index))
    all_distances = np.concatenate(
        [gt_dist_mtx[None], gt_control_dist_mtx[None]], axis=0
    )  # shape (2, n_donors, n_donors)
    all_distances = xr.DataArray(
        all_distances,
        dims=["celltype_name", "sample_x", "sample_y"],
        coords={
            "celltype_name": ["CT1:1", "CT2:1"],
            "sample_x": gt_donor_meta.index.values,
            "sample_y": gt_donor_meta.index.values,
        },
        name="distance_gt",
    )

    for match_idx, meta_key in enumerate(meta_keys):
        adata.obs[f"donor_{meta_key}"] = donor_meta[match_idx].astype(int).tolist()
    return adata, all_distances


def _construct_tree_semisynth(adata, config, depth_tree=3):
    """Modifies gene expression in two cell subpopulations according to a tree structure."""
    # construct donors
    n_donors = int(2**depth_tree)
    np.random.seed(0)
    batch_key = config["batch_key"]
    sample_key = config["sample_key"]

    random_donors = np.random.randint(0, n_donors, adata.n_obs)
    n_modules = sum([int(2**k) for k in range(1, depth_tree + 1)])

    ct_key = "leiden"
    ct1, ct2 = "0", "1"

    # construct donor trees
    leaves_id = np.array(
        [format(i, f"0{depth_tree}b") for i in range(n_donors)]
    )  # ids of leaves

    all_node_ids = []
    for dep in range(1, depth_tree + 1):
        node_ids = [format(i, f"0{dep}b") for i in range(2**dep)]
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
        print(f"perturbing {ct}")
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
    donor_metadata = (
        pd.DataFrame(
            {
                "donor_id": np.arange(n_donors),
                "tree_id1": leaves1,
                "affected_ct1": ct1,
                "tree_id2": leaves2,
                "affected_ct2": ct2,
            }
        )
        .assign(donor_name=lambda x: "donor_" + x.donor_id.astype(str))
        .astype(
            {
                "tree_id1": "category",
                "affected_ct1": "category",
                "tree_id2": "category",
                "affected_ct2": "category",
                "donor_name": "category",
            }
        )
    )

    meta_id1 = pd.DataFrame([list(x) for x in donor_metadata.tree_id1.values]).astype(
        int
    )
    n_features1 = meta_id1.shape[1]
    new_cols1 = [f"tree_id1_{i}" for i in range(n_features1)]
    donor_metadata.loc[:, new_cols1] = meta_id1.values
    meta_id2 = pd.DataFrame([list(x) for x in donor_metadata.tree_id2.values]).astype(
        int
    )
    n_features2 = meta_id2.shape[1]
    new_cols2 = [f"tree_id2_{i}" for i in range(n_features2)]
    donor_metadata.loc[:, new_cols2] = meta_id2.values

    donor_metadata.loc[:, new_cols1 + new_cols2] = donor_metadata.loc[
        :, new_cols1 + new_cols2
    ].astype("category")

    adata.obs.loc[:, sample_key] = random_donors
    adata.obs.loc[:, sample_key] = "donor_" + adata.obs[sample_key].astype(str)
    adata.obs.loc[:, sample_key] = adata.obs.loc[:, sample_key].astype("category")
    adata.obs.loc[:, batch_key] = 1
    original_index = adata.obs.index.copy()
    adata.obs = adata.obs.merge(
        donor_metadata, left_on=sample_key, right_on="donor_id", how="left"
    )
    adata.obs.index = original_index
    adata.uns["gene_modules1"] = gene_modules1
    adata.uns["gene_modules2"] = gene_modules2
    adata.uns["donor_metadata"] = donor_metadata
    adata = sc.AnnData(X_pert, obs=adata.obs, var=adata.var, uns=adata.uns)
    return adata


def _process_snrna(adata, config):
    sample_key = config["sample_key"]
    # We first remove samples processed with specific protocols that
    # are not used in other samples
    select_cell = ~(adata.obs["sample"].isin(["C41_CST", "C41_NST"]))
    adata = adata[select_cell].copy()

    # We also artificially subsample
    # one of the samples into two samples
    # as a negative control
    subplit_sample = "C72_RESEQ"
    mask_selected_lib = (adata.obs["sample"] == subplit_sample).values
    np.random.seed(0)
    mask_split = (
        np.random.randint(0, 2, size=mask_selected_lib.shape[0]).astype(bool)
        * mask_selected_lib
    )
    libraries = adata.obs[sample_key].astype(str)
    libraries.loc[mask_split] = libraries.loc[mask_split] + "_split"
    assert libraries.unique().shape[0] == 9
    adata.obs.loc[:, sample_key] = libraries.astype("category")
    return adata


def _process_semisynth2(
    adata,
    config,
    subsample=False,
):
    """Construct a semi-synthetic dataset from PBMCs.

    This function first clusters the data. In one of the clusters, it then performs hierarchical clustering to identify subclusters.
    Each of these subclusters is then used to construct
    replicated pseudo-donors.
    In the other clusters, pseudo-donors are assigned randomly.

    This function assumes that the data config file contains a `dataset_config` key, with the following atttributes:
    - `resolution`: base resolution to construct clusters.
    - `n_subclusters`: number of subclusters to construct within the positive cluster.
    - `n_replicates_per_subcluster`: number of replicates to construct for each subcluster.
    - `selected_cluster`: cluster to select as the positive cluster.

    If `subsample` is True, the function will subsample a fraction of the samples in a second cluster
    to test for model behavior in parts of the manifold with low data density.
    """
    semisynth_config = config["dataset_config"]
    resolution = semisynth_config["resolution"]
    n_subclusters = semisynth_config["n_subclusters"]
    n_replicates_per_subcluster = semisynth_config["n_replicates_per_subcluster"]
    selected_cluster = semisynth_config["selected_cluster"]
    n_genes_for_subclustering = semisynth_config["n_genes_for_subclustering"]
    if subsample:
        selected_subsample_cluster = semisynth_config["selected_subsample_cluster"]
        subsample_rates = semisynth_config["subsample_rates"]

    # use SCVI to obtain latent space
    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log)
    sc.pp.log1p(adata_log)
    sc.pp.pca(adata_log, n_comps=50)
    sc.pp.neighbors(adata_log, use_rep="X_pca")
    sc.tl.leiden(adata_log, resolution=resolution, key_added="leiden")
    adata.obs["leiden"] = adata_log.obs["leiden"]
    if n_genes_for_subclustering is not None:
        selected_genes_for_subclustering = np.random.choice(
            adata.var_names, n_genes_for_subclustering, replace=False
        )
        adata_log_ = adata_log[:, selected_genes_for_subclustering].copy()
        adata.var["is_gene_for_subclustering"] = False
        adata.var.loc[
            selected_genes_for_subclustering, "is_gene_for_subclustering"
        ] = True
        sc.pp.pca(adata_log_, n_comps=50)
        adata.obsm["X_rep_subclustering"] = adata_log_.obsm["X_pca"]
    else:
        adata.obsm["X_rep_subclustering"] = adata_log.obsm["X_pca"]

    cluster_to_sizes = adata.obs.leiden.value_counts().sort_values(ascending=False)
    positive_cluster = cluster_to_sizes.index[selected_cluster]
    sample_assignments = pd.DataFrame()
    for unique_cluster in tqdm(adata.obs["leiden"].unique()):
        adata_ = adata[adata.obs["leiden"] == unique_cluster].copy()
        latent_reps = adata_.obsm["X_rep_subclustering"]
        if unique_cluster == positive_cluster:
            res_ = construct_sample_stratifications_from_subcelltypes(
                latent_reps,
                n_subclusters,
                n_replicates_per_subcluster,
            )
            adata.uns[f"cluster{positive_cluster}_tree_gt"] = res_["tree_gt"].write()
            adata.uns[f"cluster{positive_cluster}_tree_linkage"] = res_["tree_linkage"]
            adata.uns[f"cluster{positive_cluster}_tree_gt_clusters"] = res_[
                "tree_gt_clusters"
            ].write()
            adata.uns[f"cluster{positive_cluster}_tree_linkage_clusters"] = res_[
                "tree_linkage_clusters"
            ]
            adata.uns[f"cluster{positive_cluster}_tree_linkage_leaders"] = res_[
                "tree_linkage_leaders"
            ]
            sample_assignments = pd.concat(
                [
                    sample_assignments,
                    res_["cluster_info"]
                    .loc[:, ["sample_assignments"]]
                    .assign(cell_index=adata_.obs_names),
                ]
            )

            sample_assignment_mapping_ = (
                res_["cluster_info"]
                .drop_duplicates()
                .rename(columns={"sample_assignments": "sample_assignment"})
            )
        else:
            sample_names = 1 + np.arange(n_subclusters * n_replicates_per_subcluster)
            sample_assignments_ = np.random.choice(
                sample_names, size=adata_.shape[0], replace=True
            )
            sample_assignments_ = (
                pd.Series(sample_assignments_)
                .to_frame("sample_assignments")
                .assign(cell_index=adata_.obs_names)
            )
            sample_assignments = pd.concat([sample_assignments, sample_assignments_])
    sample_assignments = sample_assignments.set_index("cell_index").loc[
        adata.obs_names
    ]["sample_assignments"]
    adata.obs = (
        adata.obs.assign(
            sample_assignment=sample_assignments.values, cell_index=adata.obs_names
        )
        .merge(sample_assignment_mapping_, on="sample_assignment", how="left")
        .assign(
            has_sample_stratification=lambda x: x.leiden == positive_cluster,
            Site="1",
            subcluster_assignment=lambda x: x.apply(
                lambda y: y.sample_group if y.has_sample_stratification else "NA",
                axis=1,
            ),
        )
        .astype(
            {
                "sample_assignment": str,
                "subcluster_assignment": str,
            }
        )
        .set_index("cell_index")
    )
    adata.obs.index.name = None
    if subsample:
        sample_assignment_mapping = sample_assignment_mapping_[
            ["sample_assignment", "sample_group"]
        ]
        sample_assignment_mapping["subcluster_assignment"] = (
            sample_assignment_mapping["sample_group"].astype(str).astype("category")
        )
        sample_assignment_mapping[
            "sample_assignment_int"
        ] = sample_assignment_mapping.sample_assignment.astype(int)
        sample_assignment_mapping["rank"] = (
            sample_assignment_mapping.groupby("subcluster_assignment")[
                "sample_assignment_int"
            ]
            .rank(method="dense", ascending=True)
            .astype(int)
        )

        subsampled_adatas = [adata[adata.obs.leiden != str(selected_subsample_cluster)]]
        subsample_info_df = pd.DataFrame()
        for rank, subsample_rate in enumerate(subsample_rates, 1):
            samples_to_subsample = sample_assignment_mapping[
                sample_assignment_mapping["rank"] == rank
            ]["sample_assignment"].to_list()

            subsample_info_df = subsample_info_df.append(
                pd.DataFrame(
                    {
                        "rank": rank,
                        "subsample_rate": subsample_rate,
                        "sample": samples_to_subsample,
                    }
                )
            )
            for sample in samples_to_subsample:
                subsample_adata = adata[
                    (adata.obs.sample_assignment == str(sample))
                    & (adata.obs["leiden"] == str(selected_subsample_cluster))
                ]
                subsample_adata = subsample_adata[
                    np.random.choice(
                        subsample_adata.shape[0],
                        int(subsample_adata.shape[0] * subsample_rate),
                        replace=False,
                    )
                ]
                subsampled_adatas.append(subsample_adata)

        res = sc.concat(subsampled_adatas)
        subsample_info_df = subsample_info_df.astype({"sample": str}).set_index("sample")
        cell_to_sample = res.obs["sample_assignment"].values
        res.obs.loc[:, f"subsample_rate_in_leiden{selected_subsample_cluster}"] = subsample_info_df.loc[cell_to_sample, "subsample_rate"].values
        res.obs.loc[:, f"rank_in_leiden{selected_subsample_cluster}"] = subsample_info_df.loc[cell_to_sample, "rank"].values
        res.obs.loc[:, "sample_metadata2"] = res.obs[f"subsample_rate_in_leiden{selected_subsample_cluster}"] <= 0.8
        res = sc.AnnData(
            X=res.X,
            obs=res.obs,
            obsm=res.obsm,
            var=adata.var,
            uns=adata.uns,
        )
        return res
    return adata


def construct_sample_stratifications_from_subcelltypes(
    latent_reps, n_subclusters, n_replicates_per_subcluster, linkage_method="ward"
):
    """Construct semisynthetic dataset"""
    dmat = pairwise_distances(latent_reps)
    dmat = squareform(dmat, checks=False)
    Z = sch.linkage(dmat, method=linkage_method)
    subclusters = sch.fcluster(Z, n_subclusters, criterion="maxclust")
    metadata1 = sch.fcluster(Z, 2, criterion="maxclust")

    # get names of internal nodes associated with each cluster
    L, M = sch.leaders(Z, subclusters)
    cluster_mapper = pd.DataFrame({"nodeid": L, "clusterid": M})

    tree = linkage_to_ete(Z)
    # trim tree to clusters
    for leader in L:
        node = tree.search_nodes(name=str(leader))[0]
        children = node.get_children()
        for child in children:
            child.detach()

    # rename leaves to cluster ids
    renamed_tree = tree.copy()
    for leaf in renamed_tree.get_leaves():
        clusterid = cluster_mapper[cluster_mapper["nodeid"] == int(leaf.name)][
            "clusterid"
        ].values[0]
        leaf.name = "cluster" + str(clusterid)

    # add leaves for replicates
    sampleid_to_clusterid = []
    for clusterid in renamed_tree.get_leaf_names():
        sampleid_to_clusterid += [clusterid] * n_replicates_per_subcluster
    sampleid_to_clusterid = (
        pd.Series(sampleid_to_clusterid)
        .to_frame("clusterid")
        .reset_index()
        .rename(columns={"index": "sampleid"})
        .assign(sampleid=lambda x: 1 + x.sampleid)
    )
    clusterid_to_sampleids = {}
    for clusterid in sampleid_to_clusterid["clusterid"].unique():
        associated_sampleids = sampleid_to_clusterid[
            sampleid_to_clusterid["clusterid"] == clusterid
        ]["sampleid"].values
        clusterid_to_sampleids[clusterid] = associated_sampleids

    tree_with_replicates = renamed_tree.copy()
    for leaf in tree_with_replicates.get_leaves():
        leaf_name = leaf.name
        sampleids = sampleid_to_clusterid[
            sampleid_to_clusterid["clusterid"] == leaf_name
        ]["sampleid"].values
        for sampleid in sampleids:
            leaf.add_child(name=f"t{sampleid}")

    # construct synthetic sample stratification
    sample_assignments = []
    for assigned_cluster in subclusters:
        clustername = "cluster" + str(assigned_cluster)
        assigned_sample = np.random.choice(clusterid_to_sampleids[clustername])
        sample_assignments.append(assigned_sample)
    cluster_info = pd.DataFrame(
        {
            "sample_assignments": sample_assignments,
            "sample_group": subclusters,
            "sample_metadata": metadata1,
        }
    )
    return {
        "tree_gt": tree_with_replicates,
        "tree_gt_clusters": tree,
        "cluster_info": cluster_info,
        "tree_linkage": Z,
        "tree_linkage_clusters": L,
        "tree_linkage_leaders": M,
    }


def _process_haniffa(adata, config_in):
    adata = adata[:, ~adata.var_names.str.startswith("AB")]
    adata = adata[adata.obs.Status.isin(["Covid", "Healthy"])]
    adata.obs.loc[:, "age_int"] = adata.obs.Age_interval.apply(
        lambda x: x.split(",")[0][1:]
    ).astype(int)
    if "subset_site" in config_in:
        adata = adata[adata.obs.Site == config_in["subset_site"]].copy()
    return adata.copy()


def _subset_celltypes(adata, config_in):
    celltype_key = config_in["labels_key"]
    celltypes_to_subset = config_in["subset_celltypes"]
    good_cells = adata.obs[celltype_key].isin(celltypes_to_subset)
    adata = adata[good_cells].copy()
    return adata


def _subset_samples(adata, config_in):
    min_obs_per_sample = config_in["min_obs_per_sample"]
    n_obs_per_sample = adata.obs.groupby(config_in["sample_key"]).size()
    selected_samples = n_obs_per_sample[n_obs_per_sample >= min_obs_per_sample].index
    adata = adata[adata.obs[config_in["sample_key"]].isin(selected_samples)].copy()
    adata.obs.loc[:, config_in["sample_key"]] = pd.Categorical(
        adata.obs.loc[:, config_in["sample_key"]]
    )
    return adata


if __name__ == "__main__":
    preprocess()
