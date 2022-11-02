import argparse

import scanpy as sc
from scvi_v2 import MrVI


def main(adata_path, save_model_path, save_adata_path, batch_key, sample_key):
    adata = sc.read(adata_path)
    MrVI.setup_anndata(
        adata, categorical_nuisance_keys=[batch_key], sample_key=sample_key
    )
    m = MrVI(adata)
    m.train()
    m.save(save_model_path, save_anndata=False)
    adata.write_h5ad(save_adata_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Integrate scRNA-seq data with scVI")
    parser.add_argument(
        "adata_path",
        type=str,
        help="Input AnnData path",
    )
    parser.add_argument(
        "--save_model_path",
        dest="save_model_path",
        type=str,
        help="Path to save the trained MrVI model",
    )
    parser.add_argument(
        "--save_adata_path",
        dest="save_adata_path",
        type=str,
        help="Path to save the AnnData used to train MrVI",
    )
    parser.add_argument(
        "--batch_key",
        dest="batch_key",
        type=str,
        help="obs key where batch metadata is stored in the AnnData. Used as the sole categorical nuisance key",
    )
    parser.add_argument(
        "--sample_key",
        dest="sample_key",
        type=str,
        help="obs key where sample metadata is stored in the AnnData",
    )
    args = parser.parse_args()
    main(
        adata_path=args.adata_path,
        model_path=args.model_path,
        save_adata_path=args.save_adata_path,
        batch_key=args.batch_key,
        sample_key=args.sample_key,
    )
