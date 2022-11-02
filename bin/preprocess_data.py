import argparse
from typing import List

import scanpy as sc
from anndata import AnnData


def _highly_variable_genes(adata: AnnData, ngenes = 2000):
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, flavor="seurat_v3", subset=True)

def main(adata_path: str, config_path: str) -> None:
    adata = sc.read_h5ad(adata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset.")

    parser.add_argument(
        "adata_path",
        type=str,
    )
    parser.add_argument(
        "config_path",
        type=str,
    )
    args = parser.parse_args()
    main(adata_path=args.adata_path, config_path=args.config_path)

