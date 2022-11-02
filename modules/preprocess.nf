#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

input_channel = Channel.fromPath("../data/*.h5ad")
output_channel = Channel.fromPath("../data/preprocessed/")

process preprocess_data { 
    input:
    file adata from input_channel

    output:
    file "${adata.baseName}_preprocessed.h5ad" into output_channel

    """
    preprocess_data.py $adata
    """
}
