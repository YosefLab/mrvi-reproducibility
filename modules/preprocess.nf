process preprocess {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.preprocessed.h5ad"
    distance_matrices_out = "${params.outputs.distance_matrices}/${adata_name}.distance_matrices_gt.nc"
    """
    python3 ${params.bin.preprocess} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --distance_matrices_out ${distance_matrices_out}
    """

    output:
    tuple path(adata_out), path(distance_matrices_out)
}
