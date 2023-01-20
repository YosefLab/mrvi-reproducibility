process get_outputs_mrvi {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    adata_model_name = adata_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    distance_matrices_out = "${params.outputs.distance_matrices}/${adata_model_name}.distance_matrices.nc"
    """
    python3 ${params.bin.get_outputs_mrvi} \\
        --config_in ${config_in} \\
        --adata_in ${adata_in} \\
        --adata_out ${adata_in} \\
        --distance_matrices_out ${distance_matrices_out}
    """

    output:
    path adata_in, emit: adata
    path distance_matrices_out, emit: distance_matrices
}
