process get_outputs_scviv2 {
    input:
    path adata_in
    path cell_distance_matrices_in
    path cell_normalized_distance_matrices_in

    script:
    adata_name = adata_in.getSimpleName()
    adata_model_name = adata_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    distance_matrices_out = "${params.outputs.distance_matrices}/${adata_model_name}.distance_matrices.nc"
    normalized_distance_matrices_out = "${params.outputs.distance_matrices}/${adata_model_name}.normalized_distance_matrices.nc"
    """
    python3 ${params.bin.get_outputs_scviv2} \\
        --config_in ${config_in} \\
        --adata_in ${adata_in} \\
        --adata_out ${adata_in} \\
        --cell_distance_matrices_in ${cell_distance_matrices_in} \\
        --cell_normalized_distance_matrices_in ${cell_normalized_distance_matrices_in} \\
        --distance_matrices_out ${distance_matrices_out} \\
        --normalized_distance_matrices_out ${normalized_distance_matrices_out}
    """

    output:
    path adata_in, emit: adata
    path distance_matrices_out, emit: distance_matrices
    path normalized_distance_matrices_out, emit: normalized_distance_matrices
}
