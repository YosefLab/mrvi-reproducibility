process get_latent_mrvi {
    input:
    path adata_in
    path model_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    method_name = model_in.getExtension()
    adata_out = "${params.outputs.data}/${adata_name}.${method_name}.h5ad"
    cell_representations_out = "${params.outputs.distance_matrices}/${adata_name}.${method_name}.cell_representations.nc"
    distance_matrices_out = "${params.outputs.distance_matrices}/${adata_name}.${method_name}.distance_matrices.nc"
    normalized_distance_matrices_out = "${params.outputs.distance_matrices}/${adata_name}.${method_name}.normalized_distance_matrices.nc"
    """
    python3 ${params.bin.get_latent_mrvi} \\
        --adata_in ${adata_in} \\
        --model_in ${model_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --cell_representations_out ${cell_representations_out} \\
        --distance_matrices_out ${distance_matrices_out} \\
        --normalized_distance_matrices_out ${normalized_distance_matrices_out} 
    """

    output:
    path adata_out, emit: adata
    path distance_matrices_out, emit: distance_matrices
    path normalized_distance_matrices_out, emit: normalized_distance_matrices
}
