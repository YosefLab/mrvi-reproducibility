process get_latent_scviv2 {
    input:
    path adata_in
    path model_in

    script:
    adata_name = adata_in.getSimpleName()
    adata_model_name = adata_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.scviv2.h5ad"
    cell_representations_out = "${params.outputs.distance_matrices}/${adata_model_name}.cell_representations.nc"
    cell_distance_matrices_out = "${params.outputs.distance_matrices}/${adata_model_name}.cell_distance_matrices.nc"
    cell_normalized_distance_matrices_out = "${params.outputs.distance_matrices}/${adata_model_name}.cell_normalized_distance_matrices.nc"
    """
    python3 ${params.bin.get_latent_scviv2} \\
        --adata_in ${adata_in} \\
        --model_in ${model_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --cell_representations_out ${cell_representations_out} \\
        --cell_distance_matrices_out ${cell_distance_matrices_out} \\
        --cell_normalized_distance_matrices_out ${cell_normalized_distance_matrices_out} 
    """

    output:
    path adata_out
    path cell_distance_matrices_out
    path cell_normalized_distance_matrices_out
}
