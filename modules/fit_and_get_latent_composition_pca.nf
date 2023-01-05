process fit_and_get_latent_composition_pca {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.composition_pca.h5ad"
    distance_matrices_out = "${params.outputs.distance_matrices}/${adata_name}.composition_pca.distance_matrices.nc"
    """
    python3 ${params.bin.fit_and_get_latent_composition_pca} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --distance_matrices_out ${distance_matrices_out}
    """

    output:
    path adata_out, emit: adata
    path distance_matrices_out, emit: distance_matrices
}
