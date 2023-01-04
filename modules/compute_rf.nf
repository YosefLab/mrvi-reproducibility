process compute_rf {
    input:
    path distance_matrices

    script:
    adata_model_name = distance_matrices.getBaseName()
    adata_name = distance_matrices.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    distance_matrices_gt = "${params.conf.distance_matrices}/${adata_name}.distance_matrices_gt.nc"
    table_out = "${params.outputs.metrics}/${adata_model_name}.rf.csv"

    """
    python3 ${params.bin.compute_rf} \\
        --distance_matrices ${adata_in} \\
        --distance_matrices_gt ${distance_matrices_gt} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """

    output:
    path table_out
}
