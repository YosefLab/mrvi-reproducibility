process vendi {
    input:
    path distance_matrix_in

    script:
    adata_name = distance_matrix_in.getSimpleName()
    distance_matrix_name = distance_matrix_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    table_out = "${params.outputs.metrics}/${distance_matrix_name}.vendi.csv"
    """
    python3 ${params.bin.vendi} \\
        --distance_matrix_in ${distance_matrix_in} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """

    output:
    path table_out
}
