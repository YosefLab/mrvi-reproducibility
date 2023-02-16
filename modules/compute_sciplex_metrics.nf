process compute_sciplex_metrics {
    input:
    path adata_in
    path distance_matrices_in
    path gt_matrices_in

    script:
    adata_name = adata_in.getSimpleName()
    adata_model_name = adata_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    table_out = "${params.outputs.metrics}/${adata_model_name}.sciplex_metrics.csv"
    """
    python3 ${params.bin.compute_sciplex_metrics} \\
        --adata_in ${adata_in} \\
        --distance_matrices_in ${distance_matrices_in} \\
        --gt_matrices_in ${gt_matrices_in} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """

    output:
    path table_out
}
