process compute_sciplex_metrics {
    input:
    path distance_matrices_in
    path gt_clusters_in

    script:
    adata_name = distance_matrices_in.getSimpleName()
    model_name = distance_matrices_in.getBaseName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    table_out = "${params.outputs.metrics}/${model_name}.sciplex_metrics.csv"
    """
    python3 ${params.bin.compute_sciplex_metrics} \\
        --distance_matrices_in ${distance_matrices_in} \\
        --gt_clusters_in ${gt_clusters_in.join(',')} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """

    output:
    path table_out
}
