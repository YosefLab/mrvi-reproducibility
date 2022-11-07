process compute_scib_metrics {
    conda "${params.env.compute_metrics}"

    input:
    path adata_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    table_out = "${params.output.compute_metrics}/scib/${adata_name}.csv"
    """
    python3 ${params.script.compute_scib_metrics} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """
}
