process compute_rf {
    input:
    tuple val(adata_name), path(adata_in)

    script:
    adata_model_name = adata_in.getBaseName()
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    table_out = "${params.outputs.metrics}/${adata_model_name}.rf.csv"

    """
    python3 ${params.bin.compute_rf} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """

    output:
    tuple val(adata_name), path(table_out)
}
