process compute_vendi {
    conda "${params.env.compute_metrics}"
    publishDir "${params.publish}"

    input:
    path adata_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    table_out = "${params.output.compute_metrics_vendi}/${adata_name}.csv"
    """
    python3 ${params.script.compute_vendi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --table_out ${table_out}
    """
    
    output:
    path table_out
}
