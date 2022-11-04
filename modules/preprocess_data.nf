process preprocess_data {
    conda "${params.env.preprocess_data}"

    input:
    path adata_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    adata_out = "${params.output.preprocess_data}/${adata_name}.h5ad"
    """
    python3 ${params.script.preprocess_data} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out}
    """

    output:
    path adata_out
}