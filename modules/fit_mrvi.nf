process fit_mrvi {
    input:
    path adata_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    adata_out = "${params.output.run_mrvi_setup_data}/${adata_name}.h5ad"
    model_out = "${params.output.run_mrvi_model}/${adata_name}"
    """
    python3 ${params.script.fit_mrvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --model_out ${model_out}
    """

    output:
    path adata_out
    path model_out
}
