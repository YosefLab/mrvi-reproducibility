process fit_mrvi {
    input:
    path adata_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    model_out = "${params.output.run_mrvi_model}/${adata_name}"
    """
    python3 ${params.script.fit_mrvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out}
    """

    output:
    path adata_in
    path model_out
}
