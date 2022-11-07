process fit_mrvi {
    conda "${params.env.run_mrvi}"

    input:
    path adata_in

    output:
    path adata_out
    path model_out

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    adata_out = "${params.output.run_mrvi_setup_data}/${adata_name}.h5ad"
    model_out = "${params.output.run_mrvi_model}/mrvi_${adata_name}.pt"
    """
    python3 ${params.script.fit_mrvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out} \\
        --model_out ${model_out}
    """
}
