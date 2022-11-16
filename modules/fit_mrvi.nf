process fit_mrvi {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    model_out = "${params.outputs.models}/${adata_name}.mrvi"
    """
    python3 ${params.bin.fit_mrvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out}
    """

    output:
    path adata_in
    path model_out
}
