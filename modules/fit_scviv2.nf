process fit_scviv2 {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    model_out = "${params.outputs.models}/${adata_name}.scviv2"
    """
    python3 ${params.bin.fit_scviv2} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out}
    """

    output:
    path adata_in
    path model_out
}