process fit_scviv2 {
    maxForks 1

    input:
    path adata_in
    val use_nonlinear

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    if (use_nonlinear) {
        method_name = "scviv2_nonlinear"
    } else {
        method_name = "scviv2"
    }
    model_out = "${params.outputs.models}/${adata_name}.${method_name}"
    """
    python3 ${params.bin.fit_scviv2} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out} \\
        --use_nonlinear ${use_nonlinear}
    """

    output:
    path adata_in
    path model_out
}
