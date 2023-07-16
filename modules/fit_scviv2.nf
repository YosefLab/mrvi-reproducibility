process fit_scviv2 {
    input:
    path adata_in
    val use_attention_noprior
    val use_attention_no_prior_mog
    val z30
    val z20_u5
    val z20_u10
    val z30_u5
    val z30_u10

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"

    if (use_attention_noprior) {
        method_name = "scviv2_attention_noprior"
    }
    else if (use_attention_no_prior_mog) {
        method_name = "scviv2_attention_no_prior_mog"
    }
    else if (z30) {
        method_name = "scviv2_z30"
    }
    else if (z20_u5) {
        method_name = "scviv2_z20_u5"
    }
    else if (z20_u10) {
        method_name = "scviv2_z20_u10"
    }
    else if (z30_u5) {
        method_name = "scviv2_z30_u5"
    }
    else if (z30_u10) {
        method_name = "scviv2_z30_u10"
    }
    else {
        method_name = "scviv2"
    }

    model_out = "${params.outputs.models}/${adata_name}.${method_name}"
    """
    python3 ${params.bin.fit_scviv2} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out} \\
        --use_attention_noprior ${use_attention_noprior} \\
        --use_attention_no_prior_mog ${use_attention_no_prior_mog} \\
        --z30 ${z30} \\
        --z20_u5 ${z20_u5} \\
        --z20_u10 ${z20_u10} \\
        --z30_u5 ${z30_u5} \\
        --z30_u10 ${z30_u10}
    """

    output:
    path adata_in
    path model_out
}
