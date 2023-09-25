process fit_scviv2 {
    input:
    path adata_in
    val use_mlp
    val use_mlp_smallu
    val use_attention
    val use_attention_smallu
    val use_attention_noprior
    val use_attention_no_prior_mog
    val use_attention_mog
    val use_attention_no_prior_mog_large

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"

    if (use_mlp) {
        method_name = "scviv2_mlp"
    }
    else if (use_mlp_smallu) {
        method_name = "scviv2_mlp_smallu"
    }
    else if (use_attention) {
        method_name = "scviv2_attention"
    }
    else if (use_attention_smallu) {
        method_name = "scviv2_attention_smallu"
    }
    else if (use_attention_noprior) {
        method_name = "scviv2_attention_noprior"
    }
    else if (use_attention_no_prior_mog) {
        method_name = "scviv2_attention_no_prior_mog"
    }
    else if (use_attention_mog) {
        method_name = "scviv2_attention_mog"
    }
    else if (use_attention_no_prior_mog_large) {
        method_name = "scviv2_attention_no_prior_mog_large"
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
        --use_mlp ${use_mlp} \\
        --use_mlp_smallu ${use_mlp_smallu} \\
        --use_attention ${use_attention} \\
        --use_attention_smallu ${use_attention_smallu} \\
        --use_attention_noprior ${use_attention_noprior} \\
        --use_attention_no_prior_mog ${use_attention_no_prior_mog} \\
        --use_attention_mog ${use_attention_mog} \\
        --use_attention_no_prior_mog_large ${use_attention_no_prior_mog_large}
    """

    output:
    path adata_in
    path model_out
}
