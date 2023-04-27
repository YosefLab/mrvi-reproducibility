process fit_scviv2 {
    input:
    path adata_in
    val use_mlp
    val use_mlp_smallu
    val use_attention
    val use_attention_smallu
    val use_double_attention_ld
    val use_double_attention_hd

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
    else if (use_double_attention_ld) {
        method_name = "scviv2_double_attention_ld"
    }
    else if (use_double_attention_hd) {
        method_name = "scviv2_double_attention_hd"
    }
    else if (use_weighted)
     {
        method_name = "scviv2_weighted"
    }
    else if (use_prior)
     {
        method_name = "scviv2_prior"
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
        --use_double_attention_ld ${use_double_attention_ld} \\
        --use_double_attention_hd ${use_double_attention_hd}
    """

    output:
    path adata_in
    path model_out
}
