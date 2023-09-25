process fit_scviv2 {
    input:
    path adata_in
    val use_attention_noprior
    val use_attention_no_prior_mog
<<<<<<< HEAD
    val z30
    val z20_u5
    val z50_u5
    val z30_u5
    val z100_u5
=======
    val use_attention_mog
    val use_attention_no_prior_mog_large
>>>>>>> main

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"

    if (use_attention_noprior) {
        method_name = "scviv2_attention_noprior"
    }
    else if (use_attention_no_prior_mog) {
        method_name = "scviv2_attention_no_prior_mog"
    }
<<<<<<< HEAD
    else if (z30) {
        method_name = "scviv2_z30"
    }
    else if (z20_u5) {
        method_name = "scviv2_z20_u5"
    }
    else if (z50_u5) {
        method_name = "scviv2_z50_u5"
    }
    else if (z30_u5) {
        method_name = "scviv2_z30_u5"
    }
    else if (z100_u5) {
        method_name = "scviv2_z100_u5"
=======
    else if (use_attention_mog) {
        method_name = "scviv2_attention_mog"
    }
    else if (use_attention_no_prior_mog_large) {
        method_name = "scviv2_attention_no_prior_mog_large"
>>>>>>> main
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
<<<<<<< HEAD
        --z30 ${z30} \\
        --z20_u5 ${z20_u5} \\
        --z50_u5 ${z50_u5} \\
        --z30_u5 ${z30_u5} \\
        --z100_u5 ${z100_u5}
=======
        --use_attention_mog ${use_attention_mog} \\
        --use_attention_no_prior_mog_large ${use_attention_no_prior_mog_large}
>>>>>>> main
    """

    output:
    path adata_in
    path model_out
}
