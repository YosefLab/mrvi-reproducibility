process fit_mrvi {
    input:
    path adata_in
    val use_reference
    val use_linear_uz
    val use_mlp_uz
    val use_same_dim_uz
    val use_encoder_regularnorm
    val use_iso_prior
    val z_dim
    val u_dim

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"

    if (use_reference) {
        method_name = "mrvi_attention_mog"
    }
    else if (use_linear_uz) {
        method_name = "mrvi_linear_uz"
    }
    else if (use_mlp_uz) {
        method_name = "mrvi_mlp_uz"
    }
    else if (use_same_dim_uz) {
        method_name = "mrvi_samedim_uz"
    }
    else if (use_encoder_regularnorm) {
        method_name = "mrvi_encoder_regularnorm"
    }
    else if (use_iso_prior) {
        method_name = "mrvi_attention_iso"
    }
    method_name = "${method_name}_z_${z_dim}_u_${u_dim}"

    model_out = "${params.outputs.models}/${adata_name}.${method_name}"
    """
    python3 ${params.bin.fit_mrvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --model_out ${model_out} \\
        --use_reference ${use_reference} \\
        --use_linear_uz ${use_linear_uz} \\
        --use_mlp_uz ${use_mlp_uz} \\
        --use_same_dim_uz ${use_same_dim_uz} \\
        --use_encoder_regularnorm ${use_encoder_regularnorm} \\
        --use_iso_prior ${use_iso_prior} \\
        --z_dim ${z_dim} \\
        --u_dim ${u_dim}
    """

    output:
    path adata_in
    path model_out
}
