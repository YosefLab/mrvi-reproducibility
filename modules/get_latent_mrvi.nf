process get_latent_mrvi {
    conda "${params.env.run_mrvi}"

    input:
    path adata_in
    path model_in

    script:
    adata_name = adata_in.getBaseName()
    config_in = "${params.conf.dataset}/${adata_name}.json"
    adata_out = "${params.output.run_mrvi_latent_data}/${adata_name}.h5ad"
    """
    python3 ${params.script.get_latent_mrvi} \\
        --adata_in ${adata_in} \\
        --model_in ${model_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out}
    """

    output:
    path adata_out
}
