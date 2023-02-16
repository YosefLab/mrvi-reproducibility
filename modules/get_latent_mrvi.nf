process get_latent_mrvi {
    maxForks 1

    input:
    path adata_in
    path model_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.mrvi.h5ad"
    """
    python3 ${params.bin.get_latent_mrvi} \\
        --adata_in ${adata_in} \\
        --model_in ${model_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out}
    """

    output:
    path adata_out
}
