process fit_and_get_latent_composition_scvi {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.composition_scvi.h5ad"
    """
    python3 ${params.bin.fit_and_get_latent_composition_scvi} \\
        --adata_in ${adata_in} \\
        --config_in ${config_in} \\
        --adata_out ${adata_out}
    """

    output:
    tuple val(adata_name), path(adata_out)
}
