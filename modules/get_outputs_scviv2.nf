process get_outputs_scviv2 {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    """
    python3 ${params.bin.get_outputs_scviv2} \\
        --config_in ${config_in} \\
        --adata_in ${adata_in} \\
        --adata_out ${adata_in}
    """

    output:
    tuple val(adata_name), path(adata_in)
}
