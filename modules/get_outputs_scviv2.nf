process get_outputs_scviv2 {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.final.scviv2.h5ad"
    """
    python3 ${params.bin.get_outputs_scviv2} \\
        --config_in ${config_in} \\
        --adata_in ${adata_in} \\
        --adata_out ${adata_out}
    """

    output:
    path adata_out
}
