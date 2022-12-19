process get_outputs_scviv2 {
    input:
    path adata_out

    script:
    adata_name = adata_out.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    adata_out = "${params.outputs.data}/${adata_name}.final.scviv2.h5ad"
    """
    python3 ${params.bin.get_outputs_scviv2} \\
        --model_in ${model_in} \\
        --adata_in ${adata_in}
    """

    output:
    path adata_out
}
