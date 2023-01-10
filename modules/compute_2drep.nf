process compute_2drep {
    input:
    path adata_in

    script:
    adata_model_name = adata_in.getBaseName()
    adata_out = "${params.outputs.data}/${adata_model_name}.final.h5ad"

    """
    python3 ${params.bin.compute_2drep} \\
        --adata_in ${adata_in} \\
        --adata_out ${adata_out}
    """

    output:
    path adata_out
}
