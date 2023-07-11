process run_milo {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    basedir = "${params.outputs.models}"
    out_da = "${params.outputs.models}/${adata_name}.MILO.da_analysis.tsv"
    out_assignments = "${params.outputs.models}/${adata_name}.MILO.assignments.mtx"


    """
    if [ ! -d ${basedir} ]; then
        mkdir -p ${basedir}
    fi
    Rscript --vanilla ${params.bin.run_milo} ${adata_in} ${config_in} ${out_da} ${out_assignments}
    """

    output:
    path out_da
    path out_assignments
}
