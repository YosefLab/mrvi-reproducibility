process run_milode {
    input:
    path adata_in

    script:
    adata_name = adata_in.getSimpleName()
    config_in = "${params.conf.datasets}/${adata_name}.json"
    basedir = "${params.outputs.models}"
    out_da = "${params.outputs.models}/${adata_name}.MILODE.de_analysis.tsv"
    out_assignments = "${params.outputs.models}/${adata_name}.MILODE.assignments.mtx"


    """
    if [ ! -d ${basedir} ]; then
        mkdir -p ${basedir}
    fi
    Rscript --vanilla ${params.bin.run_milode} ${adata_in} ${config_in} ${out_da} ${out_assignments}
    """

    output:
    path out_da
    path out_assignments
}
