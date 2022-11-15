#!/usr/bin/env nextflow

include { run_main } from "${params.workflows.root}/${params.workflow}.nf"

workflow {
    main:
    run_main()
}
