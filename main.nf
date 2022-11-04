#!/usr/bin/env nextflow

include { simple_pipeline } from params.simple_pipeline_workflow

workflow {
    if (params.workflow == "simple_pipeline") {
        simple_pipeline()
    }
}
