#!/usr/bin/env nextflow

include { simple_pipeline } from params.simple_pipeline_workflow
include { aws_pipeline } from params.aws_pipeline_workflow

workflow {
    if (params.workflow == "simple_pipeline") {
        simple_pipeline()
    } else if (params.workflow == "aws_pipeline") {
        aws_pipeline()
    }
}
