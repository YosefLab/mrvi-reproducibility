#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl = 2

include { preprocess_data } from ".workflows/preprocess_data.nf"
include { train_mrvi } from ".workflows/train_mrvi.nf"
include { downstream_analysis } from ".workflows/downstream_analysis.nf"

workflow {

    if (params.workflow == "preprocess_data") {
        preprocess_data()
    } else if (params.workflow == "train_mrvi") {
        train_mrvi()
    } else if (params.workflow == "downstream_analysis") {
        downstream_analysis()
    }
}
