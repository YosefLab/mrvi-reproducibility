include { preprocess_data } from params.subworkflow.preprocess_data
include { run_mrvi } from params.subworkflow.run_mrvi
include { compute_metrics } from params.subworkflow.compute_metrics


workflow aws_pipeline {
    main:
    raw_data = Channel.fromPath(params.input.preprocess_data)
    preprocess_data(raw_data) | run_mrvi | compute_metrics
}