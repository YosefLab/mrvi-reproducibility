include { preprocess_data } from params.subworkflows.preprocess_data
include { run_models } from params.subworkflows.run_models
include { compute_metrics } from params.subworkflows.compute_metrics

workflow run_main {
    main:
    input = Channel.fromList(params.input.preprocess_data)
    preprocess_data(input) | run_models | compute_metrics
}
