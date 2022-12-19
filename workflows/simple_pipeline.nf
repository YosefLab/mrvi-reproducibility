include { preprocess_data } from params.subworkflows.preprocess_data
include { run_models } from params.subworkflows.run_models
include { compute_metrics } from params.subworkflows.compute_metrics
include { analyze_results } from params.subworkflows.analyze_results


workflow run_main {
    main:
    inputs = Channel.fromPath(params.inputs)
    preprocess_data(inputs) | run_models | compute_metrics | analyze_results
}
