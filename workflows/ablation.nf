include { preprocess_data } from params.subworkflows.preprocess_data
include { run_models } from params.subworkflows.run_models
include { compute_metrics } from params.subworkflows.compute_metrics
include { analyze_results } from params.subworkflows.analyze_results


workflow run_main {
    main:
    input = Channel.fromList(params.inputs)

    outs = preprocess_data(input) | run_models
    // metrics = compute_metrics(outs.adatas, outs.distance_matrices)

    results = outs.adatas.concat(
        outs.rfs,
        outs.distance_matrices,
        // metrics,
    )

    // analyze_results(results)
}
