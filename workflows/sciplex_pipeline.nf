include { preprocess_data } from params.subworkflows.preprocess_data
include { run_models } from params.subworkflows.run_models
include { compute_metrics } from params.subworkflows.compute_metrics
include { compute_sciplex_metrics } from params.subworkflows.compute_sciplex_metrics
include { analyze_results } from params.subworkflows.analyze_results


workflow run_main {
    main:
    input = Channel.fromPath(params.inputs)
    gt_clusters = Channel.fromPath(params.gt_clusters).collect()

    outs = preprocess_data(input) | run_models

    sciplex_metrics = compute_sciplex_metrics(outs.distance_matrices, gt_clusters)
    metrics = compute_metrics(outs.adatas, outs.distance_matrices)

    metrics = sciplex_metrics.concat(metrics)

    results = outs.adatas.concat(
        outs.rfs,
        outs.distance_matrices,
        metrics,
    )

    analyze_results(results)
}
