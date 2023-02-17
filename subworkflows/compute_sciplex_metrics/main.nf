include { compute_sciplex_metrics } from params.modules.compute_sciplex_metrics


workflow compute_metrics {
    take:
    distance_matrices
    gt_matrices

    main:
    compute_sciplex_metrics(distance_matrices, gt_matrices)

    emit:
    compute_sciplex_metrics.out
}
