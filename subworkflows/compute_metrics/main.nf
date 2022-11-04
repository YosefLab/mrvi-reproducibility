include { compute_scib_metrics} from params.module.compute_scib_metrics
include { compute_vendi } from params.module.compute_vendi


workflow compute_metrics {
    take:
    input

    main:
    compute_scib_metrics(input)
    compute_vendi(input)
}