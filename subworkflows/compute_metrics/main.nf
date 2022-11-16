include { scib } from params.modules.scib
include { vendi } from params.modules.vendi


workflow compute_metrics {
    take:
    inputs

    main:
    scib(inputs)
    vendi(inputs)

    emit:
    scib.out
    vendi.out
}
