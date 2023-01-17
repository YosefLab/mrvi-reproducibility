include { scib } from params.modules.scib
include { vendi } from params.modules.vendi


workflow compute_metrics {
    take:
    adatas
    distance_matrices

    main:
    scib(adatas)
    vendi(distance_matrices)

    emit:
    scib.out.concat(vendi.out)
}
