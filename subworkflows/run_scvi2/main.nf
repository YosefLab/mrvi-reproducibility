include { fit_scvi2 } from params.module.fit_scvi2
include { get_latent_scvi2 } from params.module.get_latent_scvi2

workflow run_scvi2 {
    take:
    input

    main:
    fit_scvi2(input) | get_latent_scvi2
    
    emit:
    get_latent_scvi2.out
}