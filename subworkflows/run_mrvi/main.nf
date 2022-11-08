include { fit_mrvi } from params.module.fit_mrvi
include { get_latent_mrvi } from params.module.get_latent_mrvi


workflow run_mrvi {
    take:
    input

    main:
    fit_mrvi(input) | get_latent_mrvi
    
    emit:
    get_latent_mrvi.out
}
