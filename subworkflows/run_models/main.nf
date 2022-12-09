include { fit_mrvi } from params.modules.fit_mrvi
include { get_latent_mrvi } from params.modules.get_latent_mrvi
include { fit_scviv2 } from params.modules.fit_scviv2
include { get_latent_scviv2 } from params.modules.get_latent_scviv2
include { get_outs_scviv2 } from params.modules.get_outs_scviv2

workflow run_models {
    take:
    inputs // Channel of input AnnDatas
    
    main:
    fit_scviv2(inputs) | get_latent_scviv2 | get_outs_scviv2
    fit_mrvi(inputs) | get_latent_mrvi

    emit:
    get_latent_mrvi.out.concat(get_latent_scviv2.out)
}