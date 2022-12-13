include { fit_mrvi } from params.modules.fit_mrvi
include { get_latent_mrvi } from params.modules.get_latent_mrvi
include { fit_scviv2 } from params.modules.fit_scviv2
include { get_latent_scviv2 } from params.modules.get_latent_scviv2
include { get_outs_scviv2 } from params.modules.get_outs_scviv2
include { fit_and_get_latent_composition_scvi } from params.modules.fit_and_get_latent_composition_scvi
include { fit_and_get_latent_composition_pca } from params.modules.fit_and_get_latent_composition_pca

workflow run_models {
    take:
    inputs // Channel of input AnnDatas
    
    main:
    fit_scviv2(inputs) | get_latent_scviv2 | get_outs_scviv2
    fit_mrvi(inputs) | get_latent_mrvi
    fit_and_get_latent_composition_scvi(inputs)
    fit_and_get_latent_composition_pca(inputs)

    emit:
    get_latent_mrvi.out.concat(
        get_latent_scviv2.out,
        fit_and_get_latent_composition_scvi.out,
        fit_and_get_latent_composition_pca.out
    )
}