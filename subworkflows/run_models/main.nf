include { fit_mrvi } from params.modules.fit_mrvi
include { get_latent_mrvi } from params.modules.get_latent_mrvi
include { fit_scviv2 } from params.modules.fit_scviv2
include { get_latent_scviv2 } from params.modules.get_latent_scviv2
include { get_outputs_scviv2 } from params.modules.get_outputs_scviv2
include { fit_and_get_latent_composition_scvi } from params.modules.fit_and_get_latent_composition_scvi
include { fit_and_get_latent_composition_pca } from params.modules.fit_and_get_latent_composition_pca
include { compute_rf } from params.modules.compute_rf

workflow run_models {
    take:
    inputs // Channel of input AnnDatas

    main:
    // Run scviv2, compute latents, distance matrices, and RF (if there is GT distances)
    fit_scviv2(inputs) | get_latent_scviv2 | get_outputs_scviv2
    compute_rf(get_outputs_scviv2.out)

    // Run MRVI, compute latents (old code)
    fit_mrvi(inputs) | get_latent_mrvi

    // Run compositional models
    fit_and_get_latent_composition_scvi(inputs)
    fit_and_get_latent_composition_pca(inputs)

    emit:
    get_latent_mrvi.out.concat(
        get_outputs_scviv2.out,
        fit_and_get_latent_composition_scvi.out,
        fit_and_get_latent_composition_pca.out
    )
}
