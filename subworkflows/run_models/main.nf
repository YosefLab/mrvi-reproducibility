include { fit_mrvi } from params.modules.fit_mrvi
include { get_latent_mrvi } from params.modules.get_latent_mrvi
include { get_outputs_mrvi } from params.modules.get_outputs_mrvi
include { fit_scviv2 } from params.modules.fit_scviv2
include { get_latent_scviv2 } from params.modules.get_latent_scviv2
include { get_outputs_scviv2 } from params.modules.get_outputs_scviv2
include { fit_and_get_latent_composition_scvi } from params.modules.fit_and_get_latent_composition_scvi
include { fit_and_get_latent_composition_pca } from params.modules.fit_and_get_latent_composition_pca
include { compute_rf } from params.modules.compute_rf
include { compute_2dreps } from params.modules.compute_2dreps

workflow run_models {
    take:
    inputs // Channel of input AnnDatas

    main:
    adatas_in=inputs.map { it[0] }
    distance_matrices_gt=inputs.map { it[1] }

    // Step 1: Run models
    // Run scviv2, compute latents, distance matrices
    scvi_outs = fit_scviv2(adatas_in) | get_latent_scviv2 | get_outputs_scviv2
    scvi_adata = scvi_outs.adata

    distance_matrices = scvi_outs.distance_matrices
    adatas = get_latent_scviv2.out

    if ( params.runAllModels) {
        // Run MRVI, compute latents, distance matrices
        mrvi_outs = fit_mrvi(adatas_in) | get_latent_mrvi | get_outputs_mrvi
        mrvi_adata = mrvi_outs.adata

        // Run compositional models
        c_scvi_outs=fit_and_get_latent_composition_scvi(adatas_in)
        c_pca_outs=fit_and_get_latent_composition_pca(adatas_in)

        distance_matrices = distance_matrices.concat(
        mrvi_outs.distance_matrices,
        c_pca_outs.distance_matrices,
        c_scvi_outs.distance_matrices
        )
        adatas = adatas.concat(
            get_latent_mrvi.out,
            scvi_adata,
            c_pca_outs.adata,
            c_scvi_outs.adata
        )

    }
    adatas.view()

    // Step 2: Compute metrics
    // Compute RF
    dmat_gt_symsim=distance_matrices_gt.filter( { it =~ /symsim_new.*/ } )
    dmat_inf_symsim=distance_matrices.filter( { it =~ /symsim_new.*/ } )
    dmats=dmat_gt_symsim.combine(dmat_inf_symsim)
    dmats.view()
    rfs = compute_rf(dmats)

    // Step 3: Compute UMAPs
    adatas=compute_2dreps(adatas)

    emit:
    adatas
    distance_matrices
    rfs
}
