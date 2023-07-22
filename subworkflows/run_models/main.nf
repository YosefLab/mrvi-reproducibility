include {
    fit_scviv2;
    fit_scviv2 as fit_scviv2_attention_noprior;
    fit_scviv2 as fit_scviv2_attention_no_prior_mog;
    fit_scviv2 as fit_scviv2_z30;
    fit_scviv2 as fit_scviv2_z20_u5;
    fit_scviv2 as fit_scviv2_z50_u5;
    fit_scviv2 as fit_scviv2_z30_u5;
    fit_scviv2 as fit_scviv2_z100_u5;
} from params.modules.fit_scviv2
include {
    get_latent_scviv2;
    get_latent_scviv2 as get_latent_scviv2_attention_noprior;
    get_latent_scviv2 as get_latent_scviv2_attention_no_prior_mog;
    get_latent_scviv2 as get_latent_scviv2_z30;
    get_latent_scviv2 as get_latent_scviv2_z20_u5;
    get_latent_scviv2 as get_latent_scviv2_z50_u5;
    get_latent_scviv2 as get_latent_scviv2_z30_u5;
    get_latent_scviv2 as get_latent_scviv2_z100_u5;
} from params.modules.get_latent_scviv2
include {
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_scvi_clusterkey;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_pca_clusterkey;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_scvi_leiden;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_pca_leiden;
} from params.modules.fit_and_get_latent_composition_baseline
include { compute_rf } from params.modules.compute_rf
include { compute_2dreps } from params.modules.compute_2dreps

workflow run_models {
    take:
    inputs // Channel of input AnnDatas

    main:
    adatas_in=inputs.map { it[0] }
    distance_matrices_gt=inputs.map { it[1] }

    // Step 1: Run models
    // Run base model
    scvi_attention_noprior_outs = fit_scviv2_attention_noprior(adatas_in, true, false, false, false, false, false, false) | get_latent_scviv2_attention_noprior
    scvi_attention_noprior_adata = scvi_attention_noprior_outs.adata

    scvi_attention_no_prior_mog_outs = fit_scviv2_attention_no_prior_mog(adatas_in, false, true, false, false, false, false, false) | get_latent_scviv2_attention_no_prior_mog
    scvi_attention_no_prior_mog_adata = scvi_attention_no_prior_mog_outs.adata

    distance_matrices = scvi_attention_noprior_outs.distance_matrices.concat(
        scvi_attention_noprior_outs.normalized_distance_matrices,
        scvi_attention_no_prior_mog_outs.distance_matrices,
        scvi_attention_no_prior_mog_outs.normalized_distance_matrices,
    )
    adatas = scvi_attention_noprior_adata.concat(
        scvi_attention_no_prior_mog_adata,
    )

    if ( params.runAllMRVIModels ) {
        // run old base model
        scvi_outs = fit_scviv2(adatas_in, false, false, false, false, false, false, false) | get_latent_scviv2
        scvi_adata = scvi_outs.adata

        scvi_z30_outs = fit_scviv2_z30(adatas_in, false, false, true, false, false, false, false) | get_latent_scviv2_z30
        scvi_z30_adata = scvi_z30_outs.adata

        scvi_z20_u5_outs = fit_scviv2_z20_u5(adatas_in, false, false, false, true, false, false, false) | get_latent_scviv2_z20_u5
        scvi_z20_u5_adata = scvi_z20_u5_outs.adata

        scvi_z50_u5_outs = fit_scviv2_z50_u5(adatas_in, false, false, false, false, true, false, false) | get_latent_scviv2_z50_u5
        scvi_z50_u5_adata = scvi_z50_u5_outs.adata

        scvi_z30_u5_outs = fit_scviv2_z30_u5(adatas_in, false, false, false, false, false, true, false) | get_latent_scviv2_z30_u5
        scvi_z30_u5_adata = scvi_z30_u5_outs.adata

        // scvi_z100_u5_outs = fit_scviv2_z100_u5(adatas_in, false, false, false, false, false, false, true) | get_latent_scviv2_z100_u5
        // scvi_z100_u5_adata = scvi_z100_u5_outs.adata

        distance_matrices = distance_matrices.concat(
            scvi_outs.distance_matrices,
            scvi_outs.normalized_distance_matrices,
            scvi_z30_outs.distance_matrices,
            scvi_z30_outs.normalized_distance_matrices,
            scvi_z20_u5_outs.distance_matrices,
            scvi_z20_u5_outs.normalized_distance_matrices,
            scvi_z50_u5_outs.distance_matrices,
            scvi_z50_u5_outs.normalized_distance_matrices,
            scvi_z30_u5_outs.distance_matrices,
            scvi_z30_u5_outs.normalized_distance_matrices,
            // scvi_z100_u5_outs.distance_matrices,
            // scvi_z100_u5_outs.normalized_distance_matrices
        )

        // Organize all outputs
        adatas = adatas.concat(
            scvi_adata,
            scvi_attention_noprior_adata,
            scvi_attention_no_prior_mog_adata,
            scvi_z30_adata,
            scvi_z20_u5_adata,
            scvi_z50_u5_adata,
            scvi_z30_u5_adata,
            // scvi_z100_u5_adata
        )
    }

    if ( params.runAllModels) {
        // Run compositional models
        c_scvi_clusterkey_outs=fit_and_get_latent_composition_scvi_clusterkey(adatas_in, "SCVI_clusterkey_subleiden1")
        c_pca_clusterkey_outs=fit_and_get_latent_composition_pca_clusterkey(adatas_in, "PCA_clusterkey_subleiden1")
        c_scvi_leiden_outs=fit_and_get_latent_composition_scvi_leiden(adatas_in, "SCVI_leiden1_subleiden1")
        c_pca_leiden_outs=fit_and_get_latent_composition_pca_leiden(adatas_in, "PCA_leiden1_subleiden1")

        distance_matrices = distance_matrices.concat(
            c_pca_clusterkey_outs.distance_matrices,
            c_scvi_clusterkey_outs.distance_matrices,
            c_pca_leiden_outs.distance_matrices,
            c_scvi_leiden_outs.distance_matrices
        )
        adatas = adatas.concat(
            c_pca_clusterkey_outs.adata,
            c_scvi_clusterkey_outs.adata,
            c_pca_leiden_outs.adata,
            c_scvi_leiden_outs.adata
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

    adatas=compute_2dreps(adatas)

    emit:
    adatas
    distance_matrices
    rfs
}
