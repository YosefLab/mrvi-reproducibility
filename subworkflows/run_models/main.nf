include {
    fit_scviv2;
    fit_scviv2 as fit_scviv2_mlp;
    fit_scviv2 as fit_scviv2_mlp_smallu;
    fit_scviv2 as fit_scviv2_attention;
    fit_scviv2 as fit_scviv2_attention_smallu;
    fit_scviv2 as fit_scviv2_attention_noprior;
    fit_scviv2 as fit_scviv2_attention_no_prior_mog;
    fit_scviv2 as fit_scviv2_attention_mog;
    fit_scviv2 as fit_scviv2_attention_no_prior_mog_large;
} from params.modules.fit_scviv2
include {
    get_latent_scviv2;
    get_latent_scviv2 as get_latent_scviv2_mlp;
    get_latent_scviv2 as get_latent_scviv2_mlp_smallu;
    get_latent_scviv2 as get_latent_scviv2_attention;
    get_latent_scviv2 as get_latent_scviv2_attention_smallu;
    get_latent_scviv2 as get_latent_scviv2_attention_noprior;
    get_latent_scviv2 as get_latent_scviv2_attention_no_prior_mog;
    get_latent_scviv2 as get_latent_scviv2_attention_mog;
    get_latent_scviv2 as get_latent_scviv2_attention_no_prior_mog_large;
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
    scvi_attention_noprior_outs = fit_scviv2_attention_noprior(adatas_in, false, false, false, false, true, false, false, false) | get_latent_scviv2_attention_noprior
    scvi_attention_noprior_adata = scvi_attention_noprior_outs.adata

    scvi_attention_no_prior_mog_outs = fit_scviv2_attention_no_prior_mog(adatas_in, false, false, false, false, false, true, false, false) | get_latent_scviv2_attention_no_prior_mog
    scvi_attention_no_prior_mog_adata = scvi_attention_no_prior_mog_outs.adata

    scvi_attention_mog_outs = fit_scviv2_attention_mog(adatas_in, false, false, false, false, false, false, true, false) | get_latent_scviv2_attention_mog
    scvi_attention_mog_adata = scvi_attention_mog_outs.adata

    scvi_attention_no_prior_mog_large_outs = fit_scviv2_attention_no_prior_mog_large(adatas_in, false, false, false, false, false, false, false, true) | get_latent_scviv2_attention_no_prior_mog_large
    scvi_attention_no_prior_mog_large_adata = scvi_attention_no_prior_mog_large_outs.adata

    distance_matrices = scvi_attention_no_prior_mog_large_outs.distance_matrices.concat(
        scvi_attention_no_prior_mog_large_outs.normalized_distance_matrices,
    )
    // adatas = scvi_attention_no_prior_mog_large_adata

    distance_matrices = scvi_attention_noprior_outs.distance_matrices.concat(
        scvi_attention_noprior_outs.normalized_distance_matrices,
        scvi_attention_no_prior_mog_outs.distance_matrices,
        scvi_attention_no_prior_mog_outs.normalized_distance_matrices,
        scvi_attention_mog_outs.distance_matrices,
        scvi_attention_mog_outs.normalized_distance_matrices,
        scvi_attention_no_prior_mog_large_outs.distance_matrices,
        scvi_attention_no_prior_mog_large_outs.normalized_distance_matrices,
    )
    adatas = scvi_attention_noprior_adata.concat(
        scvi_attention_no_prior_mog_adata,
        scvi_attention_mog_adata,
        scvi_attention_no_prior_mog_large_adata,
    )


    if ( params.runAllMRVIModels ) {
        scvi_outs = fit_scviv2(adatas_in, false, false, false, false, false, false, false) | get_latent_scviv2
        scvi_adata = scvi_outs.adata

        // Run scviv2 mlp
        scvi_mlp_outs = fit_scviv2_mlp(adatas_in, true, false, false, false, false, false, false) | get_latent_scviv2_mlp
        scvi_mlp_adata = scvi_mlp_outs.adata

        // Run scviv2 mlp smallu
        scvi_mlp_smallu_outs = fit_scviv2_mlp_smallu(adatas_in, false, true, false, false, false, false, false) | get_latent_scviv2_mlp_smallu
        scvi_mlp_smallu_adata = scvi_mlp_smallu_outs.adata

        scvi_attention_outs = fit_scviv2_attention(adatas_in, false, false, true, false, false, false, false) | get_latent_scviv2_attention
        scvi_attention_adata = scvi_attention_outs.adata

        scvi_attention_smallu_outs = fit_scviv2_attention_smallu(adatas_in, false, false, false, true, false, false, false) | get_latent_scviv2_attention_smallu
        scvi_attention_smallu_adata = scvi_attention_smallu_outs.adata

        distance_matrices = distance_matrices.concat(
            scvi_outs.distance_matrices,
            scvi_outs.normalized_distance_matrices,
            scvi_mlp_outs.distance_matrices,
            scvi_mlp_outs.normalized_distance_matrices,
            scvi_mlp_smallu_outs.distance_matrices,
            scvi_mlp_smallu_outs.normalized_distance_matrices,
            scvi_attention_outs.distance_matrices,
            scvi_attention_outs.normalized_distance_matrices,
            scvi_attention_smallu_outs.distance_matrices,
            scvi_attention_smallu_outs.normalized_distance_matrices,
        )

        // Organize all outputs
        adatas = adatas.concat(
            scvi_adata,
            scvi_mlp_adata,
            scvi_mlp_smallu_adata,
            scvi_attention_adata,
            scvi_attention_smallu_adata,
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
