include {
    fit_scviv2 as fit_scviv2_attention_mog;
    fit_scviv2 as fit_scviv2_linear_uz;
    fit_scviv2 as fit_scviv2_mlp_uz;
    fit_scviv2 as fit_scviv2_samedim_uz;
    fit_scviv2 as fit_scviv2_regularnorm;
    fit_scviv2 as fit_scviv2_isoprior;
} from params.modules.fit_scviv2
include {
    get_latent_scviv2 as get_latent_scviv2_attention_mog;
    get_latent_scviv2 as get_latent_scviv2_linear_uz;
    get_latent_scviv2 as get_latent_scviv2_mlp_uz;
    get_latent_scviv2 as get_latent_scviv2_samedim_uz;
    get_latent_scviv2 as get_latent_scviv2_regularnorm;
    get_latent_scviv2 as get_latent_scviv2_isoprior;

} from params.modules.get_latent_scviv2
include {
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_scvi_clusterkey;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_pca_clusterkey;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_scvi_leiden;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_pca_leiden;
} from params.modules.fit_and_get_latent_composition_baseline
include { run_milo } from params.modules.run_milo
include { run_milode } from params.modules.run_milode
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
    scvi_attention_mog_outs = fit_scviv2_attention_mog(adatas_in, true, false, false, false, false, false) | get_latent_scviv2_attention_mog
    scvi_attention_mog_adata = scvi_attention_mog_outs.adata

    scvi_isoprior_outs = fit_scviv2_isoprior(adatas_in, false, false, false, false, false, true) | get_latent_scviv2_isoprior
    scvi_isoprior_adata = scvi_isoprior_outs.adata

    distance_matrices = scvi_attention_mog_outs.distance_matrices.concat(
        scvi_attention_mog_outs.normalized_distance_matrices,
        scvi_isoprior_outs.distance_matrices,
        scvi_isoprior_outs.normalized_distance_matrices,
    )
    adatas = scvi_attention_mog_adata.concat(
        scvi_isoprior_adata,
    )

    if ( params.runMILO ) {
        run_milo(adatas_in)
        run_milode(adatas_in)
    }

    if ( params.runAllMRVIModels ) {
        scvi_linear_uz_outs = fit_scviv2_linear_uz(adatas_in, false, true, false, false, false, false) | get_latent_scviv2_linear_uz
        scvi_linear_uz_adata = scvi_linear_uz_outs.adata

        scvi_mlp_uz_outs = fit_scviv2_mlp_uz(adatas_in, false, false, true, false, false, false) | get_latent_scviv2_mlp_uz
        scvi_mlp_uz_adata = scvi_mlp_uz_outs.adata

        scvi_samedim_uz_outs = fit_scviv2_samedim_uz(adatas_in, false, false, false, true, false, false) | get_latent_scviv2_samedim_uz
        scvi_samedim_uz_adata = scvi_samedim_uz_outs.adata

        scvi_regularnorm_outs = fit_scviv2_regularnorm(adatas_in, false, false, false, false, true, false) | get_latent_scviv2_regularnorm
        scvi_regularnorm_adata = scvi_regularnorm_outs.adata

        distance_matrices = distance_matrices.concat(
            scvi_linear_uz_outs.distance_matrices,
            scvi_linear_uz_outs.normalized_distance_matrices,
            scvi_mlp_uz_outs.distance_matrices,
            scvi_mlp_uz_outs.normalized_distance_matrices,
            scvi_samedim_uz_outs.distance_matrices,
            scvi_samedim_uz_outs.normalized_distance_matrices,
            scvi_regularnorm_outs.distance_matrices,
            scvi_regularnorm_outs.normalized_distance_matrices,
        )

        // Organize all outputs
        adatas = adatas.concat(
            scvi_regularnorm_adata,
            scvi_mlp_uz_adata,
            scvi_samedim_uz_adata,
            scvi_regularnorm_adata,
            
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
            c_pca_clusterkey_outs.normalized_distance_matrices,
            c_scvi_clusterkey_outs.distance_matrices,
            c_scvi_clusterkey_outs.normalized_distance_matrices,
            c_pca_leiden_outs.distance_matrices,
            c_pca_leiden_outs.normalized_distance_matrices,
            c_scvi_leiden_outs.distance_matrices,
            c_scvi_leiden_outs.normalized_distance_matrices,
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
