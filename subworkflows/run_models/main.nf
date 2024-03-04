include {
    fit_mrvi as fit_mrvi_attention_mog;
    fit_mrvi as fit_mrvi_linear_uz;
    fit_mrvi as fit_mrvi_mlp_uz;
    fit_mrvi as fit_mrvi_samedim_uz;
    fit_mrvi as fit_mrvi_regularnorm;
    fit_mrvi as fit_mrvi_isoprior_2_2;
    fit_mrvi as fit_mrvi_isoprior_5_2;
    fit_mrvi as fit_mrvi_isoprior_10_2;
    fit_mrvi as fit_mrvi_isoprior_30_2;
    fit_mrvi as fit_mrvi_isoprior_50_2;
    fit_mrvi as fit_mrvi_isoprior_5_5;
    fit_mrvi as fit_mrvi_isoprior_10_5;
    fit_mrvi as fit_mrvi_isoprior_30_5;
    fit_mrvi as fit_mrvi_isoprior_50_5;
    fit_mrvi as fit_mrvi_isoprior_10_10;
    fit_mrvi as fit_mrvi_isoprior_30_10;
    fit_mrvi as fit_mrvi_isoprior_50_10;
    fit_mrvi as fit_mrvi_isoprior_30_30;
    fit_mrvi as fit_mrvi_isoprior_50_30;
    fit_mrvi as fit_mrvi_isoprior_50_50;
} from params.modules.fit_mrvi
include {
    get_latent_mrvi as get_latent_mrvi_attention_mog;
    get_latent_mrvi as get_latent_mrvi_linear_uz;
    get_latent_mrvi as get_latent_mrvi_mlp_uz;
    get_latent_mrvi as get_latent_mrvi_samedim_uz;
    get_latent_mrvi as get_latent_mrvi_regularnorm;
    get_latent_mrvi as get_latent_mrvi_isoprior_2_2;
    get_latent_mrvi as get_latent_mrvi_isoprior_5_2;
    get_latent_mrvi as get_latent_mrvi_isoprior_10_2;
    get_latent_mrvi as get_latent_mrvi_isoprior_30_2;
    get_latent_mrvi as get_latent_mrvi_isoprior_50_2;
    get_latent_mrvi as get_latent_mrvi_isoprior_5_5;
    get_latent_mrvi as get_latent_mrvi_isoprior_10_5;
    get_latent_mrvi as get_latent_mrvi_isoprior_30_5;
    get_latent_mrvi as get_latent_mrvi_isoprior_50_5;
    get_latent_mrvi as get_latent_mrvi_isoprior_10_10;
    get_latent_mrvi as get_latent_mrvi_isoprior_30_10;
    get_latent_mrvi as get_latent_mrvi_isoprior_50_10;
    get_latent_mrvi as get_latent_mrvi_isoprior_30_30;
    get_latent_mrvi as get_latent_mrvi_isoprior_50_30;
    get_latent_mrvi as get_latent_mrvi_isoprior_50_50;
} from params.modules.get_latent_mrvi
include {
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_pca_clusterkey;
    fit_and_get_latent_composition_baseline as fit_and_get_latent_composition_sc
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
    mrvi_isoprior_2_2_outs = fit_mrvi_isoprior_2_2(adatas_in, false, false, false, false, false, true, 2, 2) | get_latent_mrvi_isoprior_2_2
    mrvi_isoprior_2_2_adata = mrvi_isoprior_2_2_outs.adata

    mrvi_isoprior_5_2_outs = fit_mrvi_isoprior_5_2(adatas_in, false, false, false, false, false, true, 5, 2) | get_latent_mrvi_isoprior_5_2
    mrvi_isoprior_5_2_adata = mrvi_isoprior_5_2_outs.adata

    mrvi_isoprior_10_2_outs = fit_mrvi_isoprior_10_2(adatas_in, false, false, false, false, false, true, 10, 2) | get_latent_mrvi_isoprior_10_2
    mrvi_isoprior_10_2_adata = mrvi_isoprior_10_2_outs.adata

    mrvi_isoprior_30_2_outs = fit_mrvi_isoprior_30_2(adatas_in, false, false, false, false, false, true, 30, 2) | get_latent_mrvi_isoprior_30_2
    mrvi_isoprior_30_2_adata = mrvi_isoprior_30_2_outs.adata

    mrvi_isoprior_50_2_outs = fit_mrvi_isoprior_50_2(adatas_in, false, false, false, false, false, true, 50, 2) | get_latent_mrvi_isoprior_50_2
    mrvi_isoprior_50_2_adata = mrvi_isoprior_50_2_outs.adata

    mrvi_isoprior_5_5_outs = fit_mrvi_isoprior_5_5(adatas_in, false, false, false, false, false, true, 5, 5) | get_latent_mrvi_isoprior_5_5
    mrvi_isoprior_5_5_adata = mrvi_isoprior_5_5_outs.adata

    mrvi_isoprior_10_5_outs = fit_mrvi_isoprior_10_5(adatas_in, false, false, false, false, false, true, 10, 5) | get_latent_mrvi_isoprior_10_5
    mrvi_isoprior_10_5_adata = mrvi_isoprior_10_5_outs.adata

    mrvi_isoprior_30_5_outs = fit_mrvi_isoprior_30_5(adatas_in, false, false, false, false, false, true, 30, 5) | get_latent_mrvi_isoprior_30_5
    mrvi_isoprior_30_5_adata = mrvi_isoprior_30_5_outs.adata

    mrvi_isoprior_50_5_outs = fit_mrvi_isoprior_50_5(adatas_in, false, false, false, false, false, true, 50, 5) | get_latent_mrvi_isoprior_50_5
    mrvi_isoprior_50_5_adata = mrvi_isoprior_50_5_outs.adata

    mrvi_isoprior_10_10_outs = fit_mrvi_isoprior_10_10(adatas_in, false, false, false, false, false, true, 10, 10) | get_latent_mrvi_isoprior_10_10
    mrvi_isoprior_10_10_adata = mrvi_isoprior_10_10_outs.adata

    mrvi_isoprior_30_10_outs = fit_mrvi_isoprior_30_10(adatas_in, false, false, false, false, false, true, 30, 10) | get_latent_mrvi_isoprior_30_10
    mrvi_isoprior_30_10_adata = mrvi_isoprior_30_10_outs.adata

    mrvi_isoprior_50_10_outs = fit_mrvi_isoprior_50_10(adatas_in, false, false, false, false, false, true, 50, 10) | get_latent_mrvi_isoprior_50_10
    mrvi_isoprior_50_10_adata = mrvi_isoprior_50_10_outs.adata

    mrvi_isoprior_30_30_outs = fit_mrvi_isoprior_30_30(adatas_in, false, false, false, false, false, true, 30, 30) | get_latent_mrvi_isoprior_30_30
    mrvi_isoprior_30_30_adata = mrvi_isoprior_30_30_outs.adata

    mrvi_isoprior_50_30_outs = fit_mrvi_isoprior_50_30(adatas_in, false, false, false, false, false, true, 50, 30) | get_latent_mrvi_isoprior_50_30
    mrvi_isoprior_50_30_adata = mrvi_isoprior_50_30_outs.adata

    mrvi_isoprior_50_50_outs = fit_mrvi_isoprior_50_50(adatas_in, false, false, false, false, false, true, 50, 50) | get_latent_mrvi_isoprior_50_50
    mrvi_isoprior_50_50_adata = mrvi_isoprior_50_50_outs.adata

    distance_matrices = mrvi_isoprior_2_2_outs.distance_matrices.concat(
        mrvi_isoprior_5_2_outs.distance_matrices,
        mrvi_isoprior_10_2_outs.distance_matrices,
        mrvi_isoprior_30_2_outs.distance_matrices,
        mrvi_isoprior_50_2_outs.distance_matrices,
        mrvi_isoprior_5_5_outs.distance_matrices,
        mrvi_isoprior_10_5_outs.distance_matrices,
        mrvi_isoprior_30_5_outs.distance_matrices,
        mrvi_isoprior_50_5_outs.distance_matrices,
        mrvi_isoprior_10_10_outs.distance_matrices,
        mrvi_isoprior_30_10_outs.distance_matrices,
        mrvi_isoprior_50_10_outs.distance_matrices,
        mrvi_isoprior_30_30_outs.distance_matrices,
        mrvi_isoprior_50_30_outs.distance_matrices,
        mrvi_isoprior_50_50_outs.distance_matrices,
    )

    adatas = mrvi_isoprior_2_2_adata.concat(
        mrvi_isoprior_5_2_adata,
        mrvi_isoprior_10_2_adata,
        mrvi_isoprior_30_2_adata,
        mrvi_isoprior_50_2_adata,
        mrvi_isoprior_5_5_adata,
        mrvi_isoprior_10_5_adata,
        mrvi_isoprior_30_5_adata,
        mrvi_isoprior_50_5_adata,
        mrvi_isoprior_10_10_adata,
        mrvi_isoprior_30_10_adata,
        mrvi_isoprior_50_10_adata,
        mrvi_isoprior_30_30_adata,
        mrvi_isoprior_50_30_adata,
        mrvi_isoprior_50_50_adata,
    )

    if ( params.runMILO ) {
        run_milo(adatas_in)
        run_milode(adatas_in)
    }

    if ( params.runAllMRVIModels ) {
        mrvi_linear_uz_outs = fit_mrvi_linear_uz(adatas_in, false, true, false, false, false, false) | get_latent_mrvi_linear_uz
        mrvi_linear_uz_adata = mrvi_linear_uz_outs.adata

        mrvi_mlp_uz_outs = fit_mrvi_mlp_uz(adatas_in, false, false, true, false, false, false) | get_latent_mrvi_mlp_uz
        mrvi_mlp_uz_adata = mrvi_mlp_uz_outs.adata

        mrvi_samedim_uz_outs = fit_mrvi_samedim_uz(adatas_in, false, false, false, true, false, false) | get_latent_mrvi_samedim_uz
        mrvi_samedim_uz_adata = mrvi_samedim_uz_outs.adata

        mrvi_regularnorm_outs = fit_mrvi_regularnorm(adatas_in, false, false, false, false, true, false) | get_latent_mrvi_regularnorm
        mrvi_regularnorm_adata = mrvi_regularnorm_outs.adata

        distance_matrices = distance_matrices.concat(
            mrvi_linear_uz_outs.distance_matrices,
            mrvi_linear_uz_outs.normalized_distance_matrices,
            mrvi_mlp_uz_outs.distance_matrices,
            mrvi_mlp_uz_outs.normalized_distance_matrices,
            mrvi_samedim_uz_outs.distance_matrices,
            mrvi_samedim_uz_outs.normalized_distance_matrices,
            mrvi_regularnorm_outs.distance_matrices,
            mrvi_regularnorm_outs.normalized_distance_matrices,
        )

        // Organize all outputs
        adatas = adatas.concat(
            mrvi_regularnorm_adata,
            mrvi_mlp_uz_adata,
            mrvi_samedim_uz_adata,
            mrvi_regularnorm_adata,
            
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
    // dmat_gt_symsim=distance_matrices_gt.filter( { it =~ /symsim_new.*/ } )
    // dmat_inf_symsim=distance_matrices.filter( { it =~ /symsim_new.*/ } )
    // dmats=dmat_gt_symsim.combine(dmat_inf_symsim)
    // dmats.view()
    // rfs = compute_rf(dmats)

    // adatas=compute_2dreps(adatas)

    emit:
    adatas
    distance_matrices
    // rfs
}
