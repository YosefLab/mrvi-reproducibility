include { produce_figures_symsim_new } from params.modules.produce_figures_symsim_new
include { produce_figures_sciplex } from params.modules.produce_figures_sciplex
include { conduct_generic_analysis } from params.modules.conduct_generic_analysis

workflow analyze_results {
    take:
    inputs

    main:
    symsim_results = inputs.filter( { it =~ /symsim_new.*/ } ).collect()
    pbmcs_results = inputs.filter( { it =~ /scvi_pbmcs.*/ } ).collect()
    nucleus_results = inputs.filter( { it =~ /nucleus.*/ } ).collect()
    sciplex_results = inputs.filter( { it =~ /sciplex.*/ } ).collect()

    all_results = symsim_results.concat(
        pbmcs_results,
        nucleus_results,
        sciplex_results
    )
    all_results.view()

    conduct_generic_analysis(all_results)

    // Dataset-specific scripts can be added here
    if (symsim_results) {
        produce_figures_symsim_new(symsim_results)
    }
    if (sciplex_results) {
        produce_figures_sciplex(sciplex_results)
    }
}
