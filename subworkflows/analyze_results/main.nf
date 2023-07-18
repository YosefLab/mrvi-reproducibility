include { produce_figures_symsim_new } from params.modules.produce_figures_symsim_new
include { produce_figures_sciplex } from params.modules.produce_figures_sciplex
include { conduct_generic_analysis } from params.modules.conduct_generic_analysis

workflow analyze_results {
    take:
    inputs

    main:
    all_results = inputs.map{ [it, it.getSimpleName()] }.groupTuple(by: 1).map { it[0] }
    all_results.view()

    conduct_generic_analysis(all_results)

    // Dataset-specific scripts can be added here
    symsim_results = inputs.filter( { it =~ /symsim_new.*/ } ).collect()

    if (symsim_results) {
        produce_figures_symsim_new(symsim_results)
    }
}
