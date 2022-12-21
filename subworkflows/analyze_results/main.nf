include { produce_figures_symsim_new } from params.modules.produce_figures_symsim_new


workflow analyze_results {
    take:
    inputs

    main:
    symsim_results = inputs.filter( { it =~ /symsim_new.*/ } ).collect()
    symsim_results.view()
    produce_figures_symsim_new(symsim_results)

    // Dataset-specific scripts can be added here
}
