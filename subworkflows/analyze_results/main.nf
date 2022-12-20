include { produce_figures } from params.modules.produce_figures


workflow analyze_results {
    take:
    inputs

    main:
    produce_figures(inputs)
}
