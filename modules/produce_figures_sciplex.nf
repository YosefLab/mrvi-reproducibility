process produce_figures_sciplex {
    input:
    path results_paths

    script:
    concatenated_paths = results_paths.join(" ")

    """
    python3 ${params.bin.produce_figures_sciplex} \\
    --results_paths $concatenated_paths \\
    --output_dir ${params.outputs.figures}
    """

    output:
    path "${params.outputs.figures}/**/*.svg"
}
