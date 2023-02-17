process conduct_generic_analysis {
    cache false
    input:
    path results_paths

    script:
    concatenated_paths = results_paths.join(" ")
    dataset_name = results_paths[0].getSimpleName()
    config_in = "${params.conf.datasets}/${dataset_name}.json"
    output_dir = "${params.outputs.figures}/${dataset_name}"

    """
    python3 ${params.bin.conduct_generic_analysis} \\
    --results_paths $concatenated_paths \\
    --output_dir $output_dir \\
    --config_in $config_in
    """

    output:
    path "${output_dir}/*"
}
