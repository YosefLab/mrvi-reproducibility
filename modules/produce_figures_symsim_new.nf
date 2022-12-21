


process produce_figures {
    input:
    tuple val(dataset_name), val(results_paths)

    script:
    if ( dataset_name == "symsim_new" ) {
        concatenated_paths = results_paths.join(" ")
        """
        python3 ${params.bin.produce_figures_symsim_new} --results_paths $concatenated_paths
        """
    }
    else {
        """
        echo "No analysis script for dataset $dataset_name"
        """
    }
}
