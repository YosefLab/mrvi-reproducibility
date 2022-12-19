include { analyze_results_symsim_new } from params.modules.analyze_results_symsim_new


process analyze_results {
    input:
    path scib_outs
    path vendi_outs
    path adatas

    script:
    dataset_name = "${params.outputs.data}"
    if ( dataset_name == "symsim" ) {
        analyze_results_symsim_new(scib_outs, vendi_outs, adatas)
    }
    else {

    }
}
