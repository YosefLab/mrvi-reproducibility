process analyze_results_symsim_new {
    input:
    path vendi_

    script:
    """
    python3 analyze_results_symsim_new.py
    """
}
