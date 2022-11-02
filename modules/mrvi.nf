nextflow.enable.dsl=2

process fit_mrvi {
    input:
        val adata_path
        val save_model_path
        val save_adata_path
        val batch_key
        val sample_key

    output:
        val save_model_path
        val save_adata_path

    script:
        """
        python3 -m fit_mrvi.py
            ${anndata_path}
            --save_model_path ${save_model_path}
            --save_adata_path ${save_adata_path}
            --batch_key ${batch_key}
            --sample_key ${sample_key}
        """
}
