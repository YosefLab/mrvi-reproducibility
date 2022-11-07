# scvi-v2-reproducibility

Nextflow workflows (found in `workflows/`) are wrappers over subworkflows and modules 
(found in `modules/` and `subworkflows/` respectively) that run end-to-end. The 
subworkflows and modules are designed to be reusable and can be run independently of the 
workflows. To run a workflow, the primary entry point is through `main.nf` in the 
project root:

```
nextflow main.nf --workflow workflow_name
```

A global configuration `.config` file can be provided with the `-params-file` or `-c` 
flags, or by placing the file in `conf/` with the same name as the workflow. Parameters
specified in the configuration can be accessed within any of the subworkflows or modules.

In addition to the Nextflow configuration file, dataset-level configurations can be 
provided in the `conf/datasets/` directory as JSON files, which can only be accessed
by Python scripts in `bin/`.

To start a new run with `simple_pipeline`, the following is required:
- Add a `.h5ad` file to `data/raw`
- Add a dataset configuration file to `conf/datasets`, following the example
in `example.json`. The config file must have the same name as the `.h5ad` file.
