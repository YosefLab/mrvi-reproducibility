# scvi-v2-reproducibility

Nextflow workflows (found in `workflows/`) are wrappers over subworkflows and modules 
(found in `modules/` and `subworkflows/` respectively) that run end-to-end. The 
subworkflows and modules are designed to be reusable and can be run independently of the 
workflows. To run a workflow, the primary entry point is through `main.nf` in the 
project root:

```
nextflow main.nf --workflow workflow_name
```

A configuration file can be provided to the workflow by placing a `.config` file in 
`conf/` with the same name as the workflow. Parameters specified in this file can be 
accessed within any of the subworkflows or modules.

In addition to the Nextflow configuration file, dataset-level configurations can be 
provided in the `conf/datasets/` directory as JSON files, which can only be accessed
by Python scripts in `bin/`.

Compute profiles that specify the kind of hardware to use can be found in
`nextflow.config`. These profiles can be used by passing in the `-profile` flag when 
running a workflow, e.g.:

```
nextflow main.nf --workflow workflow_name --profile standard
```

which will not use any GPU resources in addition to installing non-GPU supported 
dependencies.

To start a new run with `simple_pipeline`, the following is required:
- Add a `.h5ad` file to `data/`
- Add a dataset configuration file to `conf/datasets/`, following the example
in `example.json`. The config file must have the same name as the `.h5ad` file.

Intermediate and final outputs will be placed in `results/simple_pipeline/`.

## AWS
In order to pull data from AWS, you must have the AWS CLI installed and configured. 
Additionally, create a file called `aws_creds.config` in the `conf/` directory with the
following entries. This file is ignored by git.
```
aws {
    accessKey = ""
    secretKey = ""
    region = ""
}
```
Right now, `aws_pipeline.nf` is configured to just pull `s3://largedonor/cancer.h5ad`.