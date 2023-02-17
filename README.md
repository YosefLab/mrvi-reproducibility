# scvi-v2-reproducibility

## Running existing workflows

Existing workflows can be discovered under `workflows/`. These can be run as follows from
the root of the repository:

```
nextflow main.nf --workflow workflow_name --profile profile_name
```

Setting both `--workflow` and `--profile` is required. Available profiles include
`standard` (run without GPU) and `gpu` (run with GPU).

By default, intermediate and final outputs are placed in `results/`. This can be changed
by modifying the `publishDir` directive in the configuration.

### Simple pipeline

Currently, this pipeline just scans `data/` in the root directory for any `.h5ad` files
and runs each subworkflow sequentially. It expects a configuration JSON file to be
present in `conf/datasets/` with the same name as the `.h5ad` file.

### AWS pipeline

This pipeline pulls data from `s3://largedonor` and runs each subworkflow locally. In
order to run this pipeline, create the file `conf/aws_credentials.config` (ignored by
git) with the following entries:

```
aws {
    accessKey = "access_key"
    secretKey = "secret_key"
    region = "us-west-1"
}
```

You can specify the individual AnnDatas being processed by modifying `params.inputs` in
`conf/aws_pipeline.config`.

## Adding new workflows

Workflows are intended to connect subworkflows into an end-to-end pipeline. In order to
add a new workflow, follow these steps:

1. Add a `workflow_name.nf` file to `workflows/` with the following template:

```
include { subworkflow } from "${params.subworkflows.subworkflow_name}"

workflow run_main {
    main:
    inputs = Channel.fromPath(params.inputs)
    // subworkflows here
}
```

2. Create the associated `workflow_name.config` file in `conf/` with the following
   template:

```
params {
    inputs = //path or list of paths to inputs
}
```

## Adding new subworkflows

Subworkflows are intended to be reusable pieces across different workflows. To add a new
subworkflow, follow these steps:

1. Add a `main.nf` file to `subworkflows/subworkflow_name/` with the following template:

```
include { module } from "${params.modules.module_name}"

workflow subworkflow_name {
    take:
    inputs

    main:
    module(inputs)

    emit:
    module.out
}
```

2. Add a reference to the new subworkflow under `nextflow.config` to be able to import
   it as `params.subworkflows.subworkflow_name

```
params {
    subworkflow_name = "${params.subworkflows.root}/subworkflow_name/main.nf"
}
```

## Adding new modules and scripts

Modules are Nextflow wrappers over Python scripts that pass in appropriate arguments.
They can be found under `modules/` and `bin/`, respecively. To add a new module, follow
these steps:

1. Add a `module_name.nf` file to `modules/` with the following template:

```
process module_name {
    input:
    path input_path

    script:
    output_path = "output_path_here"
    """
    python3 ${params.bin.module_name} \\
        --input_path ${input_path} \\
        --output_path ${output_path}
    """

    output:
    path output_path
}
```

2. Add the corresponding Python script to `bin/` with the following template:

```
from utils import wrap_kwargs

@wrap_kwargs
def module_name(input_path, output_path):
    pass

if __name__ == "__main__":
    module_name()
```

3. Add references to the new module and script under `nextflow.config` to be able to
   import them as `params.modules.module_name` and `params.bin.module_name`, respectively.

```
params {
    modules {
        module_name = "${params.modules.root}/module_name.nf"
    }
    bin {
        module_name = "${params.bin.root}/module_name.py"
    }
}
```

4. Specify the conda environment for the new module in `nextflow.config`:

```
params {
    env {
        module_name = "${params.env.root}/module_name.yaml"
    }
}

process {
    withName: module_name {
        conda = "${params.env.module_name}"
    }
}
```
