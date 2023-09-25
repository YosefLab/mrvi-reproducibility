#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
print(args)
# %%
library(miloR)
library(SingleCellExperiment)
library(scater)
library(scran)
library(dplyr)
library(patchwork)
library(zellkonverter)
library(batchelor)
library(Matrix)
library(rjson)
library(devtools)
devtools::install_github("MarioniLab/miloDE")
library(miloDE)


file_name <- args[1]
config_file <- args[2]
save_analysis <- args[3]
save_nhoods <- args[4]
rep_key <- "pca.corrected"

file_str <- paste(readLines(config_file), collapse="\n")
config <- fromJSON(file_str)
sample_key <- config$sample_key
covariate_key <- config$covariate_key_de
nuisance_key <- config$batch_key
data <- readH5AD(file_name)

data <- logNormCounts(data, assay.type="X")
corrected_data <- fastMNN(
    data,
    batch = colData(data)[,nuisance_key],
    assay.type = "logcounts"
)
reducedDims(data) <- list(pca.corrected=reducedDim(corrected_data))


counts(data) <- assay(data, "X")
data = assign_neighbourhoods(
    data,
    k = 20,
    order = 2,
    filtering = TRUE,
    reducedDim_name = "pca.corrected"
)

if (unique(colData(data)[,nuisance_key]) == 1) {
    design <- formula(paste("~","factor(", covariate_key, ")"))
    covariates <- c(covariate_key)
} else {
    design <- formula(paste("~","factor(", covariate_key, ")", "+", "factor(", nuisance_key, ")"))
    covariates <- c(covariate_key, nuisance_key)
}

de_stat = de_test_neighbourhoods(
    data,
    sample_id = sample_key,
    design = design,
    covariates = covariates,
)

write.table(
    de_stat,
    file = save_analysis,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = TRUE
)
writeMM(obj = nhoods(data), file=save_nhoods)



