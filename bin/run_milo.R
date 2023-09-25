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

file_name <- args[1]
config_file <- args[2]
save_analysis <- args[3]
save_nhoods <- args[4]
rep_key <- "pca.corrected"

file_str <- paste(readLines(config_file), collapse="\n")
config <- fromJSON(file_str)
sample_key <- config$sample_key
covariate_key <- config$covariate_key
nuisance_key <- config$batch_key
data <- readH5AD(file_name)

# %%
data <- logNormCounts(data, assay.type="X")
corrected_data <- fastMNN(
    data,
    batch = colData(data)[,nuisance_key],
    assay.type = "logcounts"
)
reducedDims(data) <- list(pca.corrected=reducedDim(corrected_data))
# %%
data <- data[,apply(reducedDim(data, rep_key), 1, function(x) !all(is.na(x)))]
milo_data <- Milo(data)
embryo_milo <- buildGraph(milo_data, k = 30, d = 30, reduced.dim = rep_key)
embryo_milo <- makeNhoods(embryo_milo, prop = 0.1, k = 30, d=30, refined = TRUE, reduced_dims = rep_key)

embryo_milo <- countCells(embryo_milo, meta.data = as.data.frame(colData(embryo_milo)), sample=sample_key)
embryo_design <- data.frame(colData(embryo_milo))[,c(sample_key, covariate_key, nuisance_key)]

# embryo_design$sequencing.batch <- as.factor(embryo_design$sequencing.batch)
embryo_design <- distinct(embryo_design)
rownames(embryo_design) <- embryo_design[,sample_key]

embryo_milo <- calcNhoodDistance(embryo_milo, d=30, reduced.dim = rep_key)

if (unique(colData(embryo_milo)[,nuisance_key]) == 1) {
    design <- formula(paste("~","factor(", covariate_key, ")"))
} else {
    design <- formula(paste("~","factor(", covariate_key, ")", "+", "factor(", nuisance_key, ")"))
}


da_results <- testNhoods(
    embryo_milo,
    design = design,
    design.df = embryo_design,
    reduced.dim = rep_key
)
write.table(
    da_results,
    file = save_analysis,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = TRUE
)

writeMM(obj = nhoods(embryo_milo), file=save_nhoods)
