library(dplyr)
library(magrittr)
library(tidyr)
library(reshape2)

# From https://github.com/cole-trapnell-lab/sci-plex/blob/079639c50811dd43a206a779ab2f0199a147c98f/large_screen/Notebook4_Figure3.R
sciPlex_cds.list = list()
sciPlex_cds.list[["A549"]] = readRDS("data/archive/A549.RDS")
sciPlex_cds.list[["K562"]] = readRDS("data/archive/K562.RDS")
sciPlex_cds.list[["MCF7"]] = readRDS("data/archive/MCF7.RDS")

product_cluster_mat.list <- list()
cluster.enrichment.df = list()

for (cell_line in names(sciPlex_cds.list)) {
  temp = attr(sciPlex_cds.list[[cell_line]], "colData") %>%
      as.data.frame() %>%
      dplyr::mutate(product_dose = paste0(product_name, "_", dose)) %>%
      dplyr::select(product_dose, Cluster)
  product_cluster_mat.list[[cell_line]] =
    reshape2::acast(
      table(temp$product_dose, temp$Cluster) %>%
      as.data.frame(),
      Var1 ~ Var2,
      value.var = "Freq"
    )

  weighted.mat = product_cluster_mat.list[[cell_line]]
  ntc.counts = product_cluster_mat.list[[cell_line]]["Vehicle_0", ]

  cluster.enrichment.df[[cell_line]] = do.call(rbind, lapply(rownames(weighted.mat), function(product_dose) {
    do.call(rbind, lapply(1:ncol(weighted.mat), function(Cluster) {
      test = fisher.test(cbind(c(weighted.mat[product_dose, Cluster], sum(
        weighted.mat[product_dose,-Cluster]
      )),
      c(ntc.counts[Cluster], sum(ntc.counts[-Cluster]))))

      data.frame(
        product_dose = product_dose,
        Cluster = Cluster,
        odds.ratio = unname(test$estimate),
        p.value = test$p.value
      )
    }))
  }))

  cluster.enrichment.df[[cell_line]]$q.value = p.adjust(cluster.enrichment.df[[cell_line]]$p.value, "BH")

  cluster.enrichment.df[[cell_line]]$log2.odds = with(cluster.enrichment.df[[cell_line]],
                                                      ifelse(odds.ratio == 0,-5, round(log2(odds.ratio), 2)))

  cluster.enrichment.df[[cell_line]]$product_name <-
    sapply(cluster.enrichment.df[[cell_line]]$product_dose, function(x) {
      stringr::str_split(x, pattern = "_")[[1]][1]
    })

  cluster.enrichment.df[[cell_line]]$dose <-
    sapply(cluster.enrichment.df[[cell_line]]$product_dose, function(x) {
      stringr::str_split(x, pattern = "_")[[1]][2]
    })

}



significant_product_dose_combinations.list <- list()

for (cell_line in names(sciPlex_cds.list)) {
  significant_product_dose_combinations.list[[cell_line]] <-
    (
      cluster.enrichment.df[[cell_line]] %>%
        filter(q.value < 0.01 &
                 log2.odds > 2.5) %>%
        distinct(product_dose)
    )$product_dose

  print(length(significant_product_dose_combinations.list[[cell_line]]))

}

# Save to CSV
for (cell_line in names(sciPlex_cds.list)) {
  write.table(significant_product_dose_combinations.list[[cell_line]], file = paste("notebooks/output/", cell_line, ".csv", sep = ""), row.names = FALSE, col.names = FALSE, sep=",")
}
