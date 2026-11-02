# Export the supplied legacy Seurat RNA counts without requiring Seurat.
# Usage: Rscript --vanilla convert_reference.R input.rds output_directory
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
    stop("Usage: Rscript --vanilla convert_reference.R input.rds output_directory")
}

object <- readRDS(args[[1]])
assays <- attr(object, "assays")
if (is.null(assays) || is.null(assays[["RNA"]])) {
    stop("Expected a legacy Seurat object containing an RNA assay.")
}
counts <- attr(assays[["RNA"]], "counts")
if (!inherits(counts, "sparseMatrix") || any(dim(counts) == 0L)) {
    stop("The RNA assay must contain a nonempty sparse counts matrix.")
}
if (is.null(rownames(counts)) || is.null(colnames(counts))) {
    stop("RNA counts must have gene and cell names.")
}

output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
writeMM(counts, file.path(output_dir, "counts.mtx"))
writeLines(rownames(counts), file.path(output_dir, "genes.txt"))
writeLines(colnames(counts), file.path(output_dir, "cells.txt"))
message(sprintf("Exported %d genes x %d cells (%d nonzero counts).",
                nrow(counts), ncol(counts), length(counts@x)))
