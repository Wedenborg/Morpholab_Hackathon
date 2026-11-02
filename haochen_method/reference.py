"""Reference loading and HVG scoring; no hidden target expression is accepted."""

from pathlib import Path
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io, sparse


def read_reference(path, cache_dir=None):
    """Read h5ad or the existing legacy Seurat RDS/counts cache.

    For RDS, a new cache must be used when changing the source. Existing caches
    remain compatible with the former notebook. First conversion requires R
    and Matrix; subsequent reads use counts.npz directly.
    """
    path = Path(path)
    if path.suffix.lower() == ".h5ad":
        return ad.read_h5ad(path)
    if path.suffix.lower() != ".rds":
        raise ValueError("Reference must be .h5ad or .rds.")
    cache = Path(cache_dir) if cache_dir is not None else path.parent / (path.stem + "_cache")
    counts_path, genes_path, cells_path = [cache / n for n in ("counts.npz", "genes.txt", "cells.txt")]
    mtx_path = cache / "counts.mtx"
    if not (counts_path.exists() and genes_path.exists() and cells_path.exists()):
        if not (mtx_path.exists() and genes_path.exists() and cells_path.exists()):
            rscript = shutil.which("Rscript")
            if rscript is None:
                raise RuntimeError("First-time RDS conversion requires Rscript and Matrix.")
            cache.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                rscript, "--vanilla", str(Path(__file__).with_name("convert_reference.R")),
                str(path), str(cache),
            ], check=True)
        counts = io.mmread(mtx_path).T.tocsr()
        sparse.save_npz(counts_path, counts)
        mtx_path.unlink()
    else:
        counts = sparse.load_npz(counts_path)
    genes = pd.Index(genes_path.read_text().splitlines())
    cells = pd.Index(cells_path.read_text().splitlines())
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("Reference cache dimensions disagree with its gene/cell names.")
    result = ad.AnnData(X=counts, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    result.uns["expression_source"] = "counts"
    return result


def map_reference_gene_symbols(reference, annotation_path, counts_layer="counts"):
    """Map mouse Ensembl IDs using the local MGI report and sum duplicate counts.

    Ambiguous/unmapped IDs are excluded. This operates on raw counts, because
    summing already logged expression would change its meaning. Symbol-based
    references can bypass this function entirely.
    """
    if counts_layer in reference.layers:
        counts = reference.layers[counts_layer]
    elif reference.uns.get("expression_source") == "counts":
        counts = reference.X
    else:
        raise ValueError("Ensembl mapping requires a raw counts layer or an RDS counts reference.")
    report = pd.read_csv(annotation_path, sep="\t", header=None, dtype=str, keep_default_na=False)
    annotation = report.iloc[:, [1, 5]].copy()
    annotation.columns = ["gene_symbol", "ensembl_id"]
    annotation["ensembl_id"] = annotation["ensembl_id"].str.split()
    annotation = annotation.explode("ensembl_id").dropna().drop_duplicates()
    annotation = annotation[
        annotation["gene_symbol"].ne("") & annotation["ensembl_id"].str.startswith("ENSMUSG")
    ]
    ambiguous = annotation.groupby("ensembl_id")["gene_symbol"].nunique().gt(1)
    annotation = annotation[~annotation["ensembl_id"].isin(ambiguous[ambiguous].index)]
    lookup = annotation.drop_duplicates("ensembl_id").set_index("ensembl_id")["gene_symbol"]
    symbols = reference.var_names.str.replace(r"\.\d+$", "", regex=True).map(lookup)
    keep = symbols.notna()
    if not keep.any():
        raise ValueError("No reference Ensembl IDs map to gene symbols.")
    codes, unique_symbols = pd.factorize(symbols[keep], sort=False)
    aggregation = sparse.csr_matrix(
        (np.ones(keep.sum(), dtype=np.float32), (np.arange(keep.sum()), codes)),
        shape=(keep.sum(), len(unique_symbols)),
    )
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Reference counts must be finite and nonnegative.")
    mapped = ad.AnnData(
        X=(counts[:, keep] @ aggregation).tocsr(),
        obs=reference.obs.copy(), var=pd.DataFrame(index=pd.Index(unique_symbols)),
    )
    mapped.uns["expression_source"] = "counts"
    return mapped


def prepare_expression(adata, *, scale="auto", counts_layer="counts", normalize_total=True, target_sum=1e4):
    """Return an independent AnnData with finite nonnegative natural-log1p X.

    auto: use the counts layer, an explicit counts source tag, or uns['log1p'].
    counts: use counts_layer when present, otherwise X; normalize then log once.
    log1p: explicitly confirm X is already logged; do not log/normalize it again.
    Values alone cannot prove a log transform. Without counts or metadata auto
    raises, requiring the caller to confirm the input scale. For prelogged data
    the caller must ensure comparable library normalization in both domains.
    """
    if scale not in {"auto", "counts", "log1p"}:
        raise ValueError("scale must be 'auto', 'counts', or 'log1p'.")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive.")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Cell and gene names must be unique.")
    if scale == "auto":
        if counts_layer in adata.layers or adata.uns.get("expression_source") == "counts":
            scale = "counts"
        elif "log1p" in adata.uns:
            scale = "log1p"
        else:
            raise ValueError("Unknown expression scale: provide counts or explicitly set scale='counts'/'log1p'.")
    values = adata.layers[counts_layer] if scale == "counts" and counts_layer in adata.layers else adata.X
    matrix = sparse.csr_matrix(values, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape[1] == 0 or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Expression must have genes and contain finite, nonnegative values.")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    keep = totals > 0
    if keep.sum() < 2:
        raise ValueError("Need at least two nonempty cells per dataset.")
    processed = adata[keep].copy()
    processed.X = matrix[keep].copy()
    processed.var["n_cells_by_counts"] = matrix[keep].getnnz(axis=0)
    if scale == "counts":
        processed.layers["counts"] = processed.X.copy()
        processed.obs["total_counts"] = totals[keep]
        processed.obs["n_genes_by_counts"] = processed.X.getnnz(axis=1)
        processed.obs["size_factor"] = totals[keep] / target_sum if normalize_total else 1.0
        processed.uns.pop("log1p", None)
        if normalize_total:
            sc.pp.normalize_total(processed, target_sum=target_sum)
        sc.pp.log1p(processed)
    else:
        base = processed.uns.get("log1p", {}).get("base")
        if base is not None:
            if not np.isfinite(base) or base <= 1:
                raise ValueError("Log base must be finite and greater than one.")
            processed.X.data *= np.log(base)
        processed.uns["log1p"] = {"base": None}
    processed.uns["preprocessing"] = {
        "source_scale": scale, "transform": "log1p", "log_base": "natural",
        "normalization": ("normalize_total" if normalize_total else "none") if scale == "counts" else "input",
        "target_sum": float(target_sum) if scale == "counts" and normalize_total else None,
        "dropped_zero_cells": int((~keep).sum()),
    }
    return processed


def rank_sc_hvgs(reference, n_top_genes=2000, min_cells=10):
    """Rank shared genes by Seurat normalized dispersion using reference only.

    The reference must already be log1p transformed. Return global gene indices
    (strongest first) and a table in the aligned gene order. If fewer eligible
    finite scores exist, return fewer than K; ties preserve aligned gene order.
    """
    for name, value in [("n_top_genes", n_top_genes), ("min_cells", min_cells)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if "log1p" not in reference.uns:
        raise ValueError("HVG scoring requires confirmed log1p reference expression.")
    counts = reference.layers.get("counts", reference.X)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    eligible = detected >= min_cells
    if eligible.sum() < 2:
        raise ValueError("Fewer than two genes pass min_cells; lower hvg_min_cells.")
    hvg = reference[:, eligible].copy()
    # n_top_genes=None computes scores for all genes; sorting enforces exact K
    # despite Scanpy's threshold ties at the cutoff.
    sc.pp.highly_variable_genes(hvg, flavor="seurat", n_top_genes=None)
    scores = hvg.var["dispersions_norm"]
    selected = scores[np.isfinite(scores)].sort_values(ascending=False, kind="stable").head(n_top_genes)
    if selected.empty:
        raise ValueError("No finite SC HVG scores; check reference gene variation.")
    table = pd.DataFrame(index=reference.var_names.copy())
    table.index.name = "gene"
    table["sc_detected_cells"] = detected
    table["eligible_sc"] = eligible
    table["hvg_score_sc"] = scores.reindex(table.index)
    table["highly_variable_sc"] = table.index.isin(selected.index)
    table["hvg_rank_sc"] = np.nan
    table.loc[selected.index, "hvg_rank_sc"] = np.arange(1, len(selected) + 1)
    return reference.var_names.get_indexer(selected.index), table
