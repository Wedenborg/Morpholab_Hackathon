"""Shared validation for the aligned gene axis."""

import numpy as np


def _gene_indices(sequenced_genes, n_genes):
    """Validate zero-based column indices while preserving their order."""
    indices = np.asarray(sequenced_genes)
    if indices.ndim != 1:
        raise ValueError("sequenced_genes must be a one-dimensional sequence.")
    if indices.size == 0:
        return np.empty(0, dtype=int)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sequenced_genes must contain integer column indices.")
    if np.any(indices < 0) or np.any(indices >= n_genes):
        raise ValueError("sequenced_genes contains an out-of-range column index.")
    if np.unique(indices).size != indices.size:
        raise ValueError("sequenced_genes must not contain duplicates.")
    return indices


