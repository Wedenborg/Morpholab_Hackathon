"""Training data for the factor model, other than the visible embryo.

mdcla_cells              log-normalised E7.5 single cells from mdCLA (2026, 10x, other embryos)
mdcla_counts             the same cells as raw counts, with library sizes

The Pijuan-Sala atlas (and TOME / the Extended Mouse Atlas, which contain it) is the
benchmark's reference in eval_team, so the method must not train on it: a reconstruction
built from the atlas would match the atlas score by construction.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eval_team.paths import REF_DIR

from .model import aligned_matrix


MDCLA = REF_DIR / "mdcla" / "mdCLA_E7.5.h5ad"


def _mdcla():
    """mdCLA E7.5 'Early embryo' cells as log1p(counts per median library size)."""
    a = ad.read_h5ad(MDCLA)
    a = a[(a.obs["level1_group"] == "Early embryo").to_numpy()]
    X = sp.csr_matrix(a.X, dtype=np.float32)
    total = np.asarray(X.sum(1)).ravel()
    X = sp.diags(np.median(total) / total) @ X
    X.data = np.log1p(X.data)
    return ad.AnnData(X=X.tocsr(), obs=a.obs[["level4_subtypes"]].copy(), var=pd.DataFrame(index=a.var_names))


def mdcla_counts(genes):
    """(csr raw counts, cells x genes in `genes` order; library size per cell over all genes)."""
    a = ad.read_h5ad(MDCLA)
    a = a[(a.obs["level1_group"] == "Early embryo").to_numpy()]
    X = sp.csr_matrix(a.X, dtype=np.float32)
    total = np.asarray(X.sum(1)).ravel()
    return aligned_matrix(ad.AnnData(X=X, var=pd.DataFrame(index=a.var_names)), genes), total


def mdcla_cells(genes):
    """(csr cells x genes in `genes` order, cell subtype labels)."""
    a = _mdcla()
    return aligned_matrix(a, genes), a.obs["level4_subtypes"].astype(str).to_numpy()
