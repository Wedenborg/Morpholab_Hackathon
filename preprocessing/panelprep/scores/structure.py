"""Panel co-expression structure agreement (the brief's score (b)).

Requested formulation
---------------------
    "compute a correlation Subset by Subset for the gene expression data and a
    correlation for the candidate dataset matched according to the descriptor
    Subset by Subset and compute a Frobenius norm"

What is being compared
----------------------
Observation unit: one cell contributes one observation to a gene-gene
correlation computed *within* a dataset. The two ``S x S`` matrices are then
compared entry-wise. Genes are paired by symbol; cells are not paired.

Corrections applied (and why)
-----------------------------
1. **Drop the diagonal.** A correlation matrix has ones on the diagonal in both
   datasets, so the diagonal contributes exactly zero to the difference while
   inflating the matrix size. Only the strictly upper triangle is used.
2. **Normalise by panel size.** ``||A - B||_F`` grows like ``sqrt(S(S-1)/2)``,
   so the raw norm is not comparable across panels of different size. The
   reported value is the **root-mean-square correlation difference** over gene
   pairs (range ``[0, 2]``); the raw Frobenius norm stays in diagnostics.
3. **Spearman by default.** Log-CP10K single-cell expression is zero-inflated
   and strongly non-normal, so a Pearson correlation on four sparse genes is
   driven by a handful of high-expressing cells. Rank correlation is the
   defensible default; ``method="pearson"`` is available.
4. **Constant genes.** A gene with no variance in either dataset makes its
   correlations undefined. Such genes are dropped, listed in diagnostics, and
   if fewer than two genes survive the result is
   ``status="insufficient_genes"`` -- not a fabricated zero.
5. **Scale-free.** Correlation is invariant to per-gene affine rescaling, so
   this score is registered ``units_sensitive=False``: it is the one score here
   that survives a normalisation mismatch between platforms.

``corr_cosine`` is the companion: cosine similarity of the two upper-triangle
vectors. It isolates the *pattern* of co-expression from its magnitude, which
matters when one platform compresses correlations toward zero through dropout.

MODIFY HERE: ``method`` (``spearman``/``pearson``) and ``min_detected_frac``
(drop a gene detected in fewer than this fraction of cells in either dataset).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import CandidateDataset, PanelQuery, ScoreResult
from . import MIN_GENES_FOR_STRUCTURE, SCORES, prepare_panel


def _panel_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df.corr(method=method)


def _upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _prepare(query, candidate, genes, rng, max_obs, method, min_detected_frac):
    qp, cp, used, missing = prepare_panel(query, candidate, genes, max_obs=max_obs, rng=rng)
    if not used:
        return None, None, [], missing, {"reason": "no panel gene shared with the candidate"}
    dropped = {}
    keep = []
    for g in used:
        qv, cv = qp[g].to_numpy(float), cp[g].to_numpy(float)
        if np.nanstd(qv) == 0 or np.nanstd(cv) == 0:
            dropped[g] = "constant in query or candidate"
            continue
        if min_detected_frac > 0:
            dq, dc = float((qv > 0).mean()), float((cv > 0).mean())
            if dq < min_detected_frac or dc < min_detected_frac:
                dropped[g] = f"detected in {dq:.3f}/{dc:.3f} of cells, below {min_detected_frac}"
                continue
        keep.append(g)
    return qp[keep], cp[keep], keep, missing, dropped


def _structure_result(name, orientation, query, candidate, genes, rng, max_obs,
                      method, min_detected_frac, reducer):
    qp, cp, used, missing, dropped = _prepare(
        query, candidate, genes, rng, max_obs, method, min_detected_frac)
    base = dict(score_name=name, dataset_name=candidate.name, orientation=orientation,
                n_genes_requested=len(genes), n_genes_used=len(used),
                genes_missing=tuple(sorted(set(missing) | set(dropped))))
    if len(used) < MIN_GENES_FOR_STRUCTURE:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"dropped_genes": dropped,
                                        "reason": f"{len(used)} usable genes, need "
                                                  f"{MIN_GENES_FOR_STRUCTURE}"}, **base)
    A = _panel_corr(qp, method)
    B = _panel_corr(cp, method)
    a, b = _upper(A.to_numpy()), _upper(B.to_numpy())
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "undefined correlations after filtering",
                                        "dropped_genes": dropped}, **base)
    value, extra = reducer(a, b)
    return ScoreResult(
        value=float(value), status="ok",
        diagnostics={"method": method, "n_pairs": int(a.size),
                     "frobenius_raw": float(np.linalg.norm(a - b) * np.sqrt(2)),
                     "query_corr": A.round(4).to_dict(),
                     "candidate_corr": B.round(4).to_dict(),
                     "dropped_genes": dropped,
                     "n_obs": (int(qp.shape[0]), int(cp.shape[0])), **extra},
        **base)


@SCORES.register(
    "corr_frobenius", orientation="lower_is_better",
    applies_to=("single_cell", "spatial", "causal"), needs_coords=False, units_sensitive=False,
    defaults=dict(method="spearman", min_detected_frac=0.0),
    description="RMS difference between the query and candidate panel-gene correlation matrices "
                "(strictly upper triangle, Spearman by default); scale-free",
)
def corr_frobenius(query, candidate, *, genes, rng, method: str = "spearman",
                   min_detected_frac: float = 0.0, max_obs: Optional[int] = None,
                   **_) -> ScoreResult:
    """Panel correlation-matrix distance, normalised per gene pair."""
    def reducer(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))), {}

    return _structure_result("corr_frobenius", "lower_is_better", query, candidate, genes,
                             rng, max_obs, method, min_detected_frac, reducer)


@SCORES.register(
    "corr_cosine", orientation="higher_is_better",
    applies_to=("single_cell", "spatial", "causal"), needs_coords=False, units_sensitive=False,
    defaults=dict(method="spearman", min_detected_frac=0.0),
    description="cosine similarity of the two panel correlation vectors; isolates the pattern of "
                "co-expression from its magnitude (dropout compresses magnitude)",
)
def corr_cosine(query, candidate, *, genes, rng, method: str = "spearman",
                min_detected_frac: float = 0.0, max_obs: Optional[int] = None,
                **_) -> ScoreResult:
    """Cosine similarity between the upper-triangle correlation vectors."""
    def reducer(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return float("nan"), {"note": "a correlation vector is identically zero"}
        return float(np.dot(a, b) / (na * nb)), {"norm_query": float(na), "norm_candidate": float(nb)}

    return _structure_result("corr_cosine", "higher_is_better", query, candidate, genes,
                             rng, max_obs, method, min_detected_frac, reducer)
