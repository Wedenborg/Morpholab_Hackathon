"""Per-gene empirical-distribution agreement (the brief's score (a)).

Requested formulation
---------------------
    "looking at the empirical distribution of each gene in the set, compute the
    KL divergence with the corresponding one in the set, take the average of KL
    scores"

What is being compared
----------------------
Observation unit: one cell (or spot) contributes one sample of a gene's
expression. For each panel gene we build the empirical distribution of that
gene **over cells within a dataset**, and compare the query's distribution to
the candidate's. Cells are *not* paired across datasets and no score here
assumes they are; only the gene identity is paired.

Corrections applied (and why)
-----------------------------
1. **Shared support.** KL between two continuous samples is not defined without
   a common partition. We bin both samples on a *shared* edge set derived from
   the pooled values (``bins="quantile"`` by default, so bins are populated
   where the data actually is). Binning each sample on its own quantiles would
   make every pair of datasets look identical.
2. **Zero-mass blow-up.** ``KL(p||q)`` is infinite wherever ``q`` has no mass
   and ``p`` does. We add Laplace smoothing ``alpha`` (a fraction of total
   mass). The number is therefore smoothing-dependent, and
   ``diagnostics["frac_unsupported"]`` reports the pre-smoothing query mass
   that sat on candidate-empty bins -- read it before trusting the magnitude.
3. **Asymmetry.** ``KL(query||candidate) != KL(candidate||query)``. ``direction``
   selects forward / reverse / symmetrised. Because the asymmetry has no
   principled justification here (neither dataset is "the model"), the
   recommended default score is **avg_js**: Jensen-Shannon divergence is
   symmetric, bounded in ``[0, ln 2]``, and finite without smoothing.
4. **Normalisation sensitivity.** These scores compare values, not ranks, so
   they are only meaningful when both datasets are on the same scale. Both are
   registered ``units_sensitive=True``; the runner refuses mismatched units
   unless ``Config.allow_unit_mismatch`` is set.
5. **Zero inflation.** In sparse data the shared zero bin can carry most of the
   mass and dominate the average. ``drop_zero_bin=True`` recomputes on detected
   cells only; per-gene values are always in ``diagnostics`` so the average is
   auditable.

Known limitation, stated rather than hidden: a platform difference in detection
rate (e.g. Stereo-seq vs MARS-seq) shifts every gene's distribution and will
dominate these scores. They answer "are the marginals the same", which is not
the same question as "is the co-expression structure the same" -- that is score
(b) in ``structure.py``.

MODIFY HERE
-----------
``n_bins``, ``bins``, ``alpha``, ``direction``, ``drop_zero_bin`` are all
parameters; set them per score in ``Config.score_params``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import CandidateDataset, PanelQuery, ScoreResult
from . import SCORES, prepare_panel

_EPS = 1e-300


def _shared_edges(a: np.ndarray, b: np.ndarray, n_bins: int, how: str) -> np.ndarray:
    """Bin edges built from the pooled sample, so both histograms share support."""
    pooled = np.concatenate([a, b])
    pooled = pooled[np.isfinite(pooled)]
    lo, hi = float(pooled.min()), float(pooled.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo - 0.5, lo + 0.5])
    if how == "fixed":
        return np.linspace(lo, hi, n_bins + 1)
    if how != "quantile":
        raise ValueError("bins must be 'quantile' or 'fixed'")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pooled, qs))
    if edges.size < 2:
        return np.array([lo - 0.5, hi + 0.5])
    edges[0], edges[-1] = lo, hi
    # nudge the outer edges so the extreme values fall inside the histogram
    span = max(hi - lo, 1e-12)
    edges[0] -= 1e-9 * span
    edges[-1] += 1e-9 * span
    return edges


def _hist(x: np.ndarray, edges: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed probability vector, raw probability vector)."""
    counts, _ = np.histogram(x[np.isfinite(x)], bins=edges)
    total = counts.sum()
    raw = counts / total if total else np.full(counts.shape, 1.0 / counts.size)
    smoothed = (counts + alpha * max(total, 1) / counts.size)
    smoothed = smoothed / smoothed.sum()
    return smoothed, raw


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], _EPS)))))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats, bounded by ln 2."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _divergence_score(
    query: PanelQuery, candidate: CandidateDataset, *, genes: Sequence[str], rng,
    kind: str, n_bins: int = 30, bins: str = "quantile", alpha: float = 1e-3,
    direction: str = "forward", drop_zero_bin: bool = False,
    max_obs: Optional[int] = None, **_,
) -> ScoreResult:
    name = "avg_kl" if kind == "kl" else "avg_js"
    qp, cp, used, missing = prepare_panel(query, candidate, genes, max_obs=max_obs, rng=rng)
    base = dict(score_name=name, dataset_name=candidate.name,
                orientation="lower_is_better", n_genes_requested=len(genes),
                n_genes_used=len(used), genes_missing=tuple(missing))
    if not used:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "no panel gene shared with the candidate"}, **base)

    per_gene, unsupported, n_bins_used = {}, {}, {}
    for g in used:
        a = qp[g].to_numpy(dtype=float)
        b = cp[g].to_numpy(dtype=float)
        if drop_zero_bin:
            a, b = a[a > 0], b[b > 0]
            if a.size < 10 or b.size < 10:
                per_gene[g] = np.nan
                unsupported[g] = np.nan
                n_bins_used[g] = 0
                continue
        edges = _shared_edges(a, b, n_bins, bins)
        p, p_raw = _hist(a, edges, alpha)
        q, q_raw = _hist(b, edges, alpha)
        unsupported[g] = float(p_raw[q_raw == 0].sum())
        n_bins_used[g] = int(edges.size - 1)
        if kind == "js":
            per_gene[g] = _js(p, q)
        elif direction == "forward":
            per_gene[g] = _kl(p, q)
        elif direction == "reverse":
            per_gene[g] = _kl(q, p)
        elif direction == "symmetrised":
            per_gene[g] = 0.5 * (_kl(p, q) + _kl(q, p))
        else:
            raise ValueError("direction must be forward, reverse or symmetrised")

    vals = np.array([per_gene[g] for g in used], dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "every gene was degenerate after filtering",
                                        "per_gene": per_gene}, **base)
    return ScoreResult(
        value=float(finite.mean()), status="ok",
        diagnostics={"per_gene": {g: (None if not np.isfinite(per_gene[g]) else float(per_gene[g]))
                                  for g in used},
                     "frac_unsupported": {g: unsupported[g] for g in used},
                     "n_bins": n_bins_used, "alpha": alpha, "bins": bins,
                     "direction": direction if kind == "kl" else "symmetric",
                     "drop_zero_bin": drop_zero_bin,
                     "n_obs": (int(qp.shape[0]), int(cp.shape[0])),
                     "n_genes_degenerate": int(vals.size - finite.size),
                     "units": "nats"},
        **base)


@SCORES.register(
    "avg_kl", orientation="lower_is_better",
    applies_to=("single_cell", "spatial", "causal"), needs_coords=False, units_sensitive=True,
    defaults=dict(n_bins=30, bins="quantile", alpha=1e-3, direction="forward"),
    description="mean over panel genes of KL(query||candidate) between shared-support histograms "
                "of per-cell expression; asymmetric and smoothing-dependent",
)
def avg_kl(query, candidate, *, genes, rng, **params) -> ScoreResult:
    """Average per-gene KL divergence. See the module docstring for the caveats."""
    return _divergence_score(query, candidate, genes=genes, rng=rng, kind="kl", **params)


@SCORES.register(
    "avg_js", orientation="lower_is_better",
    applies_to=("single_cell", "spatial", "causal"), needs_coords=False, units_sensitive=True,
    defaults=dict(n_bins=30, bins="quantile", alpha=1e-3),
    description="mean over panel genes of Jensen-Shannon divergence between shared-support "
                "histograms of per-cell expression; symmetric, bounded by ln 2 (recommended "
                "default over avg_kl)",
)
def avg_js(query, candidate, *, genes, rng, **params) -> ScoreResult:
    """Average per-gene Jensen-Shannon divergence -- the symmetric, bounded form."""
    params.pop("direction", None)
    return _divergence_score(query, candidate, genes=genes, rng=rng, kind="js", **params)
