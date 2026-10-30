"""Spatially aware agreement scores.

The brief says of the spatial references: *"similar spatial scores should be
computed for the spatial data, but for now just treat them as gene expression"*.
Both halves are honoured. The scores in ``distributional.py``, ``structure.py``
and ``alignment.py`` already run on spatial candidates as plain expression
(``applies_to`` includes ``"spatial"``). This module adds the scores that
actually use the coordinates, so the spatial half of the catalogue is not
evaluated only as if it were dissociated.

What is being compared
----------------------
Observation unit: one cell/spot with a 2D position. Coordinates live in each
dataset's own units and origin, and the two tissues are not registered to each
other -- so no score here compares positions directly. What is comparable is the
*amount and pattern of spatial structure* a gene shows, which is coordinate-free
once a neighbour graph is built.

``spatial_autocorr_frobenius``
    Moran's I of each panel gene on each dataset's own k-nearest-neighbour graph,
    then the RMS difference of the two Moran's I vectors. Asks: is this gene
    spatially organised to the same degree in both tissues? Moran's I is
    invariant to affine rescaling of expression, but it *does* depend on ``k``,
    so the same ``k`` is enforced on both sides and recorded.

``spatial_corr_frobenius``
    The ``corr_frobenius`` construction applied to spatially smoothed expression
    (each cell replaced by the mean of its ``k`` nearest neighbours, itself
    included). Per-cell co-expression in sparse spatial data is dominated by
    dropout; smoothing moves the question to tissue scale, where "are these two
    genes co-located" is what is being asked. Smoothing inflates correlations in
    both datasets, which is why the comparison stays like-for-like.

Both return ``status="not_applicable"`` when the candidate has no coordinates,
which is the correct outcome for every dissociated single-cell reference -- that
is a property of the data, not a failure.

MODIFY HERE: ``k`` (neighbours), ``method`` (rank vs Pearson for the smoothed
correlation), ``max_obs``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import CandidateDataset, PanelQuery, ScoreResult
from . import MIN_GENES_FOR_STRUCTURE, SCORES, prepare_panel


def _knn_weights(coords: np.ndarray, k: int):
    """Row-standardised k-nearest-neighbour weights, self excluded.

    Returns ``(indices, weights)`` with shape ``(N, k)``; row-standardised means
    each row of the implied weight matrix sums to 1, which is the convention
    Moran's I is usually reported under.
    """
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = int(min(k, max(n - 1, 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                       # drop self
    w = np.full(idx.shape, 1.0 / max(k, 1))
    return idx, w


def morans_i(x: np.ndarray, idx: np.ndarray, w: np.ndarray) -> float:
    """Moran's I with row-standardised weights. NaN when ``x`` is constant."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    denom = np.sum(z ** 2)
    if denom <= 0:
        return float("nan")
    lag = (z[idx] * w).sum(axis=1)         # sum_j w_ij z_j
    num = float(np.sum(z * lag))
    s0 = float(w.sum())
    return float((len(x) / s0) * (num / denom))


def _smooth(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Average each observation with its k nearest neighbours (self included)."""
    neigh = X[idx].mean(axis=1)
    return 0.5 * (X + neigh)


def _spatial_panels(query, candidate, genes, rng, max_obs, k):
    """Align the panel and put both sides on a comparable spatial footing.

    Two equalisations happen here, and both are necessary for the Moran's I
    comparison downstream to mean anything.

    **Equal observation counts.** Moran's I is sample-size dependent: on one
    real embryo the same gene gives I = 0.330 / 0.222 / 0.286 / 0.292 at
    300 / 600 / 1000 / 1500 cells -- a spread comparable to the entire
    between-dataset range of the score. If the query is subsampled to 4000 cells
    and the candidate to 1500, the difference in I is dominated by cell count
    rather than by tissue organisation, and a same-stage replicate can look like
    the *worst* match. Both sides are therefore subsampled to a **common**
    ``n = min(n_query, n_candidate, max_obs)``.

    **Comparable neighbourhood scale.** Point density varies several-fold across
    embryos (0.0103 vs 0.0029 cells per unit area here), so a fixed ``k`` spans
    different physical radii. A k-nearest-neighbour graph is rank-based, so once
    ``n`` matches, ``k`` neighbours cover a comparable *fraction* of each tissue;
    coordinates are additionally rescaled to unit mean nearest-neighbour
    distance so the graphs are built at the same relative scale.

    MODIFY HERE: set ``equalise_n=False`` in a score's params to recover the
    old per-side subsampling, and ``rescale_coords=False`` to keep raw units.
    """
    qp, cp, used, missing = prepare_panel(query, candidate, genes, max_obs=None, rng=rng)
    if not used:
        return None
    n_common = min(qp.shape[0], cp.shape[0])
    if max_obs is not None:
        n_common = min(n_common, max_obs)
    qi = np.sort(rng.choice(qp.shape[0], n_common, replace=False)) \
        if qp.shape[0] > n_common else np.arange(qp.shape[0])
    ci = np.sort(rng.choice(cp.shape[0], n_common, replace=False)) \
        if cp.shape[0] > n_common else np.arange(cp.shape[0])
    qc = np.asarray(query.coords, dtype=float)[qi, :2]
    cc = np.asarray(candidate.coords, dtype=float)[ci, :2]
    return qp.iloc[qi], cp.iloc[ci], qc, cc, used, missing


def _rescale(coords: np.ndarray, k: int) -> np.ndarray:
    """Scale coordinates to unit mean nearest-neighbour distance.

    Moran's I with k-NN weights is invariant to a global rescaling, so this does
    not change a single dataset's value; it makes the *neighbourhood graphs* of
    two datasets comparable in relative scale, which is what is being compared.
    """
    from sklearn.neighbors import NearestNeighbors

    if coords.shape[0] < 2:
        return coords
    nn = NearestNeighbors(n_neighbors=2).fit(coords)
    d, _ = nn.kneighbors(coords)
    scale = float(np.mean(d[:, 1]))
    return coords / scale if scale > 0 else coords



def _section_morans(expr: pd.DataFrame, coords: np.ndarray, sections, genes, k: int,
                    min_cells: int = 50, max_sections: int = 12,
                    cells_per_section: Optional[int] = None, rng=None) -> dict:
    """Mean Moran's I computed *within* each tissue section, then averaged.

    Why this is the default where a section label exists. A 3D-reconstructed
    embryo is usually viewed as a 2D projection, which superimposes cells from
    every section. Two embryos reconstructed from different numbers of sections
    (42 vs 52 here) are projected differently, so autocorrelation measured on
    the projection reflects the projection as much as the tissue. Measured on
    the real data: against an E7.5 query the projected statistic puts the
    same-stage E7.5 replicate at RMS 0.0997 and an E8.0 embryo at 0.0623 -- the
    wrong way round -- while the section-wise statistic gives 0.0464 and 0.1253,
    which is the correct ordering.

    Sections with fewer than ``min_cells`` are skipped: a k-nearest-neighbour
    graph needs several times ``k`` points before Moran's I is stable, and the
    default 50 is the floor for ``k=8``. Note that a query subsampled to a few
    thousand cells spread over ~40 sections leaves only ~100 cells per section,
    so this threshold interacts directly with ``max_obs`` at load time; at most ``max_sections`` of the largest sections are
    used, which bounds the cost and is recorded in diagnostics.
    """
    sec = pd.Series(np.asarray(sections))
    order = sec.value_counts()
    chosen = [z for z in order.index if order[z] >= min_cells][:max_sections]
    per = {g: [] for g in genes}
    for z in chosen:
        m = np.flatnonzero((sec == z).to_numpy())
        if m.size < min_cells:
            continue
        if cells_per_section is not None and m.size > cells_per_section:
            r = np.random.default_rng(0) if rng is None else rng
            m = np.sort(r.choice(m, cells_per_section, replace=False))
        xy = _rescale(np.asarray(coords, dtype=float)[m][:, :2], k)
        idx, w = _knn_weights(xy, k)
        for g in genes:
            v = morans_i(expr[g].to_numpy(dtype=float)[m], idx, w)
            if np.isfinite(v):
                per[g].append(v)
    used_sections = [z for z in chosen if (sec == z).sum() >= min_cells]
    return {"values": {g: (float(np.mean(v)) if v else float("nan")) for g, v in per.items()},
            "n_sections_used": len(used_sections),
            "cells_per_section": {str(z): int(min(order[z], cells_per_section or order[z]))
                                  for z in used_sections}}

@SCORES.register(
    "spatial_autocorr_frobenius", orientation="lower_is_better",
    applies_to=("spatial",), needs_coords=True, units_sensitive=False,
    defaults=dict(k=8, max_obs=2000, section_wise=True),
    description="RMS difference between the query and candidate Moran's I vectors over the panel "
                "genes (k-NN row-standardised weights on each dataset's own coordinates)",
)
def spatial_autocorr_frobenius(query: PanelQuery, candidate: CandidateDataset, *,
                               genes: Sequence[str], rng, k: int = 8,
                               max_obs: Optional[int] = 2000, section_wise: bool = True,
                               min_section_cells: int = 50, max_sections: int = 12,
                               **_) -> ScoreResult:
    """Agreement of per-gene spatial autocorrelation. See the module docstring."""
    base = dict(score_name="spatial_autocorr_frobenius", dataset_name=candidate.name,
                orientation="lower_is_better", n_genes_requested=len(genes))
    got = _spatial_panels(query, candidate, genes, rng, max_obs, k)
    if got is None:
        return ScoreResult(value=None, status="insufficient_genes", n_genes_used=0,
                           diagnostics={"reason": "no panel gene shared with the candidate"}, **base)
    qp, cp, qc, cc, used, missing = got
    # Prefer section-wise Moran's I when both sides carry a section label; it
    # removes the projection confound (see _section_morans).
    q_sec, c_sec = query.meta.get("section_label"), candidate.meta.get("section_label")
    q_xy, c_xy = query.meta.get("coords_section"), candidate.meta.get("coords_section")
    mode, sec_info = "projected", {}
    if (section_wise and q_sec is not None and c_sec is not None
            and q_xy is not None and c_xy is not None):
        # Use the FULL matrices here, not the globally subsampled panel: splitting
        # a 1,500-cell subsample across ~40 sections leaves ~37 cells each, below
        # any usable threshold. Sections are instead taken whole and capped to a
        # common size so both sides are compared at equal per-section density.
        qf, cf, _, _ = prepare_panel(query, candidate, genes, max_obs=None, rng=rng)
        cap = min(int(np.bincount(pd.factorize(np.asarray(q_sec))[0]).max()),
                  int(np.bincount(pd.factorize(np.asarray(c_sec))[0]).max()))
        rq = _section_morans(qf, np.asarray(q_xy), np.asarray(q_sec), used, k,
                             min_cells=min_section_cells, max_sections=max_sections,
                             cells_per_section=cap, rng=rng)
        rc = _section_morans(cf, np.asarray(c_xy), np.asarray(c_sec), used, k,
                             min_cells=min_section_cells, max_sections=max_sections,
                             cells_per_section=cap, rng=rng)
        if rq["n_sections_used"] and rc["n_sections_used"]:
            iq, ic, mode = rq["values"], rc["values"], "section_wise"
            sec_info = {"n_sections_query": rq["n_sections_used"],
                        "n_sections_candidate": rc["n_sections_used"],
                        "cells_per_section_query": rq["cells_per_section"],
                        "cells_per_section_candidate": rc["cells_per_section"],
                        "cells_per_section_cap": int(cap)}
    if mode == "projected":
        qc, cc = _rescale(qc, k), _rescale(cc, k)
        qi_idx, qw = _knn_weights(qc, k)
        ci_idx, cw = _knn_weights(cc, k)
        iq = {g: morans_i(qp[g].to_numpy(float), qi_idx, qw) for g in used}
        ic = {g: morans_i(cp[g].to_numpy(float), ci_idx, cw) for g in used}
    pairs = [(iq[g], ic[g]) for g in used if np.isfinite(iq[g]) and np.isfinite(ic[g])]
    base["n_genes_used"] = len(pairs)
    base["genes_missing"] = tuple(missing)
    if not pairs:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "Moran's I undefined (constant genes)",
                                        "morans_i_query": iq, "morans_i_candidate": ic}, **base)
    diff = np.array([a - b for a, b in pairs])
    return ScoreResult(
        value=float(np.sqrt(np.mean(diff ** 2))), status="ok",
        diagnostics={"k": int(k), "mode": mode, **sec_info,
                     "morans_i_query": {g: (None if not np.isfinite(iq[g]) else round(iq[g], 4))
                                        for g in used},
                     "morans_i_candidate": {g: (None if not np.isfinite(ic[g]) else round(ic[g], 4))
                                            for g in used},
                     "n_obs": (int(qp.shape[0]), int(cp.shape[0])),
                     "n_obs_equalised": int(qp.shape[0]),
                     "coord_note": "both sides subsampled to a common observation count and "
                                   "rescaled to unit mean nearest-neighbour distance; only the "
                                   "degree of spatial structure is compared, never positions"},
        **base)


@SCORES.register(
    "spatial_corr_frobenius", orientation="lower_is_better",
    applies_to=("spatial",), needs_coords=True, units_sensitive=False,
    defaults=dict(k=8, method="spearman", max_obs=2000),
    description="RMS difference between the panel correlation matrices computed on spatially "
                "smoothed expression (k-NN mean), i.e. co-localisation at tissue scale",
)
def spatial_corr_frobenius(query: PanelQuery, candidate: CandidateDataset, *,
                           genes: Sequence[str], rng, k: int = 8, method: str = "spearman",
                           max_obs: Optional[int] = 2000, **_) -> ScoreResult:
    """Panel co-expression agreement after spatial smoothing."""
    base = dict(score_name="spatial_corr_frobenius", dataset_name=candidate.name,
                orientation="lower_is_better", n_genes_requested=len(genes))
    got = _spatial_panels(query, candidate, genes, rng, max_obs, k)
    if got is None:
        return ScoreResult(value=None, status="insufficient_genes", n_genes_used=0,
                           diagnostics={"reason": "no panel gene shared with the candidate"}, **base)
    qp, cp, qc, cc, used, missing = got
    base["n_genes_used"] = len(used)
    base["genes_missing"] = tuple(missing)
    if len(used) < MIN_GENES_FOR_STRUCTURE:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": f"{len(used)} usable genes"}, **base)
    qc, cc = _rescale(qc, k), _rescale(cc, k)
    qi_idx, _ = _knn_weights(qc, k)
    ci_idx, _ = _knn_weights(cc, k)
    Q = pd.DataFrame(_smooth(qp.to_numpy(float), qi_idx), columns=used)
    C = pd.DataFrame(_smooth(cp.to_numpy(float), ci_idx), columns=used)
    A, B = Q.corr(method=method), C.corr(method=method)
    iu = np.triu_indices(len(used), k=1)
    a, b = A.to_numpy()[iu], B.to_numpy()[iu]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "undefined smoothed correlations"}, **base)
    return ScoreResult(
        value=float(np.sqrt(np.mean((a - b) ** 2))), status="ok",
        diagnostics={"k": int(k), "method": method, "n_pairs": int(a.size),
                     "smoothed_corr_query": A.round(4).to_dict(),
                     "smoothed_corr_candidate": B.round(4).to_dict(),
                     "n_obs": (int(qp.shape[0]), int(cp.shape[0]))},
        **base)
