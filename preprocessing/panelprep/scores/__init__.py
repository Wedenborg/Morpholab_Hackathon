"""Score registry and runner.

A score answers: *how much do the panel genes agree between the measured query
and this candidate dataset?*

Adding a score
--------------
Write a function in any module in this package and decorate it::

    @SCORES.register(
        "my_score",
        orientation="lower_is_better",   # or "higher_is_better"
        applies_to=("single_cell", "spatial"),
        needs_coords=False,
        units_sensitive=True,            # refuse to run across mismatched units
        description="one line for the report",
    )
    def my_score(query, candidate, *, genes, rng, **params) -> ScoreResult: ...

then add ``"my_score"`` to ``Config.scores``. The comparison table, the colour
map and the markdown report are all generated from the registry, so nothing
else needs editing. ``panelprep/scores/custom.py`` exists as the drop-in spot.

Contract
--------
* ``query``/``candidate`` are ``PanelQuery``/``CandidateDataset``; rows are
  observations, columns are gene symbols.
* ``applies_to`` names the *categories* a score is meaningful for. The
  expression scores accept ``single_cell``, ``spatial`` and ``causal``: a
  perturbation dataset has a real cell x gene matrix, and the brief says to
  treat such references as gene expression for now. Its control and perturbed
  arms are distinguished in ``obs["condition"]``, so a future score can use the
  contrast; the present ones pool the arms and that is recorded here.
  Entries with no matrix at all (a protein language model, an interaction graph)
  are not scored -- see ``split_scoreable``.
* ``genes`` are the requested panel symbols. Use ``prepare_panel`` to resolve
  them in both objects and to get the missing list -- do not index directly.
* Return a ``ScoreResult``. Never raise for data reasons (missing genes,
  constant genes, wrong category): set ``status`` and ``value=None``.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..normalize import resolve_genes, units_compatible
from ..registry import Registry
from ..schemas import CandidateDataset, PanelQuery, ScoreResult

SCORES = Registry("score")

MIN_GENES_FOR_STRUCTURE = 2   # correlation-matrix scores need >=2 shared genes


def prepare_panel(
    query: PanelQuery,
    candidate: CandidateDataset,
    genes: Sequence[str],
    *,
    max_obs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], list[str], list[str]]:
    """Align the panel across query and candidate.

    Returns ``(query_panel, candidate_panel, used_genes, missing_genes)``.
    Both frames have the *same* columns in the same order -- that pairing is
    by gene symbol, and it is the only pairing assumption in the package
    (observations are **not** paired across datasets; no score may assume a
    row correspondence).
    """
    rng = np.random.default_rng() if rng is None else rng
    if not candidate.has_expression:
        return None, None, [], list(genes)
    q_names, q_missing = resolve_genes(genes, query.expression.columns)
    c_names, c_missing = resolve_genes(genes, candidate.expression.columns)
    missing = sorted(set(q_missing) | set(c_missing))
    used = [g for g in genes if g not in missing]
    if not used:
        return None, None, [], missing
    q_sel = [n for g, n in zip(genes, q_names) if g not in missing and n is not None]
    c_sel = [n for g, n in zip(genes, c_names) if g not in missing and n is not None]
    qp = query.expression.loc[:, q_sel].copy()
    cp = candidate.expression.loc[:, c_sel].copy()
    qp.columns = used
    cp.columns = used
    if max_obs is not None:
        qp = _subsample(qp, max_obs, rng)
        cp = _subsample(cp, max_obs, rng)
    return qp, cp, used, missing


def _subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if df.shape[0] <= n:
        return df
    idx = rng.choice(df.shape[0], size=n, replace=False)
    return df.iloc[np.sort(idx)]


def guard(
    query: PanelQuery,
    candidate: CandidateDataset,
    genes: Sequence[str],
    meta: dict,
    score_name: str,
    *,
    allow_unit_mismatch: bool = False,
) -> Optional[ScoreResult]:
    """Standard pre-checks. Returns a terminal ``ScoreResult`` or ``None``."""
    orient = meta.get("orientation", "lower_is_better")
    base = dict(score_name=score_name, dataset_name=candidate.name, value=None,
                orientation=orient, n_genes_requested=len(genes))
    applies = meta.get("applies_to", ("single_cell", "spatial"))
    if candidate.category not in applies:
        return ScoreResult(**base, status="not_applicable",
                           diagnostics={"reason": f"category {candidate.category!r} not in {applies}"})
    if not candidate.has_expression:
        return ScoreResult(**base, status="not_applicable",
                           diagnostics={"reason": "candidate carries no expression matrix"})
    if meta.get("needs_coords") and not candidate.has_coords:
        return ScoreResult(**base, status="not_applicable",
                           diagnostics={"reason": "candidate has no spatial coordinates"})
    if meta.get("units_sensitive", False) and not allow_unit_mismatch:
        if not units_compatible(query.units, candidate.units):
            return ScoreResult(**base, status="units_mismatch",
                               diagnostics={"query_units": query.units,
                                            "candidate_units": candidate.units,
                                            "reason": "distributional comparison needs matched "
                                                      "normalisation; set allow_unit_mismatch=True "
                                                      "to override"})
    return None


def split_scoreable(candidates: Iterable[CandidateDataset]
                    ) -> tuple[list[CandidateDataset], list[CandidateDataset]]:
    """Separate candidates that can be scored from pure embedding sources.

    An agreement score compares the panel genes' behaviour between two
    expression matrices. A catalogue entry such as an ESM-2 protein language
    model, a STRING interaction graph or a curated KEGG network has **no cell x
    gene matrix at all**, so it is not a scoring candidate: it is a source for
    the gene-embedding deliverable. Emitting a row of "not applicable" for it
    would imply it had been considered and rejected as a reference, which is
    the wrong claim -- it was never a reference.

    Returns ``(scoreable, embedding_sources)``. The pipeline reports the second
    list in the embedding inventory instead of in the agreement table.
    """
    scoreable, sources = [], []
    for c in candidates:
        (scoreable if c.has_expression else sources).append(c)
    return scoreable, sources


def run_scores(
    query: PanelQuery,
    candidates: Iterable[CandidateDataset],
    *,
    score_names: Sequence[str],
    genes: Optional[Sequence[str]] = None,
    params: Optional[dict] = None,
    seed: int = 0,
    max_obs: Optional[int] = None,
    allow_unit_mismatch: bool = False,
) -> pd.DataFrame:
    """Run every requested score against every scoreable candidate (long format).

    Candidates with no expression matrix are **excluded**, not reported as
    "not applicable" -- see :func:`split_scoreable`. Errors inside a score are
    captured as ``status="error"`` rows so one bad metric cannot take down the
    table.
    """
    params = params or {}
    rows = []
    candidates, _sources = split_scoreable(candidates)
    for cand in candidates:
        for name in score_names:
            entry = SCORES[name]
            rng = np.random.default_rng([seed, abs(hash(name)) % (2**31), abs(hash(cand.name)) % (2**31)])
            terminal = guard(query, cand, genes or query.panel, entry.meta, name,
                             allow_unit_mismatch=allow_unit_mismatch)
            if terminal is not None:
                rows.append(terminal.as_row())
                continue
            kw = dict(entry.meta.get("defaults", {}))
            kw.update(params.get(name, {}))
            if max_obs is not None:
                kw.setdefault("max_obs", max_obs)
            try:
                res = entry.fn(query, cand, genes=list(genes or query.panel), rng=rng, **kw)
            except Exception as exc:  # pragma: no cover - defensive
                res = ScoreResult(score_name=name, dataset_name=cand.name, value=None,
                                  orientation=entry.meta.get("orientation", "lower_is_better"),
                                  status="error", n_genes_requested=len(genes or query.panel),
                                  diagnostics={"error": f"{type(exc).__name__}: {exc}"})
            rows.append(res.as_row())
    return pd.DataFrame(rows)


# Importing the built-in score modules registers them.
from . import distributional, structure, alignment, spatial, custom  # noqa: E402,F401
