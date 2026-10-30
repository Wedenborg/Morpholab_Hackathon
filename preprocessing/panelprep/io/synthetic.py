"""Synthetic end-to-end fixture with a *known* correct answer.

This is the package's acceptance test. It generates one query experiment and a
set of candidate datasets whose true similarity to the query is known by
construction, so every score can be checked against an expected ranking
instead of being eyeballed.

Generative model
----------------
Cells sit on a 2D sheet. ``n_programs`` latent factors vary smoothly over
space (Gaussian bumps); cell-type identity is the argmax program. Gene
expression is ``log1p`` of a Poisson draw whose rate is
``exp(loadings @ program + gene_offset)``, so the output is already on a
counts-like scale and is normalised to the package's ``log1p_cp10k`` contract.

Candidates, in decreasing expected agreement
--------------------------------------------
``sibling``
    Same loadings, same program geometry, new cells. Expected rank 1.
``same_program_diff_mixture``
    Same loadings, but the spatial bumps are re-weighted so the cell-type
    proportions differ. Gene-gene structure survives, marginals shift.
``shuffled_structure``
    Same gene marginals (per-gene values permuted across cells), so the
    panel correlation structure is destroyed while the empirical
    distribution of each gene is *identical*. This separates
    distribution-based scores (which cannot see the difference) from
    structure-based scores (which can) -- the central validity check.
``unrelated``
    Independent loadings and programs. Expected last.

MODIFY HERE
-----------
``make_synthetic_case(...)`` keyword arguments control size, panel, noise and
the candidate list; ``EXPECTED_ORDER`` encodes the assertions used by
``tests/test_synthetic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..normalize import to_log1p_cp10k
from ..schemas import CandidateDataset, PanelQuery

#: Candidates ordered from most to least similar to the query, by construction.
EXPECTED_ORDER: List[str] = [
    "sibling", "same_program_diff_mixture", "shuffled_structure", "unrelated",
]

#: Scores that are *blind* to "shuffled_structure" because it preserves each
#: gene's marginal distribution exactly. Used by the acceptance tests.
MARGINAL_BLIND_SCORES = ("avg_kl", "avg_js")

DESCRIPTOR = (
    "Spatial transcriptomics of a developing mouse embryo at embryonic day 7.5. "
    "Serial sections were profiled to study germ layer patterning and the emergence of "
    "mesoderm and endoderm progenitors during early organogenesis. Cells retain their "
    "tissue coordinates."
)


@dataclass
class SyntheticCase:
    query: PanelQuery
    candidates: Dict[str, CandidateDataset]
    truth: Dict[str, object] = field(default_factory=dict)

    @property
    def candidate_list(self) -> list[CandidateDataset]:
        return list(self.candidates.values())


def _programs(coords: np.ndarray, centers: np.ndarray, width: float, weights: np.ndarray) -> np.ndarray:
    """``(N, K)`` smooth non-negative spatial programs, rows summing to 1."""
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    act = np.exp(-d2 / (2 * width ** 2)) * weights[None, :]
    return act / np.clip(act.sum(1, keepdims=True), 1e-12, None)


def _simulate(
    rng: np.random.Generator,
    n_obs: int,
    loadings: np.ndarray,       # (G, K)
    offsets: np.ndarray,        # (G,)
    centers: np.ndarray,
    width: float,
    weights: np.ndarray,
    extent: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    coords = rng.uniform(0, extent, size=(n_obs, 2))
    prog = _programs(coords, centers, width, weights)
    rate = np.exp(prog @ loadings.T + offsets[None, :])
    counts = rng.poisson(rate)
    celltype = prog.argmax(1)
    return pd.DataFrame(counts), coords, celltype


def make_synthetic_case(
    *,
    n_genes: int = 200,
    panel_size: int = 4,
    n_programs: int = 5,
    n_query: int = 800,
    n_candidate: int = 800,
    extent: float = 10.0,
    width: float = 2.2,
    seed: int = 0,
    gene_names: Optional[List[str]] = None,
) -> SyntheticCase:
    """Build the fixture. See module docstring for the candidate semantics."""
    rng = np.random.default_rng(seed)
    genes = gene_names or [f"Gene{i:04d}" for i in range(n_genes)]
    n_genes = len(genes)

    loadings = rng.normal(0.0, 1.6, size=(n_genes, n_programs))
    offsets = rng.normal(1.1, 0.5, size=n_genes)
    centers = rng.uniform(0, extent, size=(n_programs, 2))
    weights = np.ones(n_programs)

    q_counts, q_coords, q_type = _simulate(
        rng, n_query, loadings, offsets, centers, width, weights, extent)
    q_counts.columns = genes
    q_expr = to_log1p_cp10k(q_counts)

    # The panel: the measured gene subset. Pick the genes with the strongest
    # program loading so the panel is informative (as a designed panel would be).
    panel = [genes[i] for i in np.argsort(-np.abs(loadings).max(1))[:panel_size]]

    query = PanelQuery(
        expression=q_expr.loc[:, panel],
        coords=q_coords,
        descriptor=DESCRIPTOR,
        organism="mus_musculus",
        obs=pd.DataFrame({"celltype": [f"ct{c}" for c in q_type]}),
        name="query_synthetic",
        meta={"generator": "panelprep.io.synthetic", "seed": seed},
    )

    cands: Dict[str, CandidateDataset] = {}

    def add(name, counts, coords, celltype, category, description,
            permute_after_norm=False, **meta):
        """Normalise, then optionally permute each gene independently.

        The permutation must come AFTER normalisation: permuting raw counts
        changes every cell's library size, so CP10K renormalisation would then
        alter each gene's marginal distribution and the "marginals preserved,
        structure destroyed" contrast the fixture is built on would be only
        approximate. Permuting the normalised values preserves each gene's
        empirical distribution exactly.
        """
        expr = to_log1p_cp10k(counts)
        if permute_after_norm:
            expr = expr.copy()
            for g in expr.columns:
                expr[g] = rng.permutation(expr[g].to_numpy())
        cands[name] = CandidateDataset(
            name=name, category=category, expression=expr,
            coords=coords if category == "spatial" else None,
            obs=pd.DataFrame({"celltype": [f"ct{c}" for c in celltype]}),
            meta={"description": description, "organism": "mus_musculus",
                  "synthetic": True, **meta},
        )

    c, xy, ct = _simulate(rng, n_candidate, loadings, offsets, centers, width, weights, extent)
    c.columns = genes
    add("sibling", c, xy, ct, "spatial",
        "mouse embryo E7.5 spatial transcriptomics germ layer mesoderm endoderm patterning",
        stage="E7.5", modality="synthetic spatial")

    w2 = np.array([3.0 if i == 0 else 0.4 for i in range(n_programs)])
    c, xy, ct = _simulate(rng, n_candidate, loadings, offsets, centers, width, w2, extent)
    c.columns = genes
    add("same_program_diff_mixture", c, xy, ct, "single_cell",
        "mouse embryo E7.5 single cell gastrulation atlas dissociated cells mesoderm",
        stage="E7.5", modality="synthetic scRNA-seq")

    c, xy, ct = _simulate(rng, n_candidate, loadings, offsets, centers, width, weights, extent)
    c.columns = genes
    add("shuffled_structure", c, xy, ct, "single_cell",
        "mouse embryo single cell reference with gene co-expression destroyed",
        permute_after_norm=True, stage="E7.5", modality="synthetic scRNA-seq (permuted)")

    l2 = rng.normal(0.0, 1.6, size=(n_genes, n_programs))
    o2 = rng.normal(1.1, 0.5, size=n_genes)
    ctr2 = rng.uniform(0, extent, size=(n_programs, 2))
    c, xy, ct = _simulate(rng, n_candidate, l2, o2, ctr2, width, weights, extent)
    c.columns = genes
    add("unrelated", c, xy, ct, "spatial",
        "adult mouse liver spatial transcriptomics hepatocyte zonation",
        stage="adult", modality="synthetic spatial")

    return SyntheticCase(
        query=query, candidates=cands,
        truth={"panel": panel, "genes": genes, "loadings": loadings,
               "expected_order": list(EXPECTED_ORDER),
               "marginal_blind_scores": list(MARGINAL_BLIND_SCORES)},
    )
