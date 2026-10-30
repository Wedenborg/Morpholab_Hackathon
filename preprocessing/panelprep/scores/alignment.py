"""Optimal-transport cluster agreement (the brief's score (c)).

Requested formulation
---------------------
    "cluster the Cells, run an optimal transport between cells in target and
    matched and then cluster and compute ARI type scores for whether the matched
    examples are in the same cluster (kind of how people compare if embeddings
    are aligned)"

The correction that matters
---------------------------
The Adjusted Rand Index is defined for **two labelings of the same set of
objects**. Query cells and candidate cells are different objects, so
"ARI between the query clustering and the candidate clustering" is not a
defined quantity as written. The intended scientific question -- *do cells that
optimal transport matches to each other end up in corresponding clusters?* --
is well posed once the transport is used to move one labeling onto the other's
objects. That is what is implemented:

1. Cluster the **query** cells on the shared panel genes -> labels ``A``
   (``N_q`` objects).
2. Cluster the **candidate** cells independently on the same genes -> ``B``.
3. Solve entropic-regularised OT between the two cell clouds in the shared panel
   space, with uniform marginals and a squared-Euclidean cost. The panel is
   z-scored on the **pooled** sample so a single high-variance gene does not
   define the geometry.
4. Push ``B`` onto the query cells through the coupling. The cluster-by-cluster
   contingency table ``T[a, b] = `` transported mass between query cluster ``a``
   and candidate cluster ``b`` is a *soft* contingency table over ``N_q``
   objects, and the ordinary ARI algebra applies to it once the mass is
   rescaled to sum to ``N_q``. The returned value is this ``ari_soft``.
5. ``ari_hard`` (in diagnostics) is the discrete version: each query cell takes
   the candidate label receiving the most transported mass, then
   ``sklearn.adjusted_rand_score(A, induced)`` -- two labelings of the query
   cells, which is exactly what ARI requires.

Both are agreements: higher is better, 0 is chance.

Things that will fool this score, documented rather than hidden
--------------------------------------------------------------
* **Entropic smoothing.** With uniform marginals, a large ``reg`` drives the
  coupling toward the product measure, which sends ARI to 0 regardless of how
  well the datasets agree. The cost matrix is normalised to unit mean before
  regularisation so ``reg`` means the same thing across datasets, and the
  coupling entropy relative to the product measure is reported in diagnostics.
  Read it before concluding that a dataset disagrees.
* **Cluster count.** ARI depends on ``n_clusters``; it is a parameter, the same
  value is used on both sides, and it is recorded.
* **Marginals.** Uniform marginals assert that both datasets have the same cell
  composition. They usually do not. This is the score's central assumption and
  it is the reason a dataset with the same biology but a different cell-type
  mixture scores lower here than on the correlation scores.
* **Panel dimensionality.** With a 4-gene panel the transport geometry is
  4-dimensional and very coarse; this score is more informative for larger
  panels.

MODIFY HERE: ``n_clusters``, ``cluster_method`` (``kmeans``/``agglomerative``),
``reg``, ``max_obs``, ``standardise``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import CandidateDataset, PanelQuery, ScoreResult
from . import SCORES, prepare_panel


def _cluster(X: np.ndarray, n_clusters: int, method: str, seed: int) -> np.ndarray:
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    raise ValueError("cluster_method must be 'kmeans' or 'agglomerative'")


def _comb2(x: np.ndarray | float) -> np.ndarray | float:
    """``C(x, 2)`` extended to real ``x`` -- the soft-contingency generalisation."""
    return x * (x - 1.0) / 2.0


def soft_ari(table: np.ndarray) -> float:
    """Adjusted Rand Index from a (possibly fractional) contingency table.

    ``table[a, b]`` is the number of objects (or the mass of objects) assigned to
    cluster ``a`` by one labeling and ``b`` by the other. For integer tables this
    reduces exactly to the standard ARI.
    """
    n = float(table.sum())
    if n <= 1:
        return float("nan")
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    index = float(_comb2(table).sum())
    exp_a, exp_b = float(_comb2(a).sum()), float(_comb2(b).sum())
    expected = exp_a * exp_b / _comb2(n)
    maximum = 0.5 * (exp_a + exp_b)
    denom = maximum - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((index - expected) / denom)


@SCORES.register(
    "ot_cluster_ari", orientation="higher_is_better",
    applies_to=("single_cell", "spatial", "causal"), needs_coords=False, units_sensitive=True,
    defaults=dict(n_clusters=5, cluster_method="kmeans", reg=0.05, max_obs=800,
                  standardise=True),
    description="ARI between the query cell clustering and the clustering transported from the "
                "candidate by entropic optimal transport on the shared panel genes; soft "
                "contingency table over the query cells",
)
def ot_cluster_ari(
    query: PanelQuery, candidate: CandidateDataset, *, genes: Sequence[str], rng,
    n_clusters: int = 5, cluster_method: str = "kmeans", reg: float = 0.05,
    max_obs: Optional[int] = 800, standardise: bool = True, numItermax: int = 2000,
    **_,
) -> ScoreResult:
    """Transport-induced clustering agreement. See the module docstring."""
    import ot as pot

    qp, cp, used, missing = prepare_panel(query, candidate, genes, max_obs=max_obs, rng=rng)
    base = dict(score_name="ot_cluster_ari", dataset_name=candidate.name,
                orientation="higher_is_better", n_genes_requested=len(genes),
                n_genes_used=len(used), genes_missing=tuple(missing))
    if not used:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "no panel gene shared with the candidate"}, **base)

    Xq = qp.to_numpy(dtype=float)
    Xc = cp.to_numpy(dtype=float)
    if standardise:
        pooled = np.vstack([Xq, Xc])
        mu, sd = pooled.mean(0), pooled.std(0)
        sd[sd == 0] = 1.0
        Xq, Xc = (Xq - mu) / sd, (Xc - mu) / sd

    seed = int(rng.integers(0, 2**31 - 1))
    A = _cluster(Xq, n_clusters, cluster_method, seed)
    B = _cluster(Xc, n_clusters, cluster_method, seed)
    ka, kb = len(np.unique(A)), len(np.unique(B))
    if ka < 2 or kb < 2:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": f"degenerate clustering ({ka}/{kb} clusters); "
                                                  "the panel carries too little variation",
                                        "n_obs": (int(Xq.shape[0]), int(Xc.shape[0]))}, **base)

    M = pot.dist(Xq, Xc, metric="sqeuclidean")
    mean_cost = float(M.mean())
    if mean_cost > 0:
        M = M / mean_cost                       # so `reg` means the same thing everywhere
    nq, nc = Xq.shape[0], Xc.shape[0]
    wq = np.full(nq, 1.0 / nq)
    wc = np.full(nc, 1.0 / nc)
    solver = "sinkhorn"
    G = pot.sinkhorn(wq, wc, M, reg=reg, numItermax=numItermax)
    if not np.all(np.isfinite(G)) or G.sum() <= 0:
        G = pot.emd(wq, wc, M)                  # exact fallback when Sinkhorn underflows
        solver = "emd"

    # soft contingency table over the query cells
    Ah = np.zeros((nq, ka)); Ah[np.arange(nq), pd.factorize(A)[0]] = 1.0
    Bh = np.zeros((nc, kb)); Bh[np.arange(nc), pd.factorize(B)[0]] = 1.0
    T = Ah.T @ G @ Bh
    T = T / T.sum() * nq                        # rescale transported mass to N_q objects
    ari = soft_ari(T)

    # hard variant: argmax-transported candidate label per query cell
    from sklearn.metrics import adjusted_rand_score

    induced = (G @ Bh).argmax(axis=1)
    ari_hard = float(adjusted_rand_score(A, induced))

    prod = np.outer(wq, wc)
    mask = G > 0
    kl_to_product = float(np.sum(G[mask] * (np.log(G[mask]) - np.log(prod[mask]))))

    return ScoreResult(
        value=float(ari), status="ok",
        diagnostics={"ari_soft": float(ari), "ari_hard": ari_hard,
                     "n_clusters": n_clusters, "clusters_found": (int(ka), int(kb)),
                     "cluster_method": cluster_method, "reg": reg, "solver": solver,
                     "ot_cost": float(np.sum(G * M) * mean_cost),
                     "coupling_kl_to_product": kl_to_product,
                     "n_obs": (int(nq), int(nc)),
                     "note": "coupling_kl_to_product near 0 means the coupling is close to "
                             "independent, which forces ARI toward 0 regardless of agreement"},
        **base)
