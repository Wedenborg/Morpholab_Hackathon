"""Panel selection with the Gaussian factor model of `active_design/theory.md`.

Roles of the data, as in the E7.5 mouse-embryo preset:
    target    the spatial dataset: the sample the panel is planned for. Only its cell
              positions are used (for the spatial prior); its expression is never read here.
    visible   the reference dataset: a fully measured spatial sample (the visible embryo). The
              factor model's gene means, weights and noise are fitted on its neighbourhoods.
    mdCLA     single cells (mdCLA E7.5 "Early embryo"), read from a local file.

A panel is measured on every cell of the target, so one round adds the same genes to every
neighbourhood; there are no sections to choose.

    gaussian_visible   greedy expected information gain about the factors of all target
                       neighbourhoods, given the genes already measured; factors and weights
                       from the visible embryo (theory.md section 1). The posterior covariance
                       does not depend on measured values, so only gene names are needed.
    gaussian_mdcla     the same, with factor directions and weights from mdCLA, rescaled per
                       gene to the visible embryo (theory.md section 1.1).
    rotating_visible   marker genes, round-robin over the visible embryo's cell types (or
                       clusters), skipping genes already measured; each new round continues
                       with the next markers.
    rotating_mdcla     the same with mdCLA's cell subtypes.

Each round is scored by the visible-embryo model's expected share of unmeasured genes'
neighbourhood-level variance it predicts, and, when a scoring function is passed (the E7.5
preset), by the eval_team benchmark. Only numpy and scipy are used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial import cKDTree

from .models import SelectedGene


K_MAX = 16                 # factors
CELLS_PER_UNIT = 30        # cells per neighbourhood
SMOOTHNESS = 3.0           # lambda in Q = c (I + lambda L)
K_WITHIN = 6               # neighbourhood links within a section
MIN_DETECTION = 0.05       # candidate genes for the information design
N_CANDIDATES = 2000
EVAL_DETECTION = 0.02      # genes scored by the expected share
SOURCE_DETECTION = 0.02    # genes used to place neighbourhoods on mdCLA factors

METHODS = {
    "gaussian_visible": ("information", "visible", "Gaussian information gain · visible embryo"),
    "gaussian_mdcla": ("information", "mdcla", "Gaussian information gain · mdCLA"),
    "rotating_visible": ("markers", "visible", "Rotating markers · visible embryo"),
    "rotating_mdcla": ("markers", "mdcla", "Rotating markers · mdCLA"),
}
SCORE_LABELS = {"information": "Information gain (nats)", "markers": "Marker strength"}


@dataclass
class Expression:
    X: sp.csr_matrix                 # cells x target genes, log scale
    found: np.ndarray                # target genes present in this dataset
    coordinates: np.ndarray | None   # cells x 2 or 3
    cell_types: np.ndarray | None
    transform: str


@dataclass
class FactorModel:
    W: np.ndarray                    # genes x K
    r: np.ndarray                    # neighbourhood-level noise per gene
    b: np.ndarray                    # gene means
    detect: np.ndarray               # fraction of visible cells with a nonzero value
    q: np.ndarray                    # eigenvalues of the target's prior precision Q
    n_visible_units: int

    @property
    def K(self) -> int:
        return self.W.shape[1]


def _log_scale(M: Any) -> tuple[sp.csr_matrix, str]:
    """Integer counts are normalised to the median library size and log1p-transformed;
    non-integer values are used as given (assumed log scale), or log1p-transformed if their
    maximum exceeds 50."""
    X = sp.csr_matrix(M, dtype=np.float32)
    X.eliminate_zeros()
    if X.nnz and np.all(X.data == np.round(X.data)):
        total = np.asarray(X.sum(1)).ravel()
        X = sp.csr_matrix(sp.diags(np.median(total[total > 0]) / np.maximum(total, 1)) @ X, dtype=np.float32)
        X.data = np.log1p(X.data)
        return X, "integer counts: normalised to median library size, then log1p"
    if X.nnz and X.data.max() > 50:
        X.data = np.log1p(X.data)
        return X, "values above 50 found: log1p"
    return X, "used as given (log scale assumed)"


def load_expression(adata: Any, layer: str, gene_ids: list[str], target_genes: list[str],
                    cell_type_field: str | None, coordinates: np.ndarray | None = None) -> Expression:
    """Expression on a log scale with columns in the target's gene order (exact identifier
    match; genes the dataset lacks are zero columns)."""
    X, transform = _log_scale(adata.X if layer == "X" else adata.layers[layer])
    pos = {g: i for i, g in enumerate(gene_ids)}
    cols = np.array([pos.get(g, -1) for g in target_genes])
    found = cols >= 0
    S = sp.csr_matrix((np.ones(found.sum()), (cols[found], np.where(found)[0])),
                      shape=(len(gene_ids), len(target_genes)))
    types = None
    if cell_type_field and cell_type_field in adata.obs:
        types = adata.obs[cell_type_field].astype(str).to_numpy()
    return Expression(sp.csr_matrix(X @ S, dtype=np.float32), found,
                      None if coordinates is None else np.asarray(coordinates, float), types, transform)


def load_mdcla(path: Path, target_genes: list[str]) -> Expression:
    """mdCLA E7.5 cells ("Early embryo" when the group column exists), subtypes as cell types."""
    import anndata as ad

    adata = ad.read_h5ad(path)
    if "level1_group" in adata.obs:
        adata = adata[(adata.obs["level1_group"] == "Early embryo").to_numpy()]
    field = "level4_subtypes" if "level4_subtypes" in adata.obs else None
    return load_expression(adata, "X", list(map(str, adata.var_names)), target_genes, field)


# ----------------------------------------------------------------------------- model
def _kmeans(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 1 or len(points) <= k:
        return np.zeros(len(points), int) if k <= 1 else np.arange(len(points))
    _, labels = kmeans2(points, k, minit="++", seed=seed)
    return np.unique(labels, return_inverse=True)[1]


def neighbourhoods(coordinates: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Neighbourhood id per cell, and the section of each neighbourhood (None when the data
    are not in sections). 3-D data whose third coordinate takes few values are treated as
    sections: neighbourhoods are k-means clusters on (x, y) within each section."""
    n = len(coordinates)
    per = max(1, min(CELLS_PER_UNIT, n // 20))
    if coordinates.shape[1] == 3:
        z_values, z_index = np.unique(coordinates[:, 2], return_inverse=True)
        if len(z_values) > 1 and n / len(z_values) >= 2 * per:
            units = np.empty(n, int)
            section = []
            for s in range(len(z_values)):
                idx = np.where(z_index == s)[0]
                lab = _kmeans(coordinates[idx, :2], max(1, len(idx) // per), seed)
                units[idx] = lab + len(section)
                section += [s] * (lab.max() + 1)
            return units, np.array(section)
    return _kmeans(coordinates, max(1, n // per), seed), None


def prior_eigen(centroids: np.ndarray, section: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of Q = c (I + lambda L): neighbourhoods linked to their
    nearest neighbours (within a section, and to the nearest one in each adjacent section), c
    chosen so the average prior variance is 1."""
    n = len(centroids)
    A = np.zeros((n, n))
    groups = [np.arange(n)] if section is None else [np.where(section == s)[0] for s in np.unique(section)]
    dims = slice(None) if section is None else slice(0, 2)
    for idx in groups:
        k = min(K_WITHIN, len(idx) - 1)
        if k < 1:
            continue
        _, nb = cKDTree(centroids[idx, dims]).query(centroids[idx, dims], k=k + 1)
        for row, cols in zip(idx, nb[:, 1:]):
            A[row, idx[cols]] = 1
    adjacent = zip(groups[:-1], groups[1:]) if section is not None else []
    for a, b in adjacent:
        for src, dst in ((a, b), (b, a)):
            _, nb = cKDTree(centroids[dst, :2]).query(centroids[src, :2], k=1)
            A[src, dst[nb]] = 1
    A = np.maximum(A, A.T)
    L = np.diag(A.sum(1)) - A
    q, U = np.linalg.eigh(np.eye(n) + SMOOTHNESS * L)
    c = np.mean(1 / q)
    return c * q, U


def _reference_directions(ref: Expression, K: int) -> np.ndarray:
    """Top K principal directions (genes x K, unit norm) of the centred reference cells."""
    X = ref.X
    m = np.asarray(X.mean(0)).ravel()
    ones = np.ones(X.shape[0])
    op = LinearOperator(
        X.shape, dtype=np.float64,
        matvec=lambda v: X @ np.ravel(v) - ones * (m @ np.ravel(v)),
        rmatvec=lambda u: X.T @ np.ravel(u) - m * np.ravel(u).sum(),
    )
    _, svals, vt = svds(op, k=K, random_state=0)
    return vt[np.argsort(-svals)].T


def fit(visible: Expression, target_coordinates: np.ndarray, seed: int = 0,
        source: Expression | None = None) -> FactorModel:
    """Factor model on the visible embryo's neighbourhood averages (theory.md section 1): PCA
    scores scaled to variance 1, weights by least squares, residual variance per gene. With a
    source (mdCLA), the factor directions and weights come from its cells and each gene's
    weights are only rescaled to the visible embryo (theory.md section 1.1). The prior is built
    on the target's neighbourhoods (theory.md section 2)."""
    units, _ = neighbourhoods(visible.coordinates, seed)
    n_units = units.max() + 1
    M = sp.csr_matrix((np.ones(len(units)), (units, np.arange(len(units)))), shape=(n_units, len(units)))
    sizes = np.asarray(M.sum(1)).ravel()
    Y = np.asarray((sp.diags(1 / sizes) @ M @ visible.X).todense(), dtype=np.float64)
    b = Y.mean(0)
    R = Y - b
    detect = np.asarray((visible.X > 0).mean(0)).ravel()
    if source is None:
        K = max(1, min(K_MAX, n_units - 2, R.shape[1] - 1))
        evals, evecs = np.linalg.eigh(R @ R.T)
        Z = evecs[:, np.argsort(-evals)[:K]] * np.sqrt(n_units)    # Z^T Z = n I: variance 1 per factor
        W = R.T @ Z / n_units
    else:
        K = max(1, min(K_MAX, n_units - 2, source.X.shape[0] - 1, R.shape[1] - 2))
        a = _reference_directions(source, K)                         # genes x K
        ok = (detect >= SOURCE_DETECTION) & source.found
        if ok.sum() < K:
            ok = source.found & (detect > 0)
        Z = np.linalg.lstsq(a[ok], R[:, ok].T, rcond=None)[0].T      # neighbourhoods on the directions
        sd = np.maximum(Z.std(0), 1e-12)
        Z = Z / sd
        base = a * sd                                                # source pattern, Z scale
        pred = Z @ base.T
        den = (pred ** 2).sum(0)
        beta = (R * pred).sum(0) / np.where(den > 0, den, np.inf)
        good = ok & (den > 0)
        beta0 = np.median(beta[good]) if good.any() else 0.0
        w = detect / (detect + 0.05)                                 # shrink rarely detected genes
        W = base * (w * beta + (1 - w) * beta0)[:, None]
    r = (R - Z @ W.T).var(0)
    floor = 0.01 * np.median(r[r > 0]) if np.any(r > 0) else 1e-6
    r = np.maximum(r, floor)
    t_units, t_section = neighbourhoods(target_coordinates, seed)
    centroids = np.vstack([target_coordinates[t_units == u].mean(0) for u in range(t_units.max() + 1)])
    q, _ = prior_eigen(centroids, t_section)
    return FactorModel(W, r, b, detect, q, int(n_units))


def _precision_inverses(model: FactorModel, measured: list[int]) -> np.ndarray:
    """A_j = (q_j I + F)^-1 for every eigenvalue q_j of Q, F = sum_g w_g w_g^T / r_g over the
    measured genes. When every neighbourhood carries the same genes, the posterior precision
    Q (x) I + I (x) F is block diagonal in Q's eigenbasis with these blocks."""
    W, r = model.W[measured], model.r[measured]
    F = (W / r[:, None]).T @ W if measured else np.zeros((model.K, model.K))
    return np.linalg.inv(model.q[:, None, None] * np.eye(model.K) + F)


def _add_gene(A: np.ndarray, w: np.ndarray, r: float) -> None:
    Aw = A @ w                                               # n x K
    A -= Aw[:, :, None] * Aw[:, None, :] / (r + Aw @ w)[:, None, None]


# ----------------------------------------------------------------------------- methods
def information_panel(model: FactorModel, measured: list[int], eligible: list[int], ng: int
                      ) -> list[tuple[int, float, str]]:
    """Greedy information gain. Measuring gene g on every neighbourhood gains
        0.5 sum_j log(1 + w_g^T A_j w_g / r_g)
    nats about the factors (theory.md section 3, written in Q's eigenbasis)."""
    info = (model.W ** 2).sum(1) / model.r
    eligible = np.asarray(eligible, int)
    order = eligible[np.lexsort((-info[eligible], model.detect[eligible] < MIN_DETECTION))]
    cand = order[:max(N_CANDIDATES, ng)]
    A = _precision_inverses(model, measured)
    Wc, rc = model.W[cand], model.r[cand]
    taken = np.zeros(len(cand), bool)
    out = []
    for _ in range(ng):
        AW = np.einsum("jkl,cl->jck", A, Wc)                 # n x C x K
        quad = np.einsum("jck,ck->cj", AW, Wc)
        gain = 0.5 * np.log1p(quad / rc[:, None]).sum(1)
        gain[taken] = -np.inf
        best = int(np.argmax(gain))
        taken[best] = True
        g = int(cand[best])
        _add_gene(A, model.W[g], model.r[g])
        out.append((g, float(gain[best]), "Expected information gain about the factors (nats), given the "
                    "previous genes and the genes ranked above."))
    return out


def marker_panel(X: sp.csr_matrix, labels: np.ndarray, names: list[str], measured: list[int],
                 eligible: list[int], ng: int, source: str) -> list[tuple[int, float, str]]:
    """Each type's next-best marker in turn (largest mean difference to all other cells), as
    the rotating marker sets of active_design/designs.py."""
    n_cells, n_genes = X.shape
    allowed = np.zeros(n_genes, bool)
    allowed[eligible] = True
    allowed[measured] = False
    total = np.asarray(X.sum(0)).ravel()
    ranked, diffs = [], []
    for t in range(len(names)):
        mask = labels == t
        inside = np.asarray(X[mask].sum(0)).ravel()
        diff = inside / mask.sum() - (total - inside) / max(n_cells - mask.sum(), 1)
        diff = np.where(allowed, diff, -np.inf)
        ranked.append(np.argsort(-diff))
        diffs.append(diff)
    out, used = [], set()
    for depth in range(n_genes):
        for t in range(len(names)):
            g = int(ranked[t][depth])
            if len(out) < ng and allowed[g] and g not in used:
                used.add(g)
                out.append((g, float(diffs[t][g]), f"Marker for {names[t]} in the {source}: mean log "
                            f"expression {diffs[t][g]:.3f} above the other cells."))
        if len(out) >= ng:
            break
    return out


def cell_groups(X: sp.csr_matrix, cell_types: np.ndarray | None, seed: int) -> tuple[np.ndarray, list[str]]:
    """Cell types with enough cells, or, when there are none, k-means clusters of the cells'
    top principal components."""
    n = X.shape[0]
    min_size = min(20, max(2, n // 20))
    if cell_types is not None:
        names, labels = np.unique(cell_types, return_inverse=True)
        keep = [t for t in range(len(names)) if (labels == t).sum() >= min_size]
        if len(keep) >= 2:
            remap = {t: i for i, t in enumerate(keep)}
            labels = np.array([remap.get(t, -1) for t in labels])
            return labels, [str(names[t]) for t in keep]
    k = max(1, min(20, n - 1, X.shape[1] - 1))
    m = np.asarray(X.mean(0)).ravel()
    ones = np.ones(n)
    op = LinearOperator(X.shape, dtype=np.float64,
                        matvec=lambda v: X @ np.ravel(v) - ones * (m @ np.ravel(v)),
                        rmatvec=lambda u: X.T @ np.ravel(u) - m * np.ravel(u).sum())
    u, svals, _ = svds(op, k=k, random_state=seed % (2 ** 32))
    labels = _kmeans(u * svals, max(2, min(20, n // min_size)), seed)
    return labels, [f"cluster {t + 1}" for t in range(labels.max() + 1)]


# ----------------------------------------------------------------------------- evaluation
def explained(model: FactorModel, measured: list[int]) -> float:
    """Mean over unmeasured genes (detected in >= 2% of visible cells) of the expected share of
    the gene's neighbourhood-level variance that the posterior predicts:
        (|w_g|^2 - w_g^T M w_g) / (|w_g|^2 + r_g),   M = mean_j A_j,
    the posterior factor covariance averaged over neighbourhoods (the prior gives M = I)."""
    Mbar = _precision_inverses(model, measured).mean(0)
    keep = model.detect >= EVAL_DETECTION
    keep[measured] = False
    W, r = model.W[keep], model.r[keep]
    signal = (W ** 2).sum(1)
    left = np.einsum("gk,kl,gl->g", W, Mbar, W)
    return float(np.mean((signal - left) / (signal + r))) if keep.any() else float("nan")


def run(method: str, visible: Expression, mdcla: Expression | None, target_coordinates: np.ndarray,
        target_genes: list[str], previous: list[str], eligible_ids: list[str], ng: int, rounds: int,
        seed: int, display_names: dict[str, str],
        score_round: Callable[[list[str]], dict[str, Any]] | None = None,
        ) -> tuple[list[SelectedGene], list[dict[str, Any]], dict[str, Any]]:
    """Select ng genes, then simulate rounds - 1 further rounds of the same method. Returns the
    selected genes, one record per round (round 0 = previous genes only) and model facts."""
    kind, source, _ = METHODS[method]
    if source == "mdcla" and mdcla is None:
        raise ValueError("This method needs the local mdCLA file.")
    model = fit(visible, target_coordinates, seed)          # also the yardstick for expected share
    pos = {g: i for i, g in enumerate(target_genes)}
    measured = sorted({pos[g] for g in previous if g in pos})
    eligible = [pos[g] for g in eligible_ids]

    if kind == "information":
        design_model = model if source == "visible" else fit(visible, target_coordinates, seed, mdcla)
        eligible = [g for g in eligible if visible.found[g]]

        def panel(already: list[int], allowed: list[int]) -> list[tuple[int, float, str]]:
            return information_panel(design_model, already, allowed, ng)
    else:
        data = visible if source == "visible" else mdcla
        labels, names = cell_groups(data.X, data.cell_types, seed)
        eligible = [g for g in eligible if data.found[g]]
        where = "visible embryo" if source == "visible" else "mdCLA cells"

        def panel(already: list[int], allowed: list[int]) -> list[tuple[int, float, str]]:
            return marker_panel(data.X, labels, names, already, allowed, ng, where)

    picks = panel(measured, eligible)
    selected = [
        SelectedGene(gene_id=target_genes[g], display_name=display_names.get(target_genes[g], target_genes[g]),
                     rank=i, score=score, score_provenance=why)
        for i, (g, score, why) in enumerate(picks, start=1)
    ]

    def step(round_number: int, added: list[int], so_far: list[int]) -> dict[str, Any]:
        expected = explained(model, so_far)
        record = {"round": round_number, "genes": [target_genes[g] for g in added], "n_measured": len(so_far),
                  "expected_share": round(expected, 4) if np.isfinite(expected) else None}
        if score_round is not None:
            record["benchmark"] = score_round([target_genes[g] for g in so_far])
        return record

    so_far = measured + [g for g, _, _ in picks]
    steps = [step(0, [], measured), step(1, [g for g, _, _ in picks], so_far)]
    for round_number in range(2, rounds + 1):
        left = sorted(set(eligible) - set(so_far))
        more = [g for g, _, _ in panel(so_far, left)] if left else []
        if not more:
            break
        so_far = so_far + more
        steps.append(step(round_number, more, so_far))
    facts = {"factors": model.K, "visible_neighbourhoods": model.n_visible_units,
             "target_neighbourhoods": len(model.q), "visible_transform": visible.transform,
             "mdcla_transform": mdcla.transform if mdcla is not None else None}
    return selected, steps, facts
