"""Adapter for the Morpholab_Hackathon repository.

Target
------
https://github.com/Wedenborg/Morpholab_Hackathon/tree/main/preprocessing

``panelprep`` is written to drop into that repo's ``preprocessing/`` folder.
This module is the only place that knows anything about the host repo, so the
rest of the package stays importable on its own.

What the two codebases already agree on
---------------------------------------
============================  ==========================================  ==============================
Morpholab                     panelprep                                   status
============================  ==========================================  ==============================
``_gene_key``                 :func:`panelprep.normalize.gene_key`        identical rule; verified by
                                                                          :func:`check_gene_key_agreement`
obs x genes orientation       ``PanelQuery`` / ``CandidateDataset``       identical
``REFERENCE_DATASETS``        ``panelprep.io.catalog.CATALOG``            bridged by
                                                                          :func:`reference_to_catalog`
``select_dataset`` score      score ``corr_mae`` (registered below)       same formula, now a registry row
``loadings`` (genes x d)      ``GeneEmbedding.matrix`` (G x d)            same contract; see
                                                                          :func:`loadings_to_embedding`
``gene_similarity``           panel correlation matrices                  same object, wider scope
============================  ==========================================  ==============================

So the merge is additive: ``panelprep`` contributes a score registry (in which
their single score becomes one entry), an embedder registry (in which their
factor analysis becomes one entry), the four embedding categories, the prior
sources and the figures. Nothing in ``dataset_selection.py`` has to change.

Using it
--------
>>> import panelprep.integrations.morpholab as ml
>>> ml.set_repo_root("/path/to/Morpholab_Hackathon")   # or rely on auto-discovery
>>> cand = ml.load_morpholab_dataset("tome", timepoint="7.5", layer="logcounts")

and in the other direction:

>>> result = select_dataset(adata, genes=panel, d=64)   # their function
>>> emb = ml.loadings_to_embedding(result)              # -> GeneEmbedding
>>> store.put(emb, f"factor_analysis::{result['gene_set_name']}", group="expression")

MODIFY HERE
-----------
* ``REPO_ENV_VAR`` / ``_CANDIDATE_ROOTS`` -- where to look for the repo.
* :func:`reference_to_catalog` -- how a ``REFERENCE_DATASETS`` entry maps onto a
  catalog entry (category guess, loader, layer).
* ``_DEFAULT_CATEGORY`` / ``_SPATIAL_HINTS`` -- category inference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..embed import EMBEDDERS, gene_by_cell
from ..normalize import gene_key
from ..schemas import CandidateDataset, GeneEmbedding, PanelQuery, ScoreResult
from ..scores import MIN_GENES_FOR_STRUCTURE, SCORES, prepare_panel

__all__ = [
    "set_repo_root", "repo_root", "import_morpholab", "available_datasets",
    "load_morpholab_dataset", "query_from_morpholab", "reference_to_catalog",
    "loadings_to_embedding", "similarity_to_frame", "check_gene_key_agreement",
    "corr_mae", "factor_analysis",
]

#: Environment variable checked first. MODIFY HERE.
REPO_ENV_VAR = "MORPHOLAB_ROOT"

#: Fallback locations, tried in order. The first entry supports the intended
#: layout, where this package lives at ``<repo>/preprocessing/panelprep``.
_CANDIDATE_ROOTS = (
    Path(__file__).resolve().parents[3],      # <repo>/preprocessing/panelprep/integrations
    Path.cwd() / "Morpholab_Hackathon",
    Path.cwd() / "morpholab" / "Morpholab_Hackathon-main",
)

#: Category for a Morpholab dataset whose name does not say. MODIFY HERE.
_DEFAULT_CATEGORY = "single_cell"

#: Name fragments that mark a dataset as spatially resolved.
#:
#: Checked against the 53 entries of ``dataset.load_dataset.DATASET_NAMES``: the
#: short list of assay names alone missed seven spatial datasets (``mosta``,
#: ``mpsta``, ``dbit-seq``, ``xenium-neonatal``, ``starmap-plus``, ``osmfish``,
#: ``st-magic``), which would then have been scored with single-cell scores and
#: had their spatial scores silently skipped. Covers the platform names and the
#: study acronyms that encode one.
#:
#: MODIFY HERE when the host repo registers a new spatial technology. Passing
#: ``category=`` to :func:`load_morpholab_dataset` always overrides inference,
#: and a dataset that actually carries coordinates is detected regardless.
_SPATIAL_HINTS = (
    # platforms
    "seu-3d", "seu3d", "stereo", "visium", "merfish", "slide", "slide-seq",
    "xenium", "starmap", "osmfish", "seqfish", "dbit", "cosmx", "curio",
    "spatial", "tomo-seq", "geo-seq",
    # study acronyms that denote a spatial atlas
    "spateo", "mosta", "mpsta", "st-magic",
)

_ROOT: Optional[Path] = None


# ----------------------------------------------------------------- the repo
def set_repo_root(path: str | Path) -> Path:
    """Point the adapter at a Morpholab checkout and put it on ``sys.path``."""
    global _ROOT
    root = Path(path).resolve()
    if not (root / "preprocessing").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like a Morpholab checkout "
            "(expected a 'preprocessing/' directory)")
    _ROOT = root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def repo_root(*, required: bool = True) -> Optional[Path]:
    """Locate the Morpholab checkout: env var, explicit call, then guesses."""
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    env = os.environ.get(REPO_ENV_VAR)
    candidates = ([Path(env)] if env else []) + list(_CANDIDATE_ROOTS)
    for cand in candidates:
        try:
            if (cand / "preprocessing" / "dataset_selection.py").exists():
                return set_repo_root(cand)
        except OSError:
            continue
    if required:
        raise FileNotFoundError(
            "Morpholab repo not found. Set the MORPHOLAB_ROOT environment "
            "variable or call morpholab.set_repo_root('/path/to/repo'). "
            f"Tried: {', '.join(str(c) for c in candidates)}")
    return None


def import_morpholab() -> tuple[Any, Any]:
    """Import ``(dataset_selection, load_dataset)`` from the host repo.

    Raises a clear error rather than failing later at attribute-access time,
    because those modules pull in the host repo's own dependencies.
    """
    repo_root()
    try:
        from preprocessing import dataset_selection            # type: ignore
        from dataset import load_dataset as load_dataset_mod   # type: ignore
    except Exception as exc:                                   # pragma: no cover
        raise ImportError(
            f"Found the repo at {_ROOT} but could not import its modules: {exc}. "
            "Check that its requirements are installed in this environment."
        ) from exc
    return dataset_selection, load_dataset_mod


def available_datasets() -> list[str]:
    """``DATASET_NAMES`` from the host repo's dataset registry."""
    _, load_dataset_mod = import_morpholab()
    return list(getattr(load_dataset_mod, "DATASET_NAMES", ()))


# ------------------------------------------------------------------ loading
def _guess_category(name: str, adata) -> str:
    """Spatial if the name hints at it or coordinates are actually present."""
    low = str(name).lower()
    if any(h in low for h in _SPATIAL_HINTS):
        return "spatial"
    if adata is not None:
        if "spatial" in (getattr(adata, "obsm", {}) or {}):
            return "spatial"
        obs = getattr(adata, "obs", None)
        if obs is not None and {"x", "y"} <= set(obs.columns):
            return "spatial"
    return _DEFAULT_CATEGORY


def _coords_from(adata) -> Optional[np.ndarray]:
    """2D coordinates from ``obsm`` or ``obs[['x','y']]``, else ``None``."""
    obsm = getattr(adata, "obsm", {}) or {}
    for key in ("spatial", "X_spatial", "X_sagittal"):
        if key in obsm:
            arr = np.asarray(obsm[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    obs = getattr(adata, "obs", None)
    if obs is not None and {"x", "y"} <= set(obs.columns):
        return obs[["x", "y"]].to_numpy(dtype=float)
    return None


def load_morpholab_dataset(
    dataset: str,
    *,
    timepoint: Optional[str] = None,
    layer: Optional[str] = None,
    category: Optional[str] = None,
    max_obs: Optional[int] = None,
    seed: int = 0,
    units: str = "log1p_cp10k",
    name: Optional[str] = None,
    concat: bool = True,
) -> CandidateDataset:
    """Load a registered Morpholab dataset as a :class:`CandidateDataset`.

    Uses the host repo's ``dataset.load_dataset`` (which handles downloading,
    unpacking and stage selection) and its ``convert_gene_names``, so gene
    symbols arrive in the form both codebases expect.

    Parameters
    ----------
    dataset, timepoint, layer
        Passed to ``load_dataset``; ``layer=None`` means ``X``.
    category
        One of ``panelprep.schemas.CATEGORIES``. Inferred when omitted.
    max_obs
        Subsample rows reproducibly. Several scores are sample-size sensitive
        (Moran's I markedly so), so cap query and candidates the same way.
    units
        **Declared, not verified.** ``load_dataset`` returns whatever the
        source provides; with ``layer="logcounts"`` the values are log1p of a
        normalised count, but the per-cell target sum is the source's, not
        necessarily 1e4. Unit-sensitive scores honour this string, so set it
        truthfully.
    concat
        Concatenate multiple samples of one stage (inner join on genes).
    """
    _, load_dataset_mod = import_morpholab()
    samples = load_dataset_mod.load_dataset(
        str(dataset), output="anndata", timepoint=timepoint)
    if not samples:
        raise ValueError(f"{dataset!r} (timepoint={timepoint}) returned no samples")
    adatas = [load_dataset_mod.convert_gene_names(a) for a in samples.values()]
    if len(adatas) > 1 and concat:
        import anndata as ad
        adata = ad.concat(adatas, join="inner", merge="same", label="sample",
                          keys=list(samples.keys()))
    else:
        adata = adatas[0]

    expr = adata.to_df(layer=layer)
    coords = _coords_from(adata)
    obs = adata.obs.copy()
    if max_obs is not None and expr.shape[0] > max_obs:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(expr.shape[0], size=max_obs, replace=False))
        expr = expr.iloc[idx]
        coords = None if coords is None else coords[idx]
        obs = obs.iloc[idx].copy()

    label = name or (f"{dataset}_{timepoint}" if timepoint else str(dataset))
    return CandidateDataset(
        name=label,
        category=category or _guess_category(dataset, adata),
        expression=expr,
        coords=coords,
        obs=obs,
        units=units,
        meta={
            "source": "morpholab",
            "morpholab_dataset": str(dataset),
            "timepoint": timepoint,
            "layer": layer,
            "n_samples_concatenated": len(adatas) if concat else 1,
            "sample_names": list(samples.keys()),
            "units_declared_not_verified": True,
        },
    )


def query_from_morpholab(
    dataset: str,
    panel: Sequence[str],
    descriptor: str,
    *,
    timepoint: Optional[str] = None,
    layer: Optional[str] = None,
    max_obs: Optional[int] = None,
    seed: int = 0,
    units: str = "log1p_cp10k",
    name: str = "query",
) -> PanelQuery:
    """Build the measured :class:`PanelQuery` from a Morpholab dataset.

    The query keeps **only the panel columns** -- that is the premise of the
    brief (you measured ``|S|`` genes) -- plus the 2D coordinates.
    """
    cand = load_morpholab_dataset(
        dataset, timepoint=timepoint, layer=layer, max_obs=max_obs, seed=seed,
        units=units, name=name)
    if cand.coords is None:
        raise ValueError(
            f"{dataset!r} has no 2D coordinates; a PanelQuery needs cell locations. "
            "Load it as a candidate instead, or supply coords explicitly.")
    lookup = {gene_key(c): c for c in cand.expression.columns}
    cols, missing = [], []
    for g in panel:
        hit = lookup.get(gene_key(g))
        cols.append(hit) if hit is not None else missing.append(g)
    if missing:
        raise KeyError(f"panel genes absent from {dataset!r}: {missing}")
    expr = cand.expression.loc[:, cols]
    expr.columns = list(panel)
    return PanelQuery(
        expression=expr, coords=cand.coords, descriptor=descriptor,
        units=units, obs=cand.obs, name=name,
        meta={**cand.meta, "panel_requested": list(panel)},
    )


def reference_to_catalog(reference: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn one ``REFERENCE_DATASETS`` entry into a ``panelprep`` catalog entry.

    Their entry is ``{"dataset", "timepoint", "layer"}``; ours additionally
    needs a ``name``, a ``category`` and a loader key. The loader key is
    ``"morpholab"``, served by :func:`load_morpholab_dataset`.
    """
    dataset = reference["dataset"]
    timepoint = reference.get("timepoint")
    name = f"{dataset}_{timepoint}" if timepoint else str(dataset)
    return {
        "name": name,
        "category": _guess_category(dataset, None),
        "loader": "morpholab",
        "loader_kwargs": {"dataset": dataset, "timepoint": timepoint,
                          "layer": reference.get("layer")},
        "descriptor": f"Morpholab reference {dataset}"
                      + (f" at stage {timepoint}" if timepoint else ""),
        "organism": "mus_musculus",
        "source": "morpholab/dataset.load_dataset",
    }


# ------------------------------------------------------ their output -> ours
def loadings_to_embedding(
    result: Mapping[str, Any],
    *,
    category: Optional[str] = None,
    method: str = "factor_analysis",
) -> GeneEmbedding:
    """Wrap ``select_dataset(...)['loadings']`` as a :class:`GeneEmbedding`.

    Their ``loadings`` is already ``genes x d`` indexed by gene id, which is
    exactly this package's embedding contract, so this is a metadata-only
    conversion -- no numbers are touched.
    """
    loadings = pd.DataFrame(result["loadings"])
    reference = dict(result.get("reference", {}) or {})
    dataset = reference.get("dataset", "unknown")
    return GeneEmbedding(
        matrix=loadings.astype("float32"),
        method=method,
        category=category or _guess_category(str(dataset), None),
        provenance={
            "source": "morpholab.preprocessing.dataset_selection.select_dataset",
            "model": "sklearn.decomposition.FactorAnalysis",
            "gene_set_name": result.get("gene_set_name"),
            "reference_dataset": str(dataset),
            "reference_timepoint": reference.get("timepoint"),
            "layer": reference.get("layer"),
            "selection_score_mean_abs_corr_diff": result.get("score"),
            "n_genes": int(loadings.shape[0]),
            "dim": int(loadings.shape[1]),
            "n_constant_genes_dropped": len(result.get("constant_genes", []) or []),
            "status": "implemented",
            "observation_unit": "gene",
        },
    )


def similarity_to_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """``select_dataset(...)['gene_similarity']`` as a plain gene x gene frame."""
    return pd.DataFrame(result["gene_similarity"])


def check_gene_key_agreement(samples: Sequence[str] = (
        "Sox2", "ENSMUSG00000074637.4", "T", "krt8", "ENSMUSG00000062327")
) -> pd.DataFrame:
    """Verify that both repos canonicalise gene identifiers identically.

    Returns a frame with both results and an ``agree`` column. Run this after a
    merge: a silent divergence here mis-pairs every gene in every score.
    """
    dataset_selection, _ = import_morpholab()
    theirs = [dataset_selection._gene_key(s) for s in samples]
    ours = [gene_key(s) for s in samples]
    return pd.DataFrame({"input": list(samples), "morpholab": theirs,
                         "panelprep": ours,
                         "agree": [a == b for a, b in zip(theirs, ours)]})


# --------------------------------------------------------- registry entries
@SCORES.register(
    "corr_mae",
    orientation="lower_is_better",
    applies_to=("single_cell", "spatial", "causal"),
    needs_coords=False,
    units_sensitive=False,
    defaults=dict(method="pearson"),
    description="mean absolute difference between the query and candidate panel "
                "correlation matrices (strict upper triangle); the Morpholab "
                "select_dataset score, exactly as that repo computes it",
)
def corr_mae(query, candidate, *, genes, rng, method: str = "pearson",
             max_obs: Optional[int] = None, **_) -> ScoreResult:
    """Morpholab's ``select_dataset`` score, as a registry entry.

    Their formula::

        pairs = np.triu_indices(len(genes), k=1)
        score = float(np.mean(np.abs(input_pairs - reference_pairs)))

    i.e. Pearson correlation of the panel columns within each dataset, strict
    upper triangle, mean absolute difference. Registering it puts their number
    and this package's ``corr_frobenius`` (RMS instead of MAE, Spearman instead
    of Pearson) side by side in the same comparison table, so the difference
    between the two becomes a visible choice rather than a fork in the code.

    MAE and RMS order datasets differently only when the per-pair errors are
    unevenly spread, so disagreement between these two rows is itself
    informative: it says one gene pair dominates the mismatch. Pearson vs
    Spearman matters more, because log-CP10K counts are zero-inflated.

    Observation unit: one cell contributes to a within-dataset gene-gene
    correlation; the two ``S x S`` matrices are compared entry-wise. Genes are
    paired by symbol; cells are never paired.

    MODIFY HERE: ``method`` (``pearson`` matches Morpholab; ``spearman`` is the
    defensible default for sparse counts).
    """
    qp, cp, used, missing = prepare_panel(query, candidate, genes,
                                          max_obs=max_obs, rng=rng)
    base = dict(score_name="corr_mae", dataset_name=candidate.name,
                orientation="lower_is_better", n_genes_requested=len(genes),
                n_genes_used=len(used), genes_missing=tuple(sorted(missing)))
    if len(used) < MIN_GENES_FOR_STRUCTURE:
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": f"{len(used)} usable genes, need "
                                                  f"{MIN_GENES_FOR_STRUCTURE}"}, **base)
    A, B = qp.corr(method=method), cp.corr(method=method)
    iu = np.triu_indices(len(used), k=1)
    a, b = A.to_numpy()[iu], B.to_numpy()[iu]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return ScoreResult(value=None, status="insufficient_genes",
                           diagnostics={"reason": "undefined panel correlations "
                                                  "(check for constant genes)"}, **base)
    return ScoreResult(
        value=float(np.mean(np.abs(a - b))), status="ok",
        diagnostics={"method": method, "n_pairs": int(a.size),
                     "max_abs_pair_error": float(np.max(np.abs(a - b))),
                     "rms_pair_error": float(np.sqrt(np.mean((a - b) ** 2))),
                     "n_obs": (int(qp.shape[0]), int(cp.shape[0])),
                     "equivalent_to": "morpholab select_dataset score"},
        **base)


@EMBEDDERS.register(
    "factor_analysis",
    category="single_cell",
    mode="separate",
    needs_expression=True,
    defaults=dict(max_iter=1000, standardize=False),
    description="sklearn FactorAnalysis loadings (genes x d) of the cells x genes "
                "matrix; the Morpholab select_dataset gene representation",
)
def factor_analysis(datasets, *, genes, dim, rng, max_iter: int = 1000,
                    standardize: bool = False, max_obs: Optional[int] = None,
                    **_) -> GeneEmbedding:
    """Factor-analysis loadings as a gene embedding, one per dataset.

    This reproduces what Morpholab's ``select_dataset`` returns as
    ``loadings``: ``FactorAnalysis(n_components=d)`` fitted on the
    ``cells x genes`` matrix, then ``components_.T`` -- so rows are genes.

    Note the orientation difference from this package's other expression
    embedders. ``pca_baseline`` and the autoencoders treat the **gene** as the
    observation and factor the ``G x N`` matrix; factor analysis here treats
    the **cell** as the observation and factors ``N x G``, with the gene
    representation falling out as the loading matrix. Both yield ``G x d`` and
    both are legitimate, but they are not the same estimator -- which is why
    this is a separate registry entry rather than a parameter of
    ``pca_baseline``.

    Factor analysis assumes per-gene (heteroscedastic) Gaussian noise. On
    log-CP10K counts that assumption is approximate, but it is markedly better
    than PCA's homoscedastic one for genes with very different detection rates,
    which is the usual case in a sparse spatial assay.

    Missing-data behaviour: a gene absent from this dataset is an all-NaN row,
    never a zero row -- a zero vector is a position in the space and would
    cluster all absent genes together.

    Contract: a ``mode="separate"`` embedder is called once per dataset with a
    single-entry ``datasets`` mapping and returns **one** ``GeneEmbedding``;
    ``run_embedders`` supplies the ``factor_analysis::<dataset>`` key.

    MODIFY HERE: ``max_iter`` (EM iterations) and ``standardize`` (z-score the
    genes before fitting; off by default to match Morpholab).
    """
    from sklearn.decomposition import FactorAnalysis

    usable = {k: v for k, v in datasets.items() if v.has_expression}
    if len(usable) != 1:
        raise ValueError(
            "factor_analysis is a separate-mode embedder and expects exactly one "
            f"dataset with expression per call; got {len(usable)}")
    name, dataset = next(iter(usable.items()))

    gxn, mask = gene_by_cell(dataset, genes, max_obs=max_obs, rng=rng)
    cells_by_genes = gxn.to_numpy(dtype=np.float64).T            # N x G
    if standardize:
        mu = cells_by_genes.mean(axis=0, keepdims=True)
        sd = cells_by_genes.std(axis=0, keepdims=True)
        cells_by_genes = (cells_by_genes - mu) / np.where(sd > 0, sd, 1.0)
    k = int(min(dim, cells_by_genes.shape[0], cells_by_genes.shape[1]))
    model = FactorAnalysis(n_components=k, max_iter=max_iter,
                           random_state=int(rng.integers(2 ** 31 - 1)))
    model.fit(cells_by_genes)
    matrix = pd.DataFrame(
        model.components_.T.astype("float32"), index=list(genes),
        columns=[f"factor_{i + 1}" for i in range(k)])
    matrix.loc[~mask, :] = np.nan            # absent here -> NaN, never 0
    return GeneEmbedding(
        matrix=matrix, method="factor_analysis", category=dataset.category,
        provenance={
            "dataset": name, "estimator": "sklearn FactorAnalysis",
            "observation_unit": "cell (loadings transposed to genes)",
            "n_obs": int(cells_by_genes.shape[0]),
            "n_genes_present": int(mask.sum()),
            "n_genes_absent_set_nan": int((~mask).sum()),
            "dim_requested": int(dim), "dim_used": k,
            "standardize": bool(standardize), "max_iter": int(max_iter),
            "loglike_final": (float(model.loglike_[-1])
                              if getattr(model, "loglike_", None) else None),
            "n_iter": int(getattr(model, "n_iter_", -1)),
            "status": "implemented",
            "matches": "morpholab select_dataset loadings",
        })
