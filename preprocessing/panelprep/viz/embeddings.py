"""Gene-embedding projections, coloured by annotation, with a number attached.

The brief asks for "beautiful expressive visualizations of embeddings, cluster
and color them by pathway information". A scatter on its own cannot answer
*which embedding separates pathways*, because any 2D projection of any matrix
looks structured. So every renderer here is paired with
:func:`embedding_quality`, which measures how well the annotation labels are
organised in the embedding space and compares that against a label-permutation
null. The panel titles carry those numbers.

What is an observation here
---------------------------
One **gene**. ``GeneEmbedding.matrix`` is ``(G, d)``: genes are rows, embedding
dimensions are columns. This is the transposed flavour of the expression
matrices (where rows are cells) and nothing in this module ever transposes
implicitly -- it indexes genes by the embedding's own index and joins the
annotation by gene symbol via ``panelprep.normalize.gene_key``.

Quality is measured in the embedding space, not the picture
-----------------------------------------------------------
``silhouette`` and the kNN label agreement are computed on the full ``d``-
dimensional matrix by default (``space="embedding"``). UMAP/t-SNE distort
neighbourhoods to make a readable picture, so scoring the 2D projection would
measure the projection, not the embedding. ``space="projection"`` is available
when the 2D view itself is the object of interest.

Missing data
------------
A gene with no embedding (an all-``NaN`` row -- an embedder's honest way of
saying "this gene was not in my dataset") is dropped from the projection and
from the quality numbers, and the count is printed in the subtitle. Genes with
a partially non-finite row are dropped too and counted separately. Nothing is
imputed, and a dropped gene never becomes a point at the origin.

MODIFY HERE
-----------
* ``method`` -- ``"umap"`` / ``"tsne"`` / ``"pca"``; ``PROJECTION_DEFAULTS``
  holds the per-method keyword arguments.
* ``top_n`` -- how many annotation categories keep their own colour before the
  rest are pooled into "other".
* ``ROLE_MARKERS`` -- the ligand / receptor / TF marker shapes.
* ``embedding_quality(..., n_neighbors=, n_permutations=, min_per_class=)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import GeneAnnotation, GeneEmbedding

#: Per-method projection keywords. MODIFY HERE to retune a projection.
PROJECTION_DEFAULTS: Dict[str, dict] = {
    "umap": dict(n_neighbors=15, min_dist=0.3, metric="euclidean"),
    "tsne": dict(perplexity=30.0, init="pca", learning_rate="auto"),
    "pca": dict(),
}

#: Marker shapes used by ``role_overlay``. Keys are columns of
#: ``GeneAnnotation.table``; MODIFY HERE to add a role column.
ROLE_MARKERS: Dict[str, tuple] = {
    "is_ligand": ("^", "ligand"),
    "is_receptor": ("v", "receptor"),
    "is_tf": ("s", "TF"),
}

OTHER_LABEL = "other"
UNANNOTATED_LABEL = "unannotated"
OTHER_COLOUR = "#d9d9d9"
UNANNOTATED_COLOUR = "#efefef"


# ---------------------------------------------------------------------------
# helpers shared by the renderers
# ---------------------------------------------------------------------------

def _import_umap() -> tuple[Any, Optional[str]]:
    """Import ``umap``, working around numba's cache locator; never raises.

    ``umap`` JIT-compiles through numba, which refuses to run when it cannot
    place its cache next to the installed source ("no locator available").
    That is an environment property, not a data problem, so it is fixed by
    pointing ``NUMBA_CACHE_DIR`` at a writable directory -- only when the
    variable is unset, so a user's own setting is never overridden. Returns
    ``(module_or_None, note)``; the caller turns a ``None`` into a documented
    PCA fallback rather than an exception.

    MODIFY HERE: set ``NUMBA_CACHE_DIR`` yourself to choose the cache location.
    """
    import os
    import tempfile

    try:
        import umap
        return umap, None
    except Exception as exc:
        if "NUMBA_CACHE_DIR" in os.environ:
            return None, f"umap unavailable ({type(exc).__name__}: {exc})"
    import sys

    cache = str(Path(tempfile.gettempdir()) / "panelprep_numba_cache")
    os.environ["NUMBA_CACHE_DIR"] = cache
    try:
        # numba reads its config at import time, so the new variable only takes
        # effect after an explicit reload; the half-imported umap modules have
        # to go too, or the retry re-uses the broken ones.
        import numba.core.config as _nbcfg
        _nbcfg.reload_config()
    except Exception:
        pass
    for m in [m for m in list(sys.modules) if m == "umap" or m.startswith("umap.")]:
        del sys.modules[m]
    try:
        import umap
        return umap, f"set NUMBA_CACHE_DIR={cache} so umap could compile"
    except Exception as exc:
        return None, f"umap unavailable ({type(exc).__name__}: {str(exc)[:120]})"


def _style_rc() -> dict:
    return {
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
    }


def clean_matrix(embedding: GeneEmbedding) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Drop genes with no usable embedding row; report how many and why.

    Observation unit: one gene (a row). Returns ``(matrix, counts)`` where
    ``counts`` has ``n_total``, ``n_kept``, ``n_all_nan`` (no embedding at all)
    and ``n_partial_nan`` (some dimension non-finite). Nothing is imputed.

    MODIFY HERE: nothing to tune; change the *embedder* if too many genes are
    dropped -- a high ``n_all_nan`` means the gene was absent from that
    dataset, which is information, not a bug.
    """
    M = pd.DataFrame(embedding.matrix).astype(float)
    finite = np.isfinite(M.to_numpy())
    all_bad = ~finite.any(axis=1)
    part_bad = (~finite.all(axis=1)) & ~all_bad
    keep = finite.all(axis=1)
    return M.loc[keep], {
        "n_total": int(M.shape[0]), "n_kept": int(keep.sum()),
        "n_all_nan": int(all_bad.sum()), "n_partial_nan": int(part_bad.sum()),
        "dim": int(M.shape[1]),
    }


def annotation_labels(
    genes: Sequence[str],
    annotation: Optional[GeneAnnotation],
    column: str = "top_pathway",
) -> pd.Series:
    """Per-gene label series aligned to ``genes``; unmatched -> ``"unannotated"``.

    Observation unit: one gene. Genes are joined by canonical gene key, so an
    annotation table written in upper case still matches a mixed-case
    embedding index. Missing-data behaviour: a gene absent from the annotation
    table, or with a null label, becomes ``"unannotated"`` and is drawn in pale
    grey -- it is never silently assigned to a real category.

    MODIFY HERE: ``column`` -- any column of ``GeneAnnotation.table``
    (``top_pathway`` by default; ``is_tf`` and friends work too).
    """
    from ..normalize import gene_key

    idx = pd.Index([gene_key(g) for g in genes])
    if annotation is None:
        return pd.Series([UNANNOTATED_LABEL] * len(genes), index=list(genes), dtype=object)
    tab = pd.DataFrame(annotation.table)
    if column not in tab.columns:
        raise KeyError(f"annotation table has no column {column!r}; "
                       f"available: {list(tab.columns)}")
    keyed = tab.copy()
    keyed.index = [gene_key(g) for g in tab.index]
    keyed = keyed[~keyed.index.duplicated(keep="first")]
    vals = keyed[column].reindex(idx)
    out = vals.astype(object).where(vals.notna(), UNANNOTATED_LABEL)
    out = out.map(lambda v: UNANNOTATED_LABEL if (isinstance(v, str) and not v.strip()) else v)
    out.index = list(genes)
    return out.astype(object)


def category_palette(
    labels: pd.Series,
    *,
    top_n: int = 12,
    cmap: str = "tab20",
) -> tuple[Dict[Any, str], List[Any]]:
    """Colour per category, keeping the ``top_n`` most frequent ones.

    Returns ``(colour_by_label, ordered_labels)``. Everything beyond ``top_n``
    is pooled into ``"other"`` (light grey) and ``"unannotated"`` keeps its own
    paler grey, so pooled genes can never be mistaken for a named pathway. The
    same mapping is reused across the panels of :func:`embedding_panel`, which
    is what makes the panels comparable at a glance.

    MODIFY HERE: ``top_n`` and ``cmap``.
    """
    import matplotlib as mpl

    counts = labels[labels != UNANNOTATED_LABEL].value_counts()
    keep = list(counts.index[:top_n])
    base = mpl.colormaps[cmap]
    colours = {lab: mpl.colors.to_hex(base(i % base.N)) for i, lab in enumerate(keep)}
    colours[OTHER_LABEL] = OTHER_COLOUR
    colours[UNANNOTATED_LABEL] = UNANNOTATED_COLOUR
    order = keep + [OTHER_LABEL, UNANNOTATED_LABEL]
    return colours, order


def _pool(labels: pd.Series, keep: Sequence[Any]) -> pd.Series:
    keep = set(keep)
    return labels.map(lambda v: v if (v in keep or v == UNANNOTATED_LABEL) else OTHER_LABEL)


def project(
    embedding: GeneEmbedding,
    *,
    method: str = "umap",
    seed: int = 0,
    params: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Project a ``(G, d)`` gene embedding to ``(G_kept, 2)``.

    Observation unit: one gene. Normalisation assumption: columns are used as
    the embedder produced them; no scaling is applied, because an embedder's
    dimension scaling is part of its output (PCA components are meant to be
    unequal). Missing-data behaviour: see :func:`clean_matrix` -- unusable
    genes are dropped and counted, never imputed.

    Falls back to PCA, with the reason recorded in ``info["fallback"]``, when
    UMAP/t-SNE cannot run (package missing, or fewer genes than the method
    needs). A fallback is a status, not an exception.

    MODIFY HERE: ``method`` and ``PROJECTION_DEFAULTS[method]``.
    """
    if method not in PROJECTION_DEFAULTS:
        raise ValueError(f"method must be one of {sorted(PROJECTION_DEFAULTS)}; got {method!r}")
    M, counts = clean_matrix(embedding)
    info: Dict[str, Any] = {"method_requested": method, "method_used": method, **counts}
    n = M.shape[0]
    if n == 0:
        return pd.DataFrame(columns=[0, 1]), {**info, "fallback": "no gene has an embedding"}
    X = M.to_numpy(dtype=float)
    kw = dict(PROJECTION_DEFAULTS[method])
    kw.update(params or {})

    umap_mod = None
    if n < 3 or X.shape[1] < 2:
        method, info["fallback"] = "pca", f"only {n} genes x {X.shape[1]} dims"
    if method == "umap":
        umap_mod, note = _import_umap()
        if umap_mod is None:
            method, info["fallback"] = "pca", note
        elif note:
            info["umap_note"] = note
    if method == "umap" and n <= kw.get("n_neighbors", 15):
        kw["n_neighbors"] = max(2, min(n - 1, 15))
        info["n_neighbors_reduced_to"] = kw["n_neighbors"]
    if method == "tsne":
        max_perp = max(5.0, (n - 1) / 3.0)
        if kw.get("perplexity", 30.0) > max_perp:
            kw["perplexity"] = float(min(30.0, max_perp))
            info["perplexity_reduced_to"] = kw["perplexity"]

    if method == "umap":
        Y = umap_mod.UMAP(n_components=2, random_state=seed, **kw).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        Y = TSNE(n_components=2, random_state=seed, **kw).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        p = PCA(n_components=2, random_state=seed)
        Y = p.fit_transform(X)
        info["pca_explained_variance_ratio"] = [round(float(v), 4)
                                                for v in p.explained_variance_ratio_]
    info["method_used"] = method
    return pd.DataFrame(np.asarray(Y)[:, :2], index=M.index, columns=[0, 1]), info


# ---------------------------------------------------------------------------
# the number that must accompany every scatter
# ---------------------------------------------------------------------------

def embedding_quality(
    embedding: GeneEmbedding,
    annotation: Optional[GeneAnnotation],
    label_col: str = "top_pathway",
    *,
    space: str = "embedding",
    n_neighbors: int = 15,
    n_permutations: int = 50,
    min_per_class: int = 3,
    metric: str = "euclidean",
    seed: int = 0,
    projection: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """How well the annotation labels are organised in this embedding.

    Observation unit: one **gene**. Two statistics, each against a
    label-permutation null (the null that answers "would any labelling look
    this good here?"):

    ``silhouette``
        mean silhouette coefficient of the label partition, in ``[-1, 1]``.
        Global: it asks whether a pathway's genes are closer to each other than
        to the nearest other pathway.
    ``knn_agreement``
        mean fraction of a gene's ``n_neighbors`` nearest neighbours carrying
        its own label, in ``[0, 1]``. Local, and readable as "how often does a
        gene's neighbourhood share its pathway". The permutation null is not
        ``1/n_classes``: class sizes are unequal, so the null is computed, not
        assumed.

    Normalisation assumption: distances are Euclidean on the embedding as
    produced (``space="embedding"``, the full ``d`` dimensions) -- the 2D
    projection is only scored when ``space="projection"``, which measures the
    picture rather than the embedding.

    Missing-data behaviour: genes without a usable embedding row are dropped
    (:func:`clean_matrix`); ``unannotated`` genes and classes with fewer than
    ``min_per_class`` genes are dropped from the statistics and counted in the
    returned dict; if fewer than 2 classes or 3 genes survive, the status is
    ``insufficient_labels`` and the values are ``None`` -- never 0.

    Returns a dict with the statistics, their null mean/sd, one-sided
    permutation p-values, and the gene/class counts behind them.

    MODIFY HERE: ``n_neighbors`` (locality of the kNN statistic),
    ``n_permutations`` (null resolution; p cannot be smaller than
    ``1/(n_permutations+1)``), ``min_per_class``, ``space``.
    """
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.neighbors import NearestNeighbors

    out: Dict[str, Any] = {
        "method": embedding.method, "category": embedding.category,
        "label_col": label_col, "space": space, "status": "ok",
        "silhouette": None, "silhouette_null_mean": None, "silhouette_null_sd": None,
        "silhouette_p": None, "knn_agreement": None, "knn_null_mean": None,
        "knn_null_sd": None, "knn_p": None, "n_genes_scored": 0, "n_classes": 0,
        "n_permutations": int(n_permutations),
    }
    if space == "projection":
        if projection is None:
            raise ValueError("space='projection' needs the projection frame")
        M = pd.DataFrame(projection).astype(float)
        counts = {"n_total": int(M.shape[0]), "n_kept": int(M.shape[0]),
                  "n_all_nan": 0, "n_partial_nan": 0, "dim": 2}
    elif space == "embedding":
        M, counts = clean_matrix(embedding)
    else:
        raise ValueError("space must be 'embedding' or 'projection'")
    out.update({f"genes_{k}": v for k, v in counts.items()})
    out["n_neighbors"] = int(n_neighbors)

    if annotation is None:
        out["status"] = "no_annotation"
        return out
    labels = annotation_labels(list(M.index), annotation, label_col).astype(str)
    usable = labels != UNANNOTATED_LABEL
    vc = labels[usable].value_counts()
    small = set(vc.index[vc < min_per_class])
    usable &= ~labels.isin(small)
    out["n_genes_unannotated"] = int((labels == UNANNOTATED_LABEL).sum())
    out["n_genes_in_small_classes"] = int(labels.isin(small).sum())

    X = M.loc[usable.to_numpy()].to_numpy(dtype=float)
    y = labels[usable.to_numpy()].to_numpy()
    out["n_genes_scored"], out["n_classes"] = int(X.shape[0]), int(len(set(y)))
    if out["n_classes"] < 2 or X.shape[0] < 3:
        out["status"] = "insufficient_labels"
        return out

    rng = np.random.default_rng(seed)
    D = pairwise_distances(X, metric=metric)
    k = int(max(1, min(n_neighbors, X.shape[0] - 1)))
    out["n_neighbors"] = k
    nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(D)
    idx = nn.kneighbors(D, return_distance=False)[:, 1:]        # drop self

    def _knn(lab: np.ndarray) -> float:
        return float((lab[idx] == lab[:, None]).mean())

    sil_obs = float(silhouette_score(D, y, metric="precomputed"))
    knn_obs = _knn(y)
    sil_null, knn_null = [], []
    for _ in range(int(n_permutations)):
        perm = rng.permutation(y)
        sil_null.append(float(silhouette_score(D, perm, metric="precomputed")))
        knn_null.append(_knn(perm))
    sil_null, knn_null = np.asarray(sil_null), np.asarray(knn_null)

    out.update({
        "silhouette": round(sil_obs, 4),
        "silhouette_null_mean": round(float(sil_null.mean()), 4),
        "silhouette_null_sd": round(float(sil_null.std(ddof=1)), 4),
        "silhouette_p": round(float((1 + (sil_null >= sil_obs).sum()) / (1 + len(sil_null))), 4),
        "knn_agreement": round(knn_obs, 4),
        "knn_null_mean": round(float(knn_null.mean()), 4),
        "knn_null_sd": round(float(knn_null.std(ddof=1)), 4),
        "knn_p": round(float((1 + (knn_null >= knn_obs).sum()) / (1 + len(knn_null))), 4),
    })
    return out


def quality_caption(q: Mapping[str, Any]) -> str:
    """One-line "the number attached to the picture"."""
    if q.get("status") != "ok":
        return f"quality: {q.get('status')}"
    return (f"silhouette {q['silhouette']:+.3f} (null {q['silhouette_null_mean']:+.3f}, "
            f"p={q['silhouette_p']:.3f})  |  kNN-{q['n_neighbors']} label agreement "
            f"{q['knn_agreement']:.3f} (null {q['knn_null_mean']:.3f}, p={q['knn_p']:.3f})"
            f"  |  n={q['n_genes_scored']} genes, {q['n_classes']} classes")


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

@dataclass
class ScatterResult:
    """What was drawn, and the numbers that justify it."""

    coords: pd.DataFrame
    labels: pd.Series
    info: Dict[str, Any]
    quality: Dict[str, Any]
    palette: Dict[Any, str]
    fig: Any = None
    out_path: Optional[str] = None


def _axis_arrows(ax, name: str) -> None:
    """Corner arrow pair instead of meaningless projection ticks (per §6.6)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate("", xy=(0.17, 0.02), xytext=(0.02, 0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8))
    ax.annotate("", xy=(0.02, 0.17), xytext=(0.02, 0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8))
    ax.text(0.19, 0.02, f"{name} 1", transform=ax.transAxes, fontsize=6, color="#777777",
            ha="left", va="center")
    ax.text(0.02, 0.19, f"{name} 2", transform=ax.transAxes, fontsize=6, color="#777777",
            ha="left", va="bottom", rotation=90)


def _draw_points(ax, coords, labels, palette, order, *, size=14, alpha=0.85):
    for lab in order:
        m = (labels == lab).to_numpy()
        if not m.any():
            continue
        ax.scatter(coords.to_numpy()[m, 0], coords.to_numpy()[m, 1],
                   s=size, c=palette[lab], linewidths=0.2, edgecolors="#ffffff",
                   alpha=alpha, label=f"{lab} ({int(m.sum())})", zorder=2)


def _draw_roles(ax, coords, annotation, genes, *, size=30):
    """Open marker shapes on top of the coloured points; returns legend handles."""
    from matplotlib.lines import Line2D
    from ..normalize import gene_key

    handles = []
    if annotation is None:
        return handles
    tab = pd.DataFrame(annotation.table).copy()
    tab.index = [gene_key(g) for g in tab.index]
    tab = tab[~tab.index.duplicated(keep="first")]
    keys = [gene_key(g) for g in genes]
    for col, (marker, label) in ROLE_MARKERS.items():
        if col not in tab.columns:
            continue
        flag = tab[col].reindex(keys).fillna(False).to_numpy().astype(bool)
        if not flag.any():
            continue
        ax.scatter(coords.to_numpy()[flag, 0], coords.to_numpy()[flag, 1], s=size,
                   facecolors="none", edgecolors="#222222", linewidths=0.7,
                   marker=marker, zorder=3)
        handles.append(Line2D([], [], marker=marker, linestyle="none", markerfacecolor="none",
                              markeredgecolor="#222222",
                              label=f"{label} ({int(flag.sum())})"))
    return handles


def _draw_highlight(ax, coords, labels, palette, genes, *, size=260, label_points=True,
                    fontsize=7.0):
    """Mark named genes with a large solid star, filled by their own category colour.

    Used to show **where the measured panel genes sit** inside a gene embedding
    built over the whole gene universe. The star is filled with the same
    ``color_by`` colour as the surrounding points -- so it carries the pathway
    annotation rather than overriding it -- and given a dark edge plus a high
    ``zorder`` so it stays legible on top of a few thousand small markers.

    Observation unit: one gene per star. Missing-data behaviour: a requested
    gene with no embedding row is simply absent from ``coords`` and is skipped;
    the returned handle reports how many of the requested genes were drawn.

    MODIFY HERE: ``size`` (star area), ``label_points`` (gene symbol text),
    and the ``marker`` below if a star is not wanted.
    """
    from matplotlib.lines import Line2D

    from ..normalize import gene_key

    if not genes:
        return []
    want = {gene_key(g): str(g) for g in genes}
    keys = [gene_key(g) for g in coords.index]
    rows = [i for i, k in enumerate(keys) if k in want]
    if not rows:
        return []
    xy = coords.to_numpy()[rows]
    cols = [palette.get(str(labels.iloc[i]), "#777777") for i in rows]
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=cols, marker="*",
               edgecolors="#111111", linewidths=1.1, zorder=6)
    if label_points:
        for (x, y), i in zip(xy, rows):
            ax.annotate(str(coords.index[i]), (x, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=fontsize, fontweight="bold",
                        color="#111111", zorder=7,
                        path_effects=None)
    return [Line2D([], [], marker="*", linestyle="none", markersize=11,
                   markerfacecolor="#cccccc", markeredgecolor="#111111",
                   label=f"measured panel gene ({len(rows)}/{len(want)})")]


def _draw_gene_labels(ax, coords, genes: Optional[Sequence[str]] = None, *,
                      n_auto: int = 0, fontsize: float = 6.0,
                      skip: Sequence[str] = (), max_labels: int = 80,
                      min_sep: float = 0.045, color: str = "#333333") -> int:
    """Write gene symbols next to their points, with collision avoidance.

    Two ways to choose which genes get named:

    ``genes``
        An explicit list of symbols. Matching is on
        :func:`panelprep.normalize.gene_key`, so case and Ensembl version
        suffixes do not matter; a symbol with no embedding row is skipped.
    ``n_auto``
        Label the ``n_auto`` genes **furthest from the centroid** of the
        projection. Those are the ones a reader can actually resolve -- the
        dense core of a few thousand genes cannot be labelled legibly at any
        font size, and naming an arbitrary subset of it would suggest those
        genes were selected for a reason.

    Labels are placed greedily in that order and one is dropped whenever it
    falls within ``min_sep`` (in axis-fraction units) of an already-placed
    label, so the text stays readable instead of overprinting itself.

    Returns the number of labels actually drawn.

    Observation unit: one gene per label. This function adds **text only**; it
    never changes a point's position, colour or size, so it cannot alter what
    the figure claims.

    MODIFY HERE: ``min_sep`` (how aggressively labels are thinned), ``max_labels``
    (hard ceiling), ``fontsize``, and the selection rule in the ``n_auto``
    branch if "most peripheral" is not the subset you want named.
    """
    from ..normalize import gene_key

    if coords is None or len(coords) == 0:
        return 0
    xy_all = coords.to_numpy(dtype=float)
    names = [str(g) for g in coords.index]
    skip_keys = {gene_key(g) for g in (skip or ())}

    if genes:
        want = {gene_key(g) for g in genes}
        rows = [i for i, n in enumerate(names)
                if gene_key(n) in want and gene_key(n) not in skip_keys]
    elif n_auto > 0:
        centre = np.nanmean(xy_all, axis=0)
        span = np.nanmax(xy_all, axis=0) - np.nanmin(xy_all, axis=0)
        span = np.where(span > 0, span, 1.0)
        dist = np.linalg.norm((xy_all - centre) / span, axis=1)
        order = np.argsort(-dist)
        rows = [int(i) for i in order if gene_key(names[i]) not in skip_keys]
    else:
        return 0

    rows = rows[:max_labels]
    if not rows:
        return 0

    # Collision filter in axis-fraction units, so min_sep means the same thing
    # whatever the data range is.
    lo, hi = np.nanmin(xy_all, axis=0), np.nanmax(xy_all, axis=0)
    rng_xy = np.where((hi - lo) > 0, hi - lo, 1.0)
    placed: list[np.ndarray] = []
    drawn = 0
    for i in rows:
        p = (xy_all[i] - lo) / rng_xy
        if not np.all(np.isfinite(p)):
            continue
        if any(float(np.linalg.norm(p - q)) < min_sep for q in placed):
            continue
        placed.append(p)
        ax.annotate(names[i], (xy_all[i, 0], xy_all[i, 1]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=fontsize, color=color, zorder=5,
                    annotation_clip=True)
        drawn += 1
    return drawn


def embedding_scatter(
    embedding: GeneEmbedding,
    annotation: Optional[GeneAnnotation] = None,
    *,
    color_by: str = "top_pathway",
    method: str = "umap",
    out_path: Optional[str] = None,
    top_n: int = 12,
    role_overlay: bool = False,
    highlight: Optional[Sequence[str]] = None,
    highlight_size: float = 260.0,
    label_genes: Optional[Sequence[str]] = None,
    n_gene_labels: int = 0,
    label_fontsize: float = 6.0,
    compute_quality: bool = True,
    title: Optional[str] = None,
    seed: int = 0,
    point_size: float = 14.0,
    quality_kwargs: Optional[Mapping[str, Any]] = None,
    projection_params: Optional[Mapping[str, Any]] = None,
    figsize: tuple = (6.6, 4.4),
    dpi: int = 200,
    close: bool = True,
) -> ScatterResult:
    """2D projection of one gene embedding, coloured by an annotation column.

    Observation unit: one **gene** per point. Normalisation assumption: the
    embedding is projected as produced (see :func:`project`); the colour is a
    categorical annotation, not a measurement. Missing-data behaviour: genes
    with no usable embedding row are dropped and the count is printed in the
    subtitle; genes missing from the annotation are drawn pale grey as
    ``unannotated``; categories beyond ``top_n`` are pooled as ``other`` in
    light grey. The subtitle always carries the silhouette and kNN numbers
    (with their permutation nulls) so the picture is never the only evidence.

    ``highlight`` marks the named genes (typically the measured panel) with a
    large solid star filled by their own ``color_by`` colour, so their position
    inside the whole-transcriptome embedding is visible at a glance.

    MODIFY HERE: ``method`` (umap/tsne/pca), ``color_by`` (any column of
    ``GeneAnnotation.table``), ``top_n``, ``role_overlay``, ``highlight``,
    ``highlight_size``, ``point_size``, ``label_genes`` (name these genes on the
    plot), ``n_gene_labels`` (name the N most peripheral genes), ``label_fontsize``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    coords, info = project(embedding, method=method, seed=seed, params=projection_params)
    if coords.empty:
        raise ValueError(f"{embedding.method}: no gene has a usable embedding row "
                         f"({info.get('n_all_nan')} all-NaN of {info.get('n_total')})")
    labels_full = annotation_labels(list(coords.index), annotation, color_by).astype(str)
    palette, order = category_palette(labels_full, top_n=top_n)
    labels = _pool(labels_full, [o for o in order if o not in (OTHER_LABEL, UNANNOTATED_LABEL)])

    q = {}
    if compute_quality:
        q = embedding_quality(embedding, annotation, color_by, seed=seed,
                              **dict(quality_kwargs or {}))

    with mpl.rc_context(_style_rc()):
        fig, ax = plt.subplots(figsize=figsize)
        _draw_points(ax, coords, labels, palette, order, size=point_size)
        handles = _draw_roles(ax, coords, annotation, list(coords.index)) if role_overlay else []
        handles += _draw_highlight(ax, coords, labels, palette, list(highlight or []),
                                   size=highlight_size)
        # Gene symbols on the points themselves (text only; nothing moves).
        _draw_gene_labels(ax, coords, label_genes, n_auto=n_gene_labels,
                          fontsize=label_fontsize, skip=list(highlight or []))
        _axis_arrows(ax, info["method_used"].upper())
        ax.margins(0.06)
        for sp in ax.spines.values():
            sp.set_visible(False)

        head = title or (f"{embedding.method} ({embedding.category}) coloured by {color_by}")
        drop = info["n_total"] - info["n_kept"]
        sub = (f"{info['n_kept']}/{info['n_total']} genes projected with "
               f"{info['method_used']} from {info['dim']}d"
               + (f"; {drop} dropped ({info['n_all_nan']} with no embedding, "
                  f"{info['n_partial_nan']} partially non-finite)" if drop else "; none dropped"))
        if info.get("fallback"):
            sub += f"; fell back to PCA: {str(info['fallback'])[:70]}"
        if q:
            sub += "\n" + quality_caption(q)
        ax.set_title(f"{head}\n{sub}", fontsize=8, loc="left")

        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                        fontsize=7, title=color_by, title_fontsize=7, handletextpad=0.3,
                        labelspacing=0.3, borderpad=0.2)
        if handles:
            ax.add_artist(leg)
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0),
                      frameon=False, fontsize=7, title="role", title_fontsize=7,
                      handletextpad=0.3, labelspacing=0.3, borderpad=0.2)

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if close and out_path:
            plt.close(fig)

    return ScatterResult(coords=coords, labels=labels, info=info, quality=q,
                         palette=palette, fig=None if (close and out_path) else fig,
                         out_path=str(out_path) if out_path else None)


def embedding_panel(
    embeddings: Mapping[str, GeneEmbedding],
    annotation: Optional[GeneAnnotation] = None,
    out_path: Optional[str] = None,
    *,
    color_by: str = "top_pathway",
    method: str = "umap",
    top_n: int = 12,
    ncols: int = 3,
    compute_quality: bool = True,
    role_overlay: bool = False,
    highlight: Optional[Sequence[str]] = None,
    highlight_size: float = 210.0,
    label_genes: Optional[Sequence[str]] = None,
    n_gene_labels: int = 0,
    label_fontsize: float = 6.0,
    seed: int = 0,
    point_size: float = 10.0,
    sort_by_quality: bool = True,
    title: Optional[str] = None,
    panel_size: tuple = (3.1, 3.0),
    dpi: int = 200,
    close: bool = True,
) -> pd.DataFrame:
    """Grid of embeddings under one shared colour mapping.

    Answers "which embedding separates pathways" at a glance *and* by number:
    the colour mapping is built once over the union of the plotted genes, so
    the panels are comparable, and every panel title carries its silhouette and
    kNN label agreement with permutation p-values. Panels are ordered by
    silhouette (best first) unless ``sort_by_quality=False``.

    Observation unit: one gene per point, within each panel. Missing-data
    behaviour: as :func:`embedding_scatter`, per panel, with the dropped count
    in each panel's subtitle; an embedding with no usable gene gets an empty
    panel stating that, rather than being skipped silently.

    Returns the quality table (one row per embedding) -- the figure is never
    the only output.

    ``highlight`` marks the named genes (typically the measured panel) with a
    large solid star in every panel, filled by their own ``color_by`` colour.

    MODIFY HERE: ``method``, ``ncols``, ``top_n``, ``panel_size``,
    ``sort_by_quality``, ``highlight``, ``highlight_size``, ``label_genes``,
    ``n_gene_labels``, ``label_fontsize``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    names = list(embeddings)
    if not names:
        raise ValueError("no embeddings to plot")

    projections, infos, quality = {}, {}, {}
    for n in names:
        try:
            coords, info = project(embeddings[n], method=method, seed=seed)
        except Exception as exc:                      # a projection failure is a status
            coords, info = pd.DataFrame(columns=[0, 1]), {
                "method_requested": method, "method_used": method, "n_total": 0, "n_kept": 0,
                "n_all_nan": 0, "n_partial_nan": 0, "dim": 0,
                "error": f"{type(exc).__name__}: {exc}"}
        projections[n], infos[n] = coords, info
        quality[n] = (embedding_quality(embeddings[n], annotation, color_by, seed=seed)
                      if compute_quality else {"status": "not_computed"})

    all_labels = pd.concat([annotation_labels(list(projections[n].index), annotation, color_by)
                            for n in names]).astype(str)
    palette, order = category_palette(all_labels, top_n=top_n)
    keep = [o for o in order if o not in (OTHER_LABEL, UNANNOTATED_LABEL)]

    qdf = pd.DataFrame([{"embedding": n, **{k: v for k, v in quality[n].items()},
                         "n_genes_plotted": int(projections[n].shape[0]),
                         "n_genes_total": int(infos[n].get("n_total", 0)),
                         "projection": infos[n].get("method_used", method)} for n in names])
    if sort_by_quality and "silhouette" in qdf.columns:
        qdf = qdf.sort_values("silhouette", ascending=False, na_position="last") \
                 .reset_index(drop=True)
        names = list(qdf["embedding"])

    # 4 panels read better 2x2 than 3+1; MODIFY HERE to force a layout
    ncols = 2 if len(names) <= 4 else ncols
    ncols = int(max(1, min(ncols, len(names))))
    nrows = int(np.ceil(len(names) / ncols))
    with mpl.rc_context(_style_rc()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols + 1.8,
                                                        panel_size[1] * nrows + 0.9))
        axes = np.atleast_1d(axes).ravel()
        for ax, n in zip(axes, names):
            coords = projections[n]
            if coords.empty:
                ax.text(0.5, 0.5, f"no usable gene\n{infos[n].get('error', '')}",
                        ha="center", va="center", fontsize=7, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                lab = _pool(annotation_labels(list(coords.index), annotation, color_by)
                            .astype(str), keep)
                _draw_points(ax, coords, lab, palette, order, size=point_size, alpha=0.9)
                if role_overlay:
                    _draw_roles(ax, coords, annotation, list(coords.index), size=22)
                _draw_highlight(ax, coords, lab, palette, list(highlight or []),
                                size=highlight_size, fontsize=6.5)
                _draw_gene_labels(ax, coords, label_genes, n_auto=n_gene_labels,
                                  fontsize=label_fontsize,
                                  skip=list(highlight or []), min_sep=0.065)
                _axis_arrows(ax, infos[n].get("method_used", method).upper())
                ax.margins(0.06)
            q = quality[n]
            num = (f"sil {q['silhouette']:+.3f} (p={q['silhouette_p']:.3f}) | "
                   f"kNN {q['knn_agreement']:.3f} (p={q['knn_p']:.3f})"
                   if q.get("status") == "ok" else f"quality: {q.get('status')}")
            drop = infos[n].get("n_total", 0) - infos[n].get("n_kept", 0)
            ax.set_title(f"{n}\n{num}\n{infos[n].get('n_kept', 0)} genes"
                         + (f", {drop} dropped" if drop else ""),
                         fontsize=7.5, loc="left")
            for sp in ax.spines.values():
                sp.set_visible(False)
        for ax in axes[len(names):]:
            ax.axis("off")

        # legend resolves exactly the categories that were drawn (pooled "other"
        # and "unannotated" appear only if some gene actually fell into them)
        drawn = set(_pool(all_labels, keep).unique())
        shown = [o for o in order if o in drawn]
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=palette[o], markeredgecolor="none", label=str(o))
                   for o in shown]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5),
                   frameon=False, fontsize=7, title=color_by, title_fontsize=7,
                   labelspacing=0.3, handletextpad=0.3)
        used = sorted({infos[n].get("method_used", method) for n in names})
        used_txt = "/".join(used) + (f" (requested {method})" if used != [method] else "")
        head = title or (f"Gene embeddings projected with {used_txt}, coloured by {color_by}")
        fig.suptitle(f"{head}\nshared colour mapping; panel titles carry silhouette and kNN "
                     f"label agreement against a label-permutation null",
                     fontsize=8.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 0.995, 0.93))

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if close and out_path:
            plt.close(fig)
    return qdf


# ===========================================================================
# PAIRWISE comparison between embedding methods
# ===========================================================================
# "Which embedding separates pathways" (embedding_panel) and "do two sources
# agree about gene geometry" are different questions. The three functions below
# answer the second one. All three work on the genes shared by the pair -- a
# gene with no embedding in either method cannot contribute an opinion -- and
# all three report the size of that intersection, because two methods that
# cover only a handful of shared genes can look identical for no good reason.
#
# MODIFY HERE: ``PAIR_METRICS`` to add an agreement metric; everything else
# (the matrix, the figure, the null) keys off the name.
# ===========================================================================

#: Agreement metrics for ``embedding_agreement_matrix``; all are
#: higher = more agreement. MODIFY HERE to register another one.
PAIR_METRICS: tuple[str, ...] = ("knn_jaccard", "cka", "mantel")


@dataclass
class PairAgreement:
    """Method x method agreement, with the null and the coverage behind it."""

    matrix: pd.DataFrame          #: observed agreement, higher = more agreement
    null: pd.DataFrame            #: mean agreement under gene-label permutation
    n_shared: pd.DataFrame        #: genes usable in BOTH methods, per pair
    metric: str
    k: int
    n_permutations: int
    long: pd.DataFrame            #: one row per pair: value, null mean/sd, p, n_shared
    fig: Any = None
    out_path: Optional[str] = None


def _shared_matrices(a: GeneEmbedding, b: GeneEmbedding) -> tuple[np.ndarray, np.ndarray, list]:
    """``(X_a, X_b, genes)`` restricted to genes usable in BOTH embeddings."""
    from ..normalize import gene_key

    Ma, _ = clean_matrix(a)
    Mb, _ = clean_matrix(b)
    ka = {gene_key(g): g for g in Ma.index}
    kb = {gene_key(g): g for g in Mb.index}
    shared = [k for k in ka if k in kb]
    if not shared:
        return np.empty((0, 0)), np.empty((0, 0)), []
    Xa = Ma.loc[[ka[k] for k in shared]].to_numpy(dtype=float)
    Xb = Mb.loc[[kb[k] for k in shared]].to_numpy(dtype=float)
    return Xa, Xb, [ka[k] for k in shared]


def _knn_sets(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = int(max(1, min(k, X.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]       # drop self


def _knn_jaccard(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    A, B = _knn_sets(Xa, k), _knn_sets(Xb, k)
    out = np.empty(A.shape[0], dtype=float)
    for i in range(A.shape[0]):
        sa, sb = set(A[i]), set(B[i])
        out[i] = len(sa & sb) / len(sa | sb)
    return float(out.mean())


def _cka(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Linear centred kernel alignment between two gene x gene Gram matrices.

    Invariant to rotation and isotropic rescaling of either embedding, which is
    exactly the invariance wanted here: an autoencoder's latent basis is
    arbitrary (any rotation of the latent space gives the same model), so two
    methods must not be called different merely for choosing different axes.
    """
    A = Xa - Xa.mean(0, keepdims=True)
    B = Xb - Xb.mean(0, keepdims=True)
    # HSIC on linear kernels == ||A^T B||_F^2 for column-centred A, B
    num = float(np.linalg.norm(A.T @ B, ord="fro") ** 2)
    da = float(np.linalg.norm(A.T @ A, ord="fro"))
    db = float(np.linalg.norm(B.T @ B, ord="fro"))
    return float(num / (da * db)) if da > 0 and db > 0 else float("nan")


def _mantel(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Spearman correlation of the two flattened gene-gene distance matrices."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    da, db = pdist(Xa), pdist(Xb)
    if da.size < 3 or np.allclose(da, da[0]) or np.allclose(db, db[0]):
        return float("nan")
    return float(spearmanr(da, db).statistic)


def _pair_value(Xa: np.ndarray, Xb: np.ndarray, metric: str, k: int) -> float:
    if metric == "knn_jaccard":
        return _knn_jaccard(Xa, Xb, k)
    if metric == "cka":
        return _cka(Xa, Xb)
    if metric == "mantel":
        return _mantel(Xa, Xb)
    raise ValueError(f"metric must be one of {PAIR_METRICS}; got {metric!r}")


def embedding_agreement_matrix(
    embeddings: Mapping[str, GeneEmbedding],
    *,
    k: int = 15,
    metric: str = "knn_jaccard",
    out_path: Optional[str] = None,
    n_permutations: int = 10,
    seed: int = 0,
    annotate: bool = True,
    cmap: str = "RdBu_r",
    min_shared: int = 5,
    dpi: int = 200,
    close: bool = True,
) -> PairAgreement:
    """Method x method agreement about gene geometry, against a shuffled null.

    Observation unit: one **gene**, within each pair; the unit of the reported
    matrix is one *pair of methods*. All three metrics run higher = more
    agreement, so the colour direction is unambiguous (unlike the score table,
    no orientation correction is needed here).

    ``knn_jaccard``
        mean over shared genes of the Jaccard overlap between a gene's ``k``
        nearest neighbours in A and in B. Local, interpretable as "how much of
        each gene's neighbourhood survives the change of method".
    ``cka``
        linear centred kernel alignment of the two gene x gene Gram matrices;
        invariant to rotation and isotropic scaling, the right invariance for
        latent spaces whose basis is arbitrary (see :func:`_cka`).
    ``mantel``
        Spearman correlation of the flattened gene-gene distance matrices;
        global, and sensitive to the whole distance profile rather than the
        neighbourhoods.

    Normalisation assumption: the embeddings are compared as produced, with no
    rescaling -- ``cka`` is scale-free by construction, while ``knn_jaccard``
    and ``mantel`` depend only on the ranking of distances, so none of the
    three needs the two methods to share a scale.

    Missing-data behaviour: each pair uses only the genes usable in **both**
    embeddings; that intersection size is returned in ``n_shared``, printed in
    every cell and summarised in the title, because a pair sharing few genes
    can agree trivially. A pair sharing fewer than ``min_shared`` genes gets
    ``NaN`` and a grey cell rather than a number.

    The permutation baseline shuffles the gene labels of the second embedding
    (``n_permutations`` times) to show what "no agreement" looks like for this
    metric, this ``k`` and this gene count -- it is not assumed to be 0.

    Returns a :class:`PairAgreement` (matrix, null, n_shared, long-format rows
    and p-values) with the figure attached.

    MODIFY HERE: ``metric``, ``k`` (neighbourhood size for ``knn_jaccard``),
    ``n_permutations``, ``min_shared``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from .score_table import STATUS_COLOUR, _text_colour

    if metric not in PAIR_METRICS:
        raise ValueError(f"metric must be one of {PAIR_METRICS}; got {metric!r}")
    names = list(embeddings)
    if len(names) < 2:
        raise ValueError("need at least two embeddings to compare")

    rng = np.random.default_rng(seed)
    M = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    N = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    S = pd.DataFrame(0, index=names, columns=names, dtype=int)
    rows = []
    for i, a in enumerate(names):
        M.iat[i, i], N.iat[i, i] = 1.0, np.nan          # self-agreement by definition
        S.iat[i, i] = clean_matrix(embeddings[a])[1]["n_kept"]
        for b in names[i + 1:]:
            Xa, Xb, shared = _shared_matrices(embeddings[a], embeddings[b])
            n = len(shared)
            S.loc[a, b] = S.loc[b, a] = n
            if n < max(min_shared, 3):
                rows.append({"method_a": a, "method_b": b, "metric": metric, "value": None,
                             "null_mean": None, "null_sd": None, "p": None, "n_shared": n,
                             "status": "insufficient_shared_genes"})
                continue
            val = _pair_value(Xa, Xb, metric, k)
            nulls = []
            for _ in range(int(n_permutations)):
                nulls.append(_pair_value(Xa, Xb[rng.permutation(n)], metric, k))
            nulls = np.asarray(nulls, dtype=float)
            nmean = float(np.nanmean(nulls)) if nulls.size else float("nan")
            M.loc[a, b] = M.loc[b, a] = val
            N.loc[a, b] = N.loc[b, a] = nmean
            p = (float((1 + np.nansum(nulls >= val)) / (1 + nulls.size))
                 if nulls.size and np.isfinite(val) else None)
            rows.append({"method_a": a, "method_b": b, "metric": metric,
                         "value": round(float(val), 4) if np.isfinite(val) else None,
                         "null_mean": round(nmean, 4) if np.isfinite(nmean) else None,
                         "null_sd": round(float(np.nanstd(nulls, ddof=1)), 4)
                                    if nulls.size > 1 else None,
                         "p": None if p is None else round(p, 4),
                         "n_shared": n, "status": "ok"})
    long = pd.DataFrame(rows)

    off = M.to_numpy(dtype=float)[~np.eye(len(names), dtype=bool)]
    lo = float(np.nanmin(off)) if np.isfinite(off).any() else 0.0
    hi = float(np.nanmax(off)) if np.isfinite(off).any() else 1.0
    vmin, vmax = (min(0.0, lo), max(1.0, hi)) if metric != "mantel" else (-1.0, 1.0)

    with mpl.rc_context(_style_rc()):
        fig, ax = plt.subplots(figsize=(1.45 * len(names) + 3.0, 1.0 * len(names) + 2.4))
        cm = mpl.colormaps[cmap].with_extremes(bad=STATUS_COLOUR)
        im = ax.imshow(np.ma.masked_invalid(M.to_numpy(dtype=float)), cmap=cm,
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(names)), labels=names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), labels=names)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if annotate:
            for i in range(len(names)):
                for j in range(len(names)):
                    v = M.iat[i, j]
                    if not np.isfinite(v):
                        ax.text(j, i, f"n={S.iat[i, j]}", ha="center", va="center",
                                fontsize=6.5, color="#333333", style="italic")
                        continue
                    col = _text_colour(cm((v - vmin) / (vmax - vmin)))
                    if i == j:
                        ax.text(j, i, "1 (self)", ha="center", va="center", fontsize=7,
                                color=col, style="italic")
                        continue
                    ax.text(j, i - 0.16, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                            color=col)
                    ax.text(j, i + 0.2, f"null {N.iat[i, j]:.3f} | n={S.iat[i, j]}",
                            ha="center", va="center", fontsize=6, color=col)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(f"{metric} agreement\n(red = the two methods agree)")
        cbar.outline.set_visible(False)
        shared_off = S.to_numpy()[~np.eye(len(names), dtype=bool)]
        ax.set_title(
            f"Agreement between gene embeddings: {metric}"
            + (f" (k={k})" if metric == "knn_jaccard" else "")
            + f"\n{len(names)} methods; shared genes per pair "
              f"{int(shared_off.min())}-{int(shared_off.max())}; "
              f"'null' = gene labels of one method shuffled ({n_permutations}x)",
            fontsize=8, loc="left", pad=8)
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if close and out_path:
            plt.close(fig)

    return PairAgreement(matrix=M, null=N, n_shared=S, metric=metric, k=int(k),
                         n_permutations=int(n_permutations), long=long,
                         fig=None if (close and out_path) else fig,
                         out_path=str(out_path) if out_path else None)


@dataclass
class PairScatter:
    """What two methods say about the same genes, and how far apart they are."""

    coords_a: pd.DataFrame
    coords_b: pd.DataFrame
    aligned_a: pd.DataFrame        #: standardised A (Procrustes frame)
    aligned_b: pd.DataFrame        #: B rotated onto A
    disparity: float               #: Procrustes disparity, 0 = identical
    displacement: pd.DataFrame     #: per-gene distance after alignment, sorted
    n_shared: int
    info: Dict[str, Any]
    fig: Any = None
    out_path: Optional[str] = None


def embedding_pair_scatter(
    emb_a: GeneEmbedding,
    emb_b: GeneEmbedding,
    annotation: Optional[GeneAnnotation] = None,
    *,
    color_by: str = "top_pathway",
    align: str = "procrustes",
    out_path: Optional[str] = None,
    n_arrows: int = 40,
    method: str = "umap",
    top_n: int = 12,
    seed: int = 0,
    point_size: float = 12.0,
    label_arrows: bool = True,
    figsize: Optional[tuple] = None,
    dpi: int = 200,
    close: bool = True,
) -> PairScatter:
    """Two methods side by side, plus the genes they disagree about.

    Observation unit: one **gene**, present in both embeddings. Panels 1 and 2
    are the two 2D projections of the *same* shared genes with the *same*
    colours; panel 3 overlays them after orthogonal Procrustes alignment of B
    onto A and draws an arrow per gene for the ``n_arrows`` genes that move
    furthest -- those are the genes the two sources disagree about, which is
    the question behind comparing two embeddings.

    Normalisation assumption: Procrustes removes translation, rotation,
    reflection and isotropic scale (``scipy.spatial.procrustes`` standardises
    both configurations to unit Frobenius norm first). That is the correct
    invariance class: a projection's orientation and overall size carry no
    biology, while the *relative* positions of genes do. The reported
    ``disparity`` is the residual sum of squared distances after that
    alignment, 0 = identical configurations; it is printed in the title so the
    picture is not the only evidence. ``align="none"`` skips the alignment and
    compares the raw projections, which is only meaningful if both came from
    the same projector with the same seed.

    Missing-data behaviour: genes missing from either embedding are excluded,
    and the shared count is in the title; genes missing from the annotation are
    drawn pale grey as ``unannotated``.

    MODIFY HERE: ``method`` (umap/tsne/pca -- the same projector is used for
    both embeddings so the comparison is fair), ``n_arrows``, ``align``,
    ``label_arrows``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.spatial import procrustes

    Xa, Xb, shared = _shared_matrices(emb_a, emb_b)
    n = len(shared)
    if n < 3:
        raise ValueError(f"only {n} genes are usable in both embeddings "
                         f"({emb_a.method}, {emb_b.method}); nothing to compare")

    ea = GeneEmbedding(pd.DataFrame(Xa, index=shared), method=emb_a.method,
                       category=emb_a.category)
    eb = GeneEmbedding(pd.DataFrame(Xb, index=shared), method=emb_b.method,
                       category=emb_b.category)
    ca, ia = project(ea, method=method, seed=seed)
    cb, ib = project(eb, method=method, seed=seed)
    common = [g for g in ca.index if g in set(cb.index)]
    ca, cb = ca.loc[common], cb.loc[common]

    if align == "procrustes":
        m1, m2, disparity = procrustes(ca.to_numpy(), cb.to_numpy())
    elif align == "none":
        m1, m2 = ca.to_numpy(), cb.to_numpy()
        disparity = float(np.sum((m1 - m2) ** 2))
    else:
        raise ValueError("align must be 'procrustes' or 'none'")
    A2 = pd.DataFrame(m1, index=common, columns=[0, 1])
    B2 = pd.DataFrame(m2, index=common, columns=[0, 1])
    d = np.linalg.norm(A2.to_numpy() - B2.to_numpy(), axis=1)
    disp = pd.DataFrame({"gene": common, "displacement": d}) \
        .sort_values("displacement", ascending=False).reset_index(drop=True)

    labels_full = annotation_labels(common, annotation, color_by).astype(str)
    palette, order = category_palette(labels_full, top_n=top_n)
    keep = [o for o in order if o not in (OTHER_LABEL, UNANNOTATED_LABEL)]
    labels = _pool(labels_full, keep)

    figsize = figsize or (12.6, 4.3)
    with mpl.rc_context(_style_rc()):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, coords, emb, info in ((axes[0], ca, emb_a, ia), (axes[1], cb, emb_b, ib)):
            _draw_points(ax, coords, labels, palette, order, size=point_size)
            _axis_arrows(ax, info.get("method_used", method).upper())
            ax.margins(0.07)
            ax.set_title(f"{emb.method} ({emb.category})\n{len(coords)} shared genes",
                         fontsize=7.5, loc="left")

        ax = axes[2]
        ax.scatter(A2[0], A2[1], s=point_size, c=[palette[v] for v in labels],
                   linewidths=0.2, edgecolors="#ffffff", alpha=0.75, zorder=2)
        ax.scatter(B2[0], B2[1], s=point_size, facecolors="none",
                   edgecolors=[palette[v] for v in labels], linewidths=0.6, zorder=2)
        movers = list(disp["gene"].head(int(n_arrows)))
        for g in movers:
            x0, y0 = A2.loc[g]
            x1, y1 = B2.loc[g]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.6,
                                        shrinkA=0, shrinkB=0, alpha=0.8), zorder=3)
        if label_arrows:
            # Label at the A end (the arrow TAIL): an unrelated B collapses
            # toward the centroid under Procrustes, so labels placed at the
            # heads pile up on top of each other and name nothing.
            for g in movers[:min(len(movers), 10)]:          # only the worst ten
                x0, y0 = A2.loc[g]
                ax.text(x0, y0, f" {g}", fontsize=5.5, color="#222222", va="center",
                        ha="left", zorder=4)
        _axis_arrows(ax, "PROCRUSTES" if align == "procrustes" else "RAW")
        ax.margins(0.1)
        ax.set_title(f"{emb_b.method} aligned onto {emb_a.method}\n"
                     f"disparity {disparity:.4f} (0 = identical); arrows: "
                     f"{len(movers)} largest movers, max {disp['displacement'].iloc[0]:.3f}, "
                     f"median {float(np.median(d)):.3f}",
                     fontsize=7.5, loc="left")
        for a_ in axes:
            for sp in a_.spines.values():
                sp.set_visible(False)

        from matplotlib.lines import Line2D
        drawn = set(labels.unique())
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=palette[o], markeredgecolor="none", label=str(o))
                   for o in order if o in drawn]
        handles += [Line2D([], [], marker="o", linestyle="none", markersize=4,
                           markerfacecolor="#777777", markeredgecolor="none",
                           label=f"filled = {emb_a.method}"),
                    Line2D([], [], marker="o", linestyle="none", markersize=4,
                           markerfacecolor="none", markeredgecolor="#777777",
                           label=f"open = {emb_b.method}")]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5),
                   frameon=False, fontsize=7, title=color_by, title_fontsize=7,
                   labelspacing=0.3, handletextpad=0.3)
        fig.suptitle(f"Do {emb_a.method} and {emb_b.method} agree about the same genes?"
                     f"  Procrustes disparity {disparity:.4f} over {len(common)} shared genes",
                     fontsize=8.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 0.995, 0.92))
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if close and out_path:
            plt.close(fig)

    return PairScatter(coords_a=ca, coords_b=cb, aligned_a=A2, aligned_b=B2,
                       disparity=float(disparity), displacement=disp, n_shared=len(common),
                       info={"method_a": emb_a.method, "method_b": emb_b.method,
                             "projection": ia.get("method_used", method), "align": align,
                             "median_displacement": float(np.median(d)),
                             "max_displacement": float(disp["displacement"].iloc[0])},
                       fig=None if (close and out_path) else fig,
                       out_path=str(out_path) if out_path else None)


def embedding_pair_grid(
    embeddings: Mapping[str, GeneEmbedding],
    annotation: Optional[GeneAnnotation] = None,
    *,
    out_path: Optional[str] = None,
    max_methods: int = 6,
    method: str = "pca",
    k: int = 15,
    color_by: str = "top_pathway",
    top_n: int = 12,
    min_points: int = 10,
    seed: int = 0,
    point_size: float = 5.0,
    panel_size: tuple = (2.0, 2.0),
    dpi: int = 200,
    close: bool = True,
) -> pd.DataFrame:
    """Lower-triangular grid of every method pair, in one figure.

    Observation unit: one **gene** per point inside a cell; one cell is one
    pair of methods. Each cell plots the two methods' first projected
    coordinate against each other for the genes they share, and prints
    ``|rho|`` (absolute Spearman correlation) plus the shared-gene count. The
    absolute value is deliberate: the sign of a projection axis is arbitrary
    (PCA components and UMAP axes may flip), so a sign would be an artefact.

    When a pair shares fewer than ``min_points`` genes a scatter would be
    unreadable, so the cell prints the kNN-Jaccard agreement large instead --
    with the shared count next to it, since that is what makes the value
    trustworthy or not.

    Normalisation assumption: all embeddings are projected with the *same*
    projector and seed, so cells differ because the embeddings differ, not
    because the projections do. Missing-data behaviour: per cell, only genes
    usable in both embeddings; methods beyond ``max_methods`` are dropped (the
    ones with the fewest usable genes first) and named in the returned frame.

    Returns the per-pair statistics table (methods, rho, kNN-Jaccard, shared
    genes) -- the figure is never the only output.

    MODIFY HERE: ``max_methods``, ``method`` (projector), ``k``, ``min_points``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    names = list(embeddings)
    usable = {n: clean_matrix(embeddings[n])[1]["n_kept"] for n in names}
    dropped: list[str] = []
    if len(names) > max_methods:
        names = [n for n, _ in sorted(usable.items(), key=lambda kv: -kv[1])[:max_methods]]
        dropped = [n for n in embeddings if n not in names]
    if len(names) < 2:
        raise ValueError("need at least two embeddings to compare")

    proj: Dict[str, pd.DataFrame] = {}
    for n in names:
        c, _ = project(embeddings[n], method=method, seed=seed)
        proj[n] = c

    rows = []
    nrows = len(names) - 1
    with mpl.rc_context(_style_rc()):
        fig, axes = plt.subplots(nrows, nrows,
                                 figsize=(panel_size[0] * nrows + 1.6,
                                          panel_size[1] * nrows + 1.4),
                                 squeeze=False)
        for r in range(nrows):
            for c in range(nrows):
                ax = axes[r][c]
                if c > r:                                   # upper triangle unused
                    ax.axis("off")
                    continue
                a, b = names[c], names[r + 1]
                Xa, Xb, shared = _shared_matrices(embeddings[a], embeddings[b])
                n_sh = len(shared)
                pa = proj[a].reindex(shared).dropna()
                pb = proj[b].reindex(shared).dropna()
                common = [g for g in pa.index if g in set(pb.index)]
                rho = float("nan")
                jac = float("nan")
                if n_sh >= 3:
                    jac = _knn_jaccard(Xa, Xb, k)
                if len(common) >= min_points:
                    x, y = pa.loc[common, 0].to_numpy(), pb.loc[common, 0].to_numpy()
                    rho = float(spearmanr(x, y).statistic)
                    lab = annotation_labels(common, annotation, color_by).astype(str)
                    palette, _ = category_palette(lab, top_n=top_n)
                    cols = [palette.get(v, OTHER_COLOUR) for v in lab]
                    ax.scatter(x, y, s=point_size, c=cols, linewidths=0, alpha=0.8)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.02, 0.98, f"|rho| {abs(rho):.2f}\nJ {jac:.2f}\nn={n_sh}",
                            transform=ax.transAxes, ha="left", va="top", fontsize=6,
                            color="#222222")
                else:
                    ax.text(0.5, 0.58, "-" if not np.isfinite(jac) else f"{jac:.2f}",
                            transform=ax.transAxes, ha="center", va="center", fontsize=13,
                            color="#222222")
                    ax.text(0.5, 0.3, f"kNN-Jaccard\nn={n_sh} shared", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6, color="#555555")
                    ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#cccccc")
                if c == 0:
                    ax.set_ylabel(b, fontsize=6.5)
                if r == nrows - 1:
                    ax.set_xlabel(a, fontsize=6.5)
                rows.append({"method_a": a, "method_b": b,
                             "abs_spearman_coord1": None if not np.isfinite(rho)
                                                    else round(abs(rho), 4),
                             "knn_jaccard": None if not np.isfinite(jac) else round(jac, 4),
                             "n_shared": n_sh, "n_plotted": len(common),
                             "projection": method})
        fig.suptitle("Every pair of gene embeddings: first projected coordinate of one against "
                     "the other\n|rho| = absolute Spearman (axis signs are arbitrary); "
                     f"J = mean kNN-Jaccard (k={k}); n = genes shared by the pair"
                     + (f"; dropped {len(dropped)} method(s) over max_methods={max_methods}"
                        if dropped else ""),
                     fontsize=8, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if close and out_path:
            plt.close(fig)

    out = pd.DataFrame(rows)
    out.attrs["dropped_methods"] = dropped
    return out
