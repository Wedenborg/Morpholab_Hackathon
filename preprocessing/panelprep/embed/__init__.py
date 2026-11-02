"""Gene-embedding registry and runner.

An embedder maps one or more datasets to a ``(G, d)`` gene embedding.

Adding an embedder
------------------
::

    @EMBEDDERS.register(
        "my_embedder",
        category="single_cell",      # one of schemas.CATEGORIES
        mode="separate",             # "separate" | "joint" | "external"
        needs_expression=True,
        description="one line for the report",
    )
    def my_embedder(datasets, *, genes, dim, rng, **params) -> GeneEmbedding | dict[str, GeneEmbedding]: ...

then add the name to ``Config.embedders``.

Contract
--------
* ``datasets``: ``{name: CandidateDataset}``. ``mode="separate"`` embedders are
  called once per dataset and return one ``GeneEmbedding``; ``mode="joint"``
  embedders receive all datasets at once. ``mode="external"`` embedders ignore
  the matrices and use ``genes`` only (ESM, PPI, pathways).
* Input to expression embedders is ``G x N_k``: genes as *rows*. The runner
  transposes the stored observations x genes matrices for you via
  ``gene_by_cell``.
* Missing genes: ``gene_by_cell`` fills genes absent from a dataset with a
  mask, and embedders must honour ``mask`` (see ``panelprep.embed.expression``).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..normalize import resolve_genes
from ..registry import Registry, stable_seed
from ..schemas import CandidateDataset, GeneEmbedding

EMBEDDERS = Registry("embedder")

__all__ = ["EMBEDDERS", "gene_by_cell", "gene_universe", "run_embedders",
           "describe", "compare"]


def gene_by_cell(
    dataset: CandidateDataset,
    genes: Sequence[str],
    *,
    max_obs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    fill: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(G x N)`` genes-by-observations matrix plus a presence mask.

    Rows are ``genes`` in the given order. Genes absent from the dataset are
    filled with ``fill`` and flagged ``False`` in the mask so embedders can
    exclude them from the loss instead of learning from fabricated zeros.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not dataset.has_expression:
        raise ValueError(f"{dataset.name}: no expression matrix")
    expr = dataset.expression
    if max_obs is not None and expr.shape[0] > max_obs:
        idx = np.sort(rng.choice(expr.shape[0], size=max_obs, replace=False))
        expr = expr.iloc[idx]
    names, missing = resolve_genes(genes, expr.columns)
    mask = np.array([n is not None for n in names], dtype=bool)
    out = np.full((len(genes), expr.shape[0]), fill, dtype=np.float32)
    present = [n for n in names if n is not None]
    if present:
        out[mask] = expr.loc[:, present].to_numpy(dtype=np.float32).T
    return pd.DataFrame(out, index=list(genes)), mask


def gene_universe(datasets: Dict[str, CandidateDataset], how: str = "union",
                  max_genes: Optional[int] = None,
                  rng: Optional[np.random.Generator] = None,
                  must_include: Optional[Sequence[str]] = None) -> list[str]:
    """Gene list the embedders run over (item 0: not just the measured panel).

    ``must_include`` genes are kept whatever ``max_genes`` says. Pass the
    measured panel here: ``max_genes`` subsamples the universe at random, and a
    random 2,000 of ~16,000 genes will almost surely drop all four panel genes
    -- which silently defeats the point of the deliverable, since locating the
    measured panel relative to the unmeasured transcriptome is exactly what the
    embedding is for. Genes in ``must_include`` that are absent from the
    datasets cannot be recovered and are skipped.

    MODIFY HERE: ``how`` (union vs intersection) and ``max_genes`` (cap for
    cheap runs; ``None`` = the full universe).
    """
    sets = [set(d.genes) for d in datasets.values() if d.has_expression]
    if not sets:
        return []
    genes = set.union(*sets) if how == "union" else set.intersection(*sets)
    pinned = sorted(genes.intersection({str(g) for g in (must_include or [])}))
    rest = sorted(genes.difference(pinned))
    if max_genes is not None and len(pinned) + len(rest) > max_genes:
        rng = np.random.default_rng(0) if rng is None else rng
        room = max(max_genes - len(pinned), 0)
        if room < len(rest):
            rest = sorted(np.array(rest)[rng.choice(len(rest), room, replace=False)])
    return sorted(set(pinned) | set(rest))


def run_embedders(
    datasets: Dict[str, CandidateDataset],
    *,
    embedder_names: Sequence[str],
    genes: Sequence[str],
    dim: int = 32,
    params: Optional[dict] = None,
    seed: int = 0,
    max_obs: Optional[int] = None,
) -> Dict[str, GeneEmbedding]:
    """Run the requested embedders; keys are ``"<embedder>"`` or ``"<embedder>::<dataset>"``."""
    params = params or {}
    out: Dict[str, GeneEmbedding] = {}
    for name in embedder_names:
        entry = EMBEDDERS[name]
        kw = dict(entry.meta.get("defaults", {}))
        kw.update(params.get(name, {}))
        if max_obs is not None:
            kw.setdefault("max_obs", max_obs)
        mode = entry.meta.get("mode", "joint")
        rng = np.random.default_rng([seed, stable_seed(name)])
        if mode == "separate":
            usable = {k: v for k, v in datasets.items() if v.has_expression}
            for dname, ds in usable.items():
                emb = entry.fn({dname: ds}, genes=list(genes), dim=dim, rng=rng, **kw)
                out[f"{name}::{dname}"] = emb
        else:
            emb = entry.fn(datasets, genes=list(genes), dim=dim, rng=rng, **kw)
            if isinstance(emb, dict):
                out.update({f"{name}::{k}": v for k, v in emb.items()})
            else:
                out[name] = emb
    return out


def describe(embeddings: Dict[str, GeneEmbedding]) -> "pd.DataFrame":
    """Summarise what a :func:`run_embedders` call returned, one row per embedding.

    Answers the three questions you ask of any embedding dict: *what came back*
    (key, method, mode, which dataset), *what shape is it* (genes x dim), and
    *how much of it is actually populated* (NaN rows, which are genes the
    source could not cover).

    Columns
    -------
    ``key``
        The dict key: ``<method>`` for a joint embedder, ``<method>::<dataset>``
        for a separate one.
    ``mode``
        From the registry: ``separate`` = one matrix per dataset, ``joint`` =
        one matrix built from all datasets at once.
    ``trained_on``
        The dataset name, or the number of views for a joint embedder (read from
        ``provenance['views']`` when the embedder records it).
    ``n_genes``, ``dim``, ``shape``
        The matrix geometry. ``n_genes`` is always ``len(genes)`` as passed in:
        the index is exactly the gene list you asked for, in that order.
    ``n_nan_genes``
        Rows that are entirely NaN -- genes absent from the source. Never zeros.
    ``status``
        ``provenance['status']`` where the embedder sets it.

    Example
    -------
    >>> out = run_embedders(datasets, embedder_names=["pca_baseline", "dcca"],
    ...                     genes=genes, dim=32)
    >>> describe(out)[["key", "mode", "shape", "n_nan_genes"]]
    """
    import pandas as pd

    rows = []
    for key, emb in embeddings.items():
        method = key.split("::")[0]
        dataset = key.split("::", 1)[1] if "::" in key else None
        mode = EMBEDDERS[method].meta.get("mode", "?") if method in EMBEDDERS else "?"
        views = emb.provenance.get("views")
        n_views = len(views) if isinstance(views, (list, tuple)) else None
        rows.append({
            "key": key,
            "method": method,
            "mode": mode,
            "trained_on": dataset if dataset is not None else (
                f"{n_views} views (integrated)" if n_views else "all datasets"),
            "category": emb.category,
            "n_genes": int(emb.matrix.shape[0]),
            "dim": int(emb.matrix.shape[1]),
            "shape": f"{emb.matrix.shape[0]} x {emb.matrix.shape[1]}",
            "n_nan_genes": int(emb.matrix.isna().all(axis=1).sum()),
            "dtype": str(emb.matrix.dtypes.iloc[0]) if emb.matrix.shape[1] else "",
            "status": str(emb.provenance.get("status", "")),
        })
    return pd.DataFrame(rows).sort_values(["mode", "key"]).reset_index(drop=True)


def compare(a: GeneEmbedding, b: GeneEmbedding, *, metric: str = "cka",
            k: int = 15) -> float:
    """How similar are two gene embeddings, over the genes they share?

    Both metrics are invariant to the things that should not matter -- rotation,
    reflection and the arbitrary ordering of the ``d`` columns -- so a value
    near 1 means the two embeddings really do place genes in the same geometry,
    not merely that they have the same shape.

    ``metric='cka'``
        Linear centred kernel alignment of the two gene-gene similarity
        matrices. In ``[0, 1]``. Sensitive to the global geometry.
    ``metric='knn_jaccard'``
        Mean Jaccard overlap of each gene's ``k`` nearest neighbours. In
        ``[0, 1]``. Sensitive to the *local* neighbourhoods, which is usually
        the thing a downstream method actually consumes, and is a much harsher
        test: real-data values of 0.02-0.16 are normal where CKA reads 0.9+.

    Genes that are NaN in either embedding are dropped pairwise. Returns ``nan``
    if fewer than ``k + 1`` genes survive.

    There is **no null model here** -- use
    :func:`panelprep.viz.embedding_agreement_matrix`, which computes the same
    numbers against a gene-label-shuffled null, before calling any value high
    or low.
    """
    from ..viz.embeddings import _cka, _knn_jaccard, _shared_matrices

    Xa, Xb, shared = _shared_matrices(a, b)
    if Xa is None or len(shared) < k + 1:
        return float("nan")
    if metric == "cka":
        return float(_cka(Xa, Xb))
    if metric == "knn_jaccard":
        return float(_knn_jaccard(Xa, Xb, k))
    raise ValueError(f"metric must be 'cka' or 'knn_jaccard'; got {metric!r}")


from . import expression, joint, external, custom  # noqa: E402,F401
