"""One call that visualises every embedding you have.

:func:`visualize_all` walks an :class:`~panelprep.embed.store.EmbeddingStore`
(or a plain ``{name: GeneEmbedding}`` mapping) and renders the full set of
figures that this package knows how to make, writing each to ``outdir`` and
returning a frame of what was produced, with the numbers behind each figure.

The figures, and the question each answers
------------------------------------------
=========================  =========================================================
figure                     question
=========================  =========================================================
``panel_<group>.png``      *Within* each embedding: does it separate pathways?
                           One sub-panel per embedding, one shared colour mapping,
                           silhouette + kNN label agreement with permutation
                           p-values printed in every title.
``agreement_knn.png``      *Between* embeddings: do two methods order genes'
``agreement_cka.png``      neighbourhoods the same way (kNN-Jaccard), and do they
                           induce the same gene-gene geometry (linear CKA)?
``pair_grid.png``          Every pair of methods side by side, Procrustes-aligned.
``pair_<a>_vs_<b>.png``    One chosen pair, with arrows on the genes that move
                           most between the two embeddings.
=========================  =========================================================

Grouping
--------
Embeddings are grouped for the panel figures so that comparable things sit in
one figure. Default: by store ``group`` (``expression`` / ``external`` /
``prior``), then by method within ``expression`` -- because a figure mixing
"PCA of embryo A" with "ESM-2 of the proteome" invites a comparison that is not
meaningful (different observation sets, different feature semantics, different
categories). Pass ``group_by="method"``, ``"category"``, ``"dataset"`` or
``None`` (one figure for everything) to override.

Cost
----
Projections dominate. UMAP on ~2,000 genes takes a few seconds per embedding,
and the agreement matrices are O(n_embeddings^2) projections-free but still
O(n^2) kNN computations. With 24 embeddings expect a couple of minutes;
``method="pca"`` makes it seconds, and ``quality=False`` drops the permutation
nulls, which are the other big cost.

MODIFY HERE
-----------
* ``FIGURE_PLAN`` -- switch individual figure families on/off by default.
* ``_group_key`` -- the grouping rule.
* ``max_per_panel`` -- split a group into several figures when it is large.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from ..schemas import GeneAnnotation, GeneEmbedding
from .embeddings import (embedding_agreement_matrix, embedding_pair_grid,
                         embedding_pair_scatter, embedding_panel)

__all__ = ["visualize_all", "FIGURE_PLAN"]

#: Which figure families to render by default. MODIFY HERE.
FIGURE_PLAN: Dict[str, bool] = {
    "panels": True,          # per-group embedding_panel
    "agreement": True,       # kNN-Jaccard and CKA method x method matrices
    "pair_grid": True,       # all pairs in one figure
    "pair_focus": True,      # one highlighted pair, with movement arrows
}


def _as_mapping(source: Any, genes: Optional[Sequence[str]],
                names: Optional[Sequence[str]]) -> tuple[Dict[str, GeneEmbedding],
                                                         pd.DataFrame]:
    """Accept a store or a mapping; return ``({name: emb}, inventory)``."""
    if hasattr(source, "inventory") and hasattr(source, "load_all"):
        inv = source.inventory()
        if names is not None:
            inv = inv[inv["name"].isin(list(names))]
        embs = source.load_all(inv["name"].tolist(), genes=genes)
        return {k: v for k, v in embs.items() if v is not None}, inv
    embs = dict(source)
    if names is not None:
        embs = {k: v for k, v in embs.items() if k in set(names)}
    # A plain mapping carries no store layout, so recover the group from the
    # provenance a loaded embedding keeps (``store_group``); fall back to the
    # method name only for embeddings that were never in a store. Getting this
    # wrong mislabels a prior figure as an expression figure.
    inv = pd.DataFrame([
        {"name": k,
         "group": str(v.provenance.get("store_group",
                      "prior" if v.provenance.get("coverage_fraction") is not None
                      else "expression")),
         "method": k.split("::")[0],
         "category": v.category, "n_genes": v.matrix.shape[0],
         "dim": v.matrix.shape[1], "shape": f"{v.matrix.shape[0]} x {v.matrix.shape[1]}",
         "path": v.provenance.get("loaded_from", "")}
        for k, v in embs.items()])
    return embs, inv


def _group_key(row: pd.Series, how: Optional[str]) -> str:
    """Figure-grouping rule. MODIFY HERE to change how panels are split."""
    if how is None:
        return "all"
    if how == "method":
        return str(row["method"])
    if how == "category":
        return str(row["category"])
    if how == "dataset":
        name = str(row["name"])
        return name.split("::", 1)[1] if "::" in name else "(joint)"
    # default: store group, with expression split by method
    group = str(row.get("group", "expression"))
    return f"expression_{row['method']}" if group == "expression" else group


def visualize_all(
    source: Any,
    annotation: Optional[GeneAnnotation] = None,
    *,
    outdir: str | Path,
    names: Optional[Sequence[str]] = None,
    genes: Optional[Sequence[str]] = None,
    group_by: Optional[str] = "store_group",
    color_by: str = "top_pathway",
    method: str = "umap",
    highlight: Optional[Sequence[str]] = None,
    top_n: int = 12,
    k: int = 15,
    max_per_panel: int = 9,
    pair_focus: Optional[tuple[str, str]] = None,
    quality: bool = True,
    plan: Optional[Mapping[str, bool]] = None,
    seed: int = 0,
    dpi: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Render every figure this package knows how to make, for every embedding.

    Parameters
    ----------
    source
        An :class:`EmbeddingStore` or a ``{name: GeneEmbedding}`` mapping.
    annotation
        Pathway / role annotations used for colour. Without it the points are
        all ``unannotated`` grey and the quality numbers are omitted.
    outdir
        Directory for the PNGs; created if needed.
    genes
        Reindex every embedding to this gene list first, so the panels and the
        agreement matrices are computed over one common set. Strongly
        recommended when mixing expression embeddings (one universe) with
        priors (genome-wide).
    group_by
        ``"store_group"`` (default), ``"method"``, ``"category"``, ``"dataset"``
        or ``None``.
    highlight
        Genes drawn as large stars with labels -- pass the measured panel.
    pair_focus
        ``(name_a, name_b)`` for the arrow figure. Defaults to the first
        expression embedding vs the first joint/integrated one, which is the
        comparison that usually matters.
    quality
        Compute silhouette / kNN label agreement with permutation nulls.
        Turning this off makes the call several times faster.

    Returns
    -------
    DataFrame
        One row per figure: ``figure, kind, group, n_embeddings, path`` plus,
        where applicable, the per-embedding quality numbers joined in
        ``quality_csv``.
    """
    plan = {**FIGURE_PLAN, **(plan or {})}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    embs, inv = _as_mapping(source, genes, names)
    if not embs:
        raise ValueError("no embeddings to visualise")
    if verbose:
        print(f"[gallery] {len(embs)} embeddings -> {out}")

    inv = inv.copy()
    inv["figure_group"] = inv.apply(lambda r: _group_key(r, group_by), axis=1)
    rows: list[Dict[str, Any]] = []
    quality_frames: list[pd.DataFrame] = []

    # ---------------------------------------------------- within-embedding
    if plan["panels"]:
        for gname, sub in inv.groupby("figure_group", sort=True):
            members = [n for n in sub["name"] if n in embs]
            for part, start in enumerate(range(0, len(members), max_per_panel)):
                chunk = members[start:start + max_per_panel]
                suffix = f"_{part + 1}" if len(members) > max_per_panel else ""
                path = out / f"panel_{gname}{suffix}.png"
                if verbose:
                    print(f"  panel  {gname}{suffix}: {len(chunk)} embeddings")
                q = embedding_panel(
                    {n: embs[n] for n in chunk}, annotation, str(path),
                    color_by=color_by, method=method, top_n=top_n,
                    compute_quality=quality, highlight=highlight, seed=seed,
                    ncols=min(3, len(chunk)), dpi=dpi,
                    title=f"gene embeddings - {gname}")
                if isinstance(q, pd.DataFrame) and not q.empty:
                    q = q.copy()
                    q["figure_group"] = gname
                    quality_frames.append(q)
                rows.append({"figure": path.name, "kind": "panel", "group": gname,
                             "n_embeddings": len(chunk), "path": str(path)})

    # --------------------------------------------------- between-embeddings
    if plan["agreement"] and len(embs) >= 2:
        for metric, fname in (("knn_jaccard", "agreement_knn.png"),
                              ("cka", "agreement_cka.png")):
            path = out / fname
            if verbose:
                print(f"  agreement {metric}")
            pa = embedding_agreement_matrix(
                embs, k=k, metric=metric, out_path=str(path), seed=seed, dpi=dpi)
            long = getattr(pa, "long", None)
            if isinstance(long, pd.DataFrame) and not long.empty:
                long.to_csv(out / f"{path.stem}.csv", index=False)
            rows.append({"figure": path.name, "kind": f"agreement_{metric}",
                         "group": "all", "n_embeddings": len(embs),
                         "path": str(path)})

    if plan["pair_grid"] and len(embs) >= 2:
        path = out / "pair_grid.png"
        if verbose:
            print("  pair grid")
        embedding_pair_grid(embs, annotation, out_path=str(path),
                            max_methods=min(6, len(embs)), k=k,
                            color_by=color_by, top_n=top_n, seed=seed, dpi=dpi)
        rows.append({"figure": path.name, "kind": "pair_grid", "group": "all",
                     "n_embeddings": min(6, len(embs)), "path": str(path)})

    if plan["pair_focus"] and len(embs) >= 2:
        pair = pair_focus or _default_pair(inv, embs)
        if pair is not None and pair[0] in embs and pair[1] in embs:
            safe = f"{pair[0]}_vs_{pair[1]}".replace("::", "-").replace("/", "-")
            path = out / f"pair_{safe}.png"
            if verbose:
                print(f"  pair focus {pair[0]} vs {pair[1]}")
            embedding_pair_scatter(embs[pair[0]], embs[pair[1]], annotation,
                                   color_by=color_by, out_path=str(path),
                                   method=method, top_n=top_n, seed=seed, dpi=dpi)
            rows.append({"figure": path.name, "kind": "pair_focus",
                         "group": f"{pair[0]}|{pair[1]}", "n_embeddings": 2,
                         "path": str(path)})

    manifest = pd.DataFrame(rows)
    if quality_frames:
        quality_table = pd.concat(quality_frames, ignore_index=True)
        qpath = out / "embedding_quality.csv"
        quality_table.to_csv(qpath, index=False)
        manifest["quality_csv"] = str(qpath)
        if verbose:
            print(f"  quality table -> {qpath}")
    manifest.to_csv(out / "figure_manifest.csv", index=False)
    if verbose:
        print(f"[gallery] {len(manifest)} figures written")
    return manifest


def _default_pair(inv: pd.DataFrame, embs: Mapping[str, GeneEmbedding]
                  ) -> Optional[tuple[str, str]]:
    """First per-dataset embedding vs the first integrated one, if both exist."""
    separate = [n for n in inv["name"] if "::" in str(n) and n in embs]
    joint = [n for n in inv["name"] if "::" not in str(n) and n in embs]
    if separate and joint:
        return separate[0], joint[0]
    names = [n for n in inv["name"] if n in embs]
    return (names[0], names[1]) if len(names) >= 2 else None
