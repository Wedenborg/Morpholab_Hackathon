"""The panel-embedding dashboard: where your measured genes sit, everywhere.

One question, asked of every reference dataset and every embedding method at
once: **given that I measured these few genes, what does each dataset think
they are near?**

:func:`panel_dashboard` renders

* a ``methods x datasets`` **grid** of 2D projections, one shared colour mapping
  across the whole grid, with the measured panel drawn as large labelled stars;
* a **per-method** figure for the joint/integrated embeddings, which have no
  per-dataset counterpart;
* a **panel-neighbourhood table**: for each embedding, which genes are nearest
  to each measured gene -- the part you can act on;
* an **HTML index** tying them together with the score table.

Why a grid rather than one plot per figure: the colour mapping is built once
over the union of plotted genes, so a colour means the same pathway in every
cell of the grid, and rows (methods) and columns (datasets) are directly
comparable. Reading a stack of separately-coloured figures is how you conclude
two embeddings disagree when only their palettes did.

What a projection does and does not show
---------------------------------------
UMAP/t-SNE distances are not metric and cluster sizes carry no meaning. The
dashboard therefore prints, in every panel title, the silhouette and kNN label
agreement **against a label-permutation null** -- so a panel that looks
structured but scores at chance says so. The neighbourhood table is computed in
the full ``d``-dimensional space, never from the 2D coordinates.

MODIFY HERE
-----------
* ``MAX_GRID_CELLS`` -- the guard against a grid too large to read.
* ``NEIGHBOURS_PER_GENE`` -- how many nearest genes the table lists.
* :func:`panel_dashboard` ``projection`` -- ``umap`` / ``pca`` / ``tsne``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import GeneAnnotation, GeneEmbedding
from .embeddings import embedding_panel, embedding_scatter

__all__ = ["panel_dashboard", "panel_neighbours", "MAX_GRID_CELLS"]

#: Refuse to draw a grid bigger than this; subset ``methods``/``datasets`` instead.
MAX_GRID_CELLS = 40

#: Nearest genes listed per measured panel gene. MODIFY HERE.
NEIGHBOURS_PER_GENE = 8


def _split_keys(embeddings: Mapping[str, GeneEmbedding]
                ) -> tuple[Dict[str, Dict[str, str]], list[str]]:
    """``{method: {dataset: key}}`` for separate embedders, plus joint keys."""
    grid: Dict[str, Dict[str, str]] = {}
    joint: list[str] = []
    for key in embeddings:
        if "::" in key:
            method, dataset = key.split("::", 1)
            grid.setdefault(method, {})[dataset] = key
        else:
            joint.append(key)
    return grid, sorted(joint)


def panel_neighbours(
    embeddings: Mapping[str, GeneEmbedding],
    panel: Sequence[str],
    *,
    k: int = NEIGHBOURS_PER_GENE,
    annotation: Optional[GeneAnnotation] = None,
) -> pd.DataFrame:
    """For each embedding and each measured gene, its ``k`` nearest genes.

    Distances are cosine, computed in the **full d-dimensional space** -- not
    from the 2D projection, whose distances are not metric. Genes with an
    all-NaN row are excluded. Returns a long frame with one row per
    ``(embedding, panel gene, rank)``.

    This is the actionable output of the dashboard: a candidate list of
    unmeasured genes that behave like the ones you did measure.
    """
    from ..normalize import gene_key

    rows: list[Dict[str, Any]] = []
    want = {gene_key(g) for g in panel}
    ann = annotation.table if annotation is not None else None
    for name, emb in embeddings.items():
        M = emb.matrix
        keep = M.dropna(how="all")
        if keep.empty:
            continue
        X = keep.to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.where(norm > 0, norm, 1.0)
        index = list(keep.index)
        pos = {gene_key(g): i for i, g in enumerate(index)}
        for g in panel:
            i = pos.get(gene_key(g))
            if i is None:
                continue
            sim = Xn @ Xn[i]
            sim[i] = -np.inf
            order = np.argsort(-sim)[:k]
            for rank, j in enumerate(order, 1):
                row = {"embedding": name, "panel_gene": str(g), "rank": rank,
                       "neighbour": str(index[j]),
                       "cosine": round(float(sim[j]), 4)}
                if ann is not None and index[j] in ann.index:
                    row["neighbour_pathway"] = str(ann.loc[index[j], "top_pathway"])
                    for flag in ("is_tf", "is_ligand", "is_receptor"):
                        if flag in ann.columns:
                            row[flag] = bool(ann.loc[index[j], flag])
                rows.append(row)
    return pd.DataFrame(rows)


def panel_dashboard(
    embeddings: Mapping[str, GeneEmbedding],
    annotation: Optional[GeneAnnotation] = None,
    *,
    panel: Sequence[str],
    outdir: str | Path,
    genes: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    projection: str = "umap",
    color_by: str = "top_pathway",
    label_genes: Optional[Sequence[str]] = None,
    top_n: int = 12,
    scores: Optional[pd.DataFrame] = None,
    seed: int = 0,
    dpi: int = 170,
    verbose: bool = True,
) -> Dict[str, str]:
    """Render the dashboard. Returns ``{label: path}``.

    Parameters
    ----------
    embeddings
        ``{name: GeneEmbedding}`` as returned by ``run_embedders`` /
        ``align_references``. Keys ``<method>::<dataset>`` form the grid; bare
        ``<method>`` keys are treated as integrated embeddings.
    panel
        Your measured genes. Drawn as large labelled stars in every panel.
    genes
        Restrict every embedding to this gene list first, so the grid is
        computed over one common set.
    methods, datasets
        Subset the grid. Without them everything present is drawn, subject to
        :data:`MAX_GRID_CELLS`.
    scores
        The long-format score frame; included in the HTML index if given.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs: Dict[str, str] = {}

    embs = dict(embeddings)
    if genes is not None:
        embs = {k: GeneEmbedding(matrix=v.matrix.reindex(list(genes)),
                                 method=v.method, category=v.category,
                                 provenance=v.provenance)
                for k, v in embs.items()}

    grid, joint = _split_keys(embs)
    if methods is not None:
        grid = {m: v for m, v in grid.items() if m in set(methods)}
        joint = [j for j in joint if j in set(methods)]
    if datasets is not None:
        keep = set(datasets)
        grid = {m: {d: k for d, k in v.items() if d in keep} for m, v in grid.items()}
        grid = {m: v for m, v in grid.items() if v}

    n_cells = sum(len(v) for v in grid.values())
    if n_cells > MAX_GRID_CELLS:
        raise ValueError(
            f"{n_cells} grid cells exceeds MAX_GRID_CELLS={MAX_GRID_CELLS}; "
            f"pass methods=/datasets= to subset, or raise the limit knowingly")

    quality_frames: list[pd.DataFrame] = []

    # ---- one figure per method: that method across every dataset ----------
    for method, per_ds in sorted(grid.items()):
        keys = [per_ds[d] for d in sorted(per_ds)]
        path = out / f"grid_{method}.png"
        if verbose:
            print(f"  [dashboard] {method}: {len(keys)} datasets")
        q = embedding_panel(
            {k.split('::', 1)[1]: embs[k] for k in keys}, annotation, str(path),
            color_by=color_by, method=projection, top_n=top_n,
            highlight=list(panel), highlight_size=240, label_genes=label_genes,
            ncols=min(3, len(keys)), seed=seed, dpi=dpi, sort_by_quality=False,
            title=f"{method} — the measured panel in each reference dataset")
        if isinstance(q, pd.DataFrame) and not q.empty:
            q = q.copy(); q["method"] = method; q["kind"] = "separate"
            quality_frames.append(q)
        figs[f"grid_{method}"] = str(path)

    # ---- the integrated embeddings, which have no per-dataset version ------
    if joint:
        path = out / "grid_integrated.png"
        if verbose:
            print(f"  [dashboard] integrated: {len(joint)} methods")
        q = embedding_panel(
            {k: embs[k] for k in joint}, annotation, str(path),
            color_by=color_by, method=projection, top_n=top_n,
            highlight=list(panel), highlight_size=240, label_genes=label_genes,
            ncols=min(3, len(joint)), seed=seed, dpi=dpi, sort_by_quality=False,
            title="integrated embeddings — one map built from all references at once")
        if isinstance(q, pd.DataFrame) and not q.empty:
            q = q.copy(); q["method"] = "integrated"; q["kind"] = "joint"
            quality_frames.append(q)
        figs["grid_integrated"] = str(path)

    # ---- one large hero panel: the best-covered single embedding ----------
    hero_key = joint[0] if joint else (sorted(embs)[0] if embs else None)
    if hero_key:
        path = out / "panel_in_context.png"
        embedding_scatter(
            embs[hero_key], annotation, color_by=color_by, method=projection,
            highlight=list(panel), highlight_size=340, label_genes=label_genes,
            out_path=str(path), seed=seed, point_size=11, figsize=(13.0, 6.0),
            title=f"{hero_key}: the measured panel inside the gene universe")
        figs["panel_in_context"] = str(path)

    # ---- the actionable table ---------------------------------------------
    nb = panel_neighbours(embs, panel, annotation=annotation)
    if not nb.empty:
        nb_path = out / "panel_neighbours.csv"
        nb.to_csv(nb_path, index=False)
        figs["panel_neighbours"] = str(nb_path)

    quality = (pd.concat(quality_frames, ignore_index=True)
               if quality_frames else pd.DataFrame())
    if not quality.empty:
        qp = out / "embedding_quality.csv"
        quality.to_csv(qp, index=False)
        figs["embedding_quality"] = str(qp)

    # ---- the index ---------------------------------------------------------
    html_path = out / "dashboard.html"
    html_path.write_text(_render_html(
        figs, panel=list(panel), embeddings=embs, scores=scores,
        quality=quality, neighbours=nb, projection=projection))
    figs["dashboard_html"] = str(html_path)
    if verbose:
        print(f"  [dashboard] -> {html_path}")
    return figs


def _fmt_table(df: pd.DataFrame, *, floatfmt: str = "{:.4f}",
               index: bool = True) -> str:
    if df is None or df.empty:
        return "<p class='muted'>not available</p>"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return d.to_html(index=index, border=0, classes="tbl", escape=False,
                     na_rep="n/a")


def _render_html(figs: Mapping[str, str], *, panel: Sequence[str],
                 embeddings: Mapping[str, GeneEmbedding],
                 scores: Optional[pd.DataFrame],
                 quality: pd.DataFrame, neighbours: pd.DataFrame,
                 projection: str) -> str:
    """Self-contained index page; images are referenced by relative filename."""
    def img(label: str, title: str, note: str) -> str:
        if label not in figs:
            return ""
        name = Path(figs[label]).name
        return (f"<figure><figcaption><b>{html.escape(title)}</b><br>"
                f"<span class='muted'>{note}</span></figcaption>"
                f"<img src='{name}' alt='{html.escape(title)}'></figure>")

    score_html = ""
    if scores is not None and not scores.empty and {"dataset", "score", "value"} <= set(scores.columns):
        pivot = scores.pivot(index="dataset", columns="score", values="value")
        score_html = _fmt_table(pivot)

    grids = "".join(
        img(k, k.replace("grid_", "").replace("_", " ") +
            (" — integrated" if k.endswith("integrated") else " — per reference dataset"),
            "one shared colour mapping across the figure; stars are the measured panel; "
            "titles carry silhouette and kNN label agreement with permutation p-values")
        for k in sorted(figs) if k.startswith("grid_"))

    nb_html = ""
    if neighbours is not None and not neighbours.empty:
        top = neighbours[neighbours["rank"] <= 5]
        cols = [c for c in ["embedding", "panel_gene", "rank", "neighbour",
                            "cosine", "neighbour_pathway"] if c in top.columns]
        nb_html = _fmt_table(top[cols].reset_index(drop=True), index=False)

    qual_html = ""
    if quality is not None and not quality.empty:
        cols = [c for c in ["embedding", "method", "kind", "silhouette",
                            "silhouette_p", "knn_agreement", "knn_p",
                            "n_genes_scored"] if c in quality.columns]
        qual_html = _fmt_table(quality[cols].reset_index(drop=True), index=False)

    n_genes = max((e.matrix.shape[0] for e in embeddings.values()), default=0)
    dims = sorted({e.matrix.shape[1] for e in embeddings.values()})

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gene panel embedding dashboard</title>
<style>
 body {{ font: 15px/1.55 -apple-system, "Helvetica Neue", Arial, sans-serif;
        color:#1a1a1a; margin:0 auto; max-width:1180px; padding:32px 24px 80px; }}
 h1 {{ font-size:26px; margin:0 0 4px; }}
 h2 {{ font-size:19px; margin:38px 0 10px; padding-top:14px;
       border-top:1px solid #e3e3e3; }}
 .muted {{ color:#6b6b6b; font-size:13px; }}
 .kicker {{ color:#b2182b; font-weight:700; font-size:11px; letter-spacing:.08em; }}
 figure {{ margin:0 0 26px; }}
 figure img {{ width:100%; border:1px solid #e3e3e3; border-radius:3px; }}
 figcaption {{ margin-bottom:7px; }}
 table.tbl {{ border-collapse:collapse; font-size:12.5px; margin:8px 0 4px; }}
 table.tbl th {{ background:#1a1a1a; color:#fff; text-align:left;
                 padding:5px 9px; font-weight:600; }}
 table.tbl td {{ padding:4px 9px; border-bottom:1px solid #eee; }}
 table.tbl tr:nth-child(even) td {{ background:#fafafa; }}
 code {{ background:#f2f2f2; padding:1px 5px; border-radius:3px; font-size:13px; }}
 .warn {{ background:#fdf3f4; border-left:3px solid #b2182b;
          padding:10px 14px; margin:14px 0; font-size:13.5px; }}
</style></head><body>
<p class="kicker">GENE PANEL PREPROCESSING</p>
<h1>Where the measured panel sits in every reference</h1>
<p class="muted">Measured panel: <code>{html.escape(', '.join(map(str, panel)))}</code>
 &nbsp;·&nbsp; {len(embeddings)} embeddings &nbsp;·&nbsp; {n_genes} genes &nbsp;·&nbsp;
 d = {dims} &nbsp;·&nbsp; projection: {html.escape(projection)}</p>

<div class="warn"><b>How to read a projection.</b> {html.escape(projection.upper())}
distances are not metric and cluster sizes carry no meaning. Use the silhouette and
kNN numbers in each panel title — both come with a label-permutation null, so a panel
that looks structured but scores at chance says so. The neighbour table below is
computed in the full <i>d</i>-dimensional space, never from these 2D coordinates.</div>

<h2>1. How well does each reference match the panel?</h2>
<p class="muted">Raw score values. Each score has a direction — see
<code>score_heatmap.png</code> (red = agrees most) and <code>score_heatmap_raw.png</code>
(red = numerically large). Read correlation and OT scores across platforms; read
<code>avg_js</code> only within one.</p>
{score_html or "<p class='muted'>no score table supplied</p>"}

<h2>2. The panel across references and methods</h2>
{grids}

<h2>3. One embedding, full size</h2>
{img("panel_in_context", "The measured panel inside the gene universe",
     "large stars are measured genes, coloured by their own pathway annotation")}

<h2>4. Does the embedding recover pathway structure?</h2>
<p class="muted">Silhouette and kNN label agreement per panel, each against a
label-permutation null. A p near 1 means indistinguishable from chance.</p>
{qual_html or "<p class='muted'>not computed</p>"}

<h2>5. What is near the measured genes</h2>
<p class="muted">Top 5 nearest genes per measured gene, cosine distance in the full
embedding space. This is the actionable output: candidate unmeasured genes that behave
like the ones you measured. Full table in <code>panel_neighbours.csv</code>.</p>
{nb_html or "<p class='muted'>not computed</p>"}
</body></html>
"""
