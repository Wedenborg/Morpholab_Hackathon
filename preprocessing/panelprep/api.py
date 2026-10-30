"""The two calls most users need.

You have **your data** (a measured gene panel with cell coordinates) and **a set of
reference datasets** (single-cell, spatial, perturbation). Two questions follow:

1. *Which references match my data?* → :func:`align_references`
2. *Where does my panel sit inside each of them?* → :func:`panel_dashboard`

Both are thin wrappers over :func:`panelprep.run.run_analysis`; everything they
do is available piecewise if you need finer control. See ``USER_MANUAL.md``.

>>> from panelprep.api import align_references
>>> result = align_references(my_data, references=["seu3d_E7.5_rep2",
...                                                "mittnenzweig_wt_E7.5"])
>>> result.table            # dataset x score
>>> result.ranking         # mean agreement, best first
>>> print(result.summary())

MODIFY HERE
-----------
* ``DEFAULT_SCORES`` / ``DEFAULT_METHODS`` -- what runs when you pass nothing.
* :class:`Alignment` -- add a field if you need something else returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import Config
from .schemas import CandidateDataset, GeneAnnotation, GeneEmbedding, PanelQuery

__all__ = ["align_references", "panel_dashboard", "Alignment",
           "DEFAULT_SCORES", "DEFAULT_METHODS"]

#: Scores run when ``scores=None``. Spatial ones are skipped automatically for
#: references with no coordinates. MODIFY HERE.
DEFAULT_SCORES: tuple[str, ...] = (
    "avg_js",                      # per-gene distribution divergence (within one platform)
    "corr_frobenius",              # panel co-expression structure, scale-free
    "corr_cosine",                 # the pattern of co-expression, ignoring magnitude
    "ot_cluster_ari",              # optimal transport between cell clusters
    "spatial_autocorr_frobenius",  # section-wise Moran's I agreement (spatial only)
)

#: Embedders run when ``methods=None``: one cheap per-dataset baseline plus one
#: integrated model. Add ``ae_separate``/``vae_separate``/``joint_ae``/
#: ``contrastive_ae`` for the full panel. MODIFY HERE.
DEFAULT_METHODS: tuple[str, ...] = ("pca_baseline", "dcca")

References = Union[Sequence[str], Mapping[str, CandidateDataset]]


@dataclass
class Alignment:
    """What :func:`align_references` returns.

    Attributes
    ----------
    table
        ``datasets x scores`` of raw values -- the primary answer.
    ranking
        One row per reference, ordered best first, with the rank-normalised
        mean agreement across scores. Use this to pick a reference; use
        ``table`` to see *why*.
    best
        The top-ranked reference name.
    scores
        The long-format frame: one row per ``(dataset, score)`` with status,
        gene coverage and the score's direction. Everything else is derived
        from this.
    embeddings
        ``{name: GeneEmbedding}``, each ``|G| x d`` indexed by gene symbol.
    annotation
        Pathway and ligand/receptor/TF flags for the gene universe.
    figures
        ``{label: path}`` for everything written.
    blocked
        Anything requested that could not run, with the reason. Non-empty is
        normal -- an expensive cache-only embedding reports itself here rather
        than silently retraining.
    """

    table: pd.DataFrame
    ranking: Optional[pd.DataFrame]
    scores: pd.DataFrame
    embeddings: Dict[str, GeneEmbedding] = field(default_factory=dict)
    annotation: Optional[GeneAnnotation] = None
    figures: Dict[str, str] = field(default_factory=dict)
    genes: list = field(default_factory=list)
    blocked: Dict[str, str] = field(default_factory=dict)
    outdir: str = ""
    panel: list = field(default_factory=list)

    @property
    def best(self) -> Optional[str]:
        if self.ranking is None or self.ranking.empty:
            return None
        return str(self.ranking.index[0] if self.ranking.index.name
                   else self.ranking.iloc[0, 0])

    def summary(self) -> str:
        """A few lines you can paste into a notebook or a message."""
        lines = [f"panel: {', '.join(map(str, self.panel))}",
                 f"references scored: {self.table.shape[0]}",
                 f"scores: {', '.join(map(str, self.table.columns))}"]
        if self.best:
            lines.append(f"best match: {self.best}")
        ok = self.scores[self.scores["status"] == "ok"] if "status" in self.scores else self.scores
        if not ok.empty and "n_genes_used" in ok:
            lines.append(f"panel genes used per score: "
                         f"{int(ok['n_genes_used'].min())}-{int(ok['n_genes_used'].max())} "
                         f"of {int(ok['n_genes_requested'].max())}")
        if self.embeddings:
            dims = sorted({e.matrix.shape[1] for e in self.embeddings.values()})
            lines.append(f"embeddings: {len(self.embeddings)} x "
                         f"({len(self.genes)} genes, d={dims})")
        if self.blocked:
            lines.append(f"blocked: {', '.join(self.blocked)}")
        if self.figures:
            lines.append(f"figures: {len(self.figures)} written to {self.outdir}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Alignment(references={self.table.shape[0]}, "
                f"scores={self.table.shape[1]}, best={self.best!r}, "
                f"embeddings={len(self.embeddings)})")


def align_references(
    data: PanelQuery,
    references: Optional[References] = None,
    *,
    scores: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    outdir: str = "outputs/alignment",
    store: Optional[Any] = None,
    dim: int = 32,
    max_genes: Optional[int] = 2000,
    gene_universe: str = "intersection",
    max_obs: int = 1500,
    annotate: bool = True,
    embeddings: bool = True,
    dashboard: bool = False,
    projection: str = "umap",
    seed: int = 0,
    config: Optional[Config] = None,
    verbose: bool = True,
) -> Alignment:
    """Score a set of reference datasets against your measured panel.

    Parameters
    ----------
    data
        Your measurement, as a :class:`~panelprep.schemas.PanelQuery`: the
        ``N x S`` panel matrix, the ``N x 2`` coordinates and a free-text
        descriptor. Build one with
        :func:`panelprep.io.loaders.query_from_seu3d`,
        :func:`panelprep.integrations.morpholab.query_from_morpholab`, or the
        constructor directly.
    references
        Either catalogue names (``["seu3d_E7.5_rep2", ...]`` -- see
        :func:`panelprep.io.catalog.catalog_table`) or an already-loaded
        ``{name: CandidateDataset}`` mapping. ``None`` uses the configured
        candidate list.
    scores
        Score names from ``panelprep.scores.SCORES``; defaults to
        :data:`DEFAULT_SCORES`. A score that does not apply to a reference's
        category, or needs coordinates it does not have, is reported with a
        status rather than silently dropped.
    methods
        Embedder names from ``panelprep.embed.EMBEDDERS``; defaults to
        :data:`DEFAULT_METHODS`. Set ``embeddings=False`` to score only.
    dashboard
        Also build the panel-embedding dashboard (adds a few minutes).

    Returns
    -------
    Alignment
        ``.table`` is the answer; ``.ranking`` orders the references;
        ``.summary()`` prints the lot.

    Notes
    -----
    **Reading the scores.** Each carries a direction, and the heatmap is written
    in both colourings (``score_heatmap.png`` = red means *agrees most*,
    ``score_heatmap_raw.png`` = red means *numerically large*), because a low
    divergence and a high agreement are the same fact with opposite signs.

    **Across platforms.** Detection rates differ several-fold between assays
    (e.g. Stereo-seq 8-22% vs MARS-seq 36-88% on a 4-gene panel), which shifts
    every gene's marginal. Read ``corr_cosine``, ``corr_frobenius`` and
    ``ot_cluster_ari`` across platforms; read ``avg_js``/``avg_kl`` only within
    one. This is a property of the assays, not of the implementation.

    **References with no expression matrix** (a protein language model, an
    interaction graph) are excluded from scoring entirely rather than listed as
    "not applicable" -- they were never candidate references.
    """
    from .embed.store import EmbeddingStore
    from .run import run_analysis

    names: Optional[list[str]] = None
    preloaded: Optional[Dict[str, CandidateDataset]] = None
    if references is not None:
        if isinstance(references, Mapping):
            preloaded = dict(references)
            names = list(preloaded)
        else:
            names = [str(r) for r in references]

    cfg = (config or Config()).replace(
        run_name=Path(outdir).name or "alignment",
        outdir=outdir,
        seed=seed,
        candidates=names,
        scores=list(scores if scores is not None else DEFAULT_SCORES),
        embedders=list(methods if methods is not None else DEFAULT_METHODS),
        embedding_dim=dim,
        max_genes=max_genes,
        gene_universe=gene_universe,
        max_obs_per_dataset=max_obs,
    )
    if not annotate:
        cfg = cfg.replace(annotation_sources=[])

    stages = ["score"]
    if embeddings:
        stages += ["embed"]
        if annotate:
            stages += ["annotate"]
        stages += ["figures"]
    stages += ["report"]

    result = run_analysis(
        cfg, query=data, store=store or EmbeddingStore(Path(outdir) / "store"),
        candidates=preloaded, stages=tuple(stages), projection=projection,
        verbose=verbose)

    alignment = Alignment(
        table=result.get("score_table", pd.DataFrame()),
        ranking=result.get("ranking"),
        scores=result.get("scores", pd.DataFrame()),
        embeddings=result.get("embeddings", {}),
        annotation=result.get("annotation"),
        figures=dict(result.get("figures", {})),
        genes=list(result.get("genes", [])),
        blocked=dict(result.get("blocked", {})),
        outdir=str(cfg.out),
        panel=list(data.panel),
    )

    if dashboard and alignment.embeddings:
        from .viz.dashboard import panel_dashboard as _dash
        paths = _dash(alignment.embeddings, alignment.annotation,
                      panel=alignment.panel, genes=alignment.genes,
                      outdir=Path(cfg.out) / "dashboard", projection=projection,
                      scores=alignment.scores, seed=seed, verbose=verbose)
        alignment.figures.update(paths)

    if verbose:
        print("\n" + alignment.summary())
    return alignment


def panel_dashboard(
    alignment: Alignment,
    *,
    outdir: Optional[str] = None,
    projection: str = "umap",
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    color_by: str = "top_pathway",
    label_genes: Optional[Sequence[str]] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Build the panel-embedding dashboard from an :class:`Alignment`.

    One page showing where your measured panel genes sit inside every reference
    dataset, under every embedding method -- the measured genes drawn as large
    labelled stars over the rest of the transcriptome, coloured by pathway.

    Returns ``{label: path}``; open the ``dashboard_html`` entry in a browser.
    """
    from .viz.dashboard import panel_dashboard as _dash

    return _dash(alignment.embeddings, alignment.annotation,
                 panel=alignment.panel, genes=alignment.genes,
                 outdir=Path(outdir or (Path(alignment.outdir) / "dashboard")),
                 projection=projection, methods=methods, datasets=datasets,
                 color_by=color_by, label_genes=label_genes,
                 scores=alignment.scores, seed=seed, verbose=verbose)
