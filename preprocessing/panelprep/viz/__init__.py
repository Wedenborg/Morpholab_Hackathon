"""Visualisation: the agreement heatmap and the gene-embedding projections.

Two rules hold everywhere in this package:

1. **A picture never carries a claim on its own.** Every renderer returns the
   numbers behind it (``score_heatmap`` -> the raw/agreement/status matrices and
   the ranking; ``embedding_scatter``/``embedding_panel`` -> silhouette and kNN
   label agreement with label-permutation nulls) and prints them in the figure.
2. **Nothing is hand-listed.** The heatmap's columns, their directions and
   their descriptions come from the long-format frame returned by
   ``panelprep.scores.run_scores`` and from ``panelprep.scores.SCORES`` -- so a
   newly registered score appears with no edit to this package.

Two different questions about embeddings, two sets of functions:

*Within* an embedding -- "which embedding separates pathways?"
    ``embedding_scatter``, ``embedding_panel``, ``embedding_quality``
    (silhouette and kNN label agreement vs a label-permutation null).

*Between* embeddings -- "do two methods agree about gene geometry, and which
genes do they disagree about?"
    ``embedding_agreement_matrix`` (method x method kNN-Jaccard / CKA / Mantel
    with a gene-label-shuffled null), ``embedding_pair_scatter`` (one pair,
    Procrustes-aligned, arrows on the genes that move most),
    ``embedding_pair_grid`` (every pair in one figure).

Quick use::

    from panelprep.viz import score_heatmap, embedding_panel
    tab = score_heatmap(results, out_path="figures/score_heatmap.png")
    tab.ranking                      # mean agreement per dataset
    q = embedding_panel(embeddings, annotation, "figures/embeddings.png")
    q[["embedding", "silhouette", "silhouette_p", "knn_agreement", "knn_p"]]
    pa = embedding_agreement_matrix(embeddings, out_path="figures/pairs.png")
    pa.long                          # value, null_mean, p, n_shared per pair

MODIFY HERE: the knobs live in the two submodules
(``panelprep/viz/score_table.py``, ``panelprep/viz/embeddings.py``); this file
only re-exports.
"""
from .embeddings import (  # noqa: F401
    PAIR_METRICS,
    PROJECTION_DEFAULTS,
    ROLE_MARKERS,
    PairAgreement,
    PairScatter,
    ScatterResult,
    annotation_labels,
    category_palette,
    clean_matrix,
    embedding_agreement_matrix,
    embedding_pair_grid,
    embedding_pair_scatter,
    embedding_panel,
    embedding_quality,
    embedding_scatter,
    project,
    quality_caption,
)
from .score_table import (  # noqa: F401
    STATUS_COLOUR,
    STATUS_LABELS,
    AgreementTable,
    agreement_frame,
    score_heatmap,
    score_table_styled,
)

__all__ = [
    "score_heatmap", "score_table_styled", "agreement_frame", "AgreementTable",
    "STATUS_LABELS", "STATUS_COLOUR",
    "embedding_scatter", "embedding_panel", "embedding_quality", "project",
    "clean_matrix", "annotation_labels", "category_palette", "quality_caption",
    "ScatterResult", "PROJECTION_DEFAULTS", "ROLE_MARKERS",
    "embedding_agreement_matrix", "embedding_pair_scatter", "embedding_pair_grid",
    "PairAgreement", "PairScatter", "PAIR_METRICS",
]

