"""panelprep -- preprocessing for spatial gene-panel planning.

Pipeline, in the order of the brief:

0. ``panelprep.embed``        gene embeddings for the whole transcriptome, tagged
                              spatial / single_cell / non_transcriptomic / causal
1. ``panelprep.io.catalog`` + ``panelprep.match`` + ``panelprep.scores``
                              find and score candidate datasets D against the
                              measured panel
2. ``panelprep.embed.expression`` / ``.joint``
                              separate and joint autoencoder gene embeddings
3. ``panelprep.embed.external``
                              sequence (ESM-2), PPI (STRING) and causal (KEGG)
4. ``panelprep.annotate``     pathway membership and ligand / receptor / TF roles

One configuration entry point: ``panelprep.config.Config``.
"""
from .config import Config  # noqa: F401
from .schemas import (  # noqa: F401
    CATEGORIES, CandidateDataset, GeneAnnotation, GeneEmbedding, PanelQuery, ScoreResult,
)
from .registry import Registry  # noqa: F401

__all__ = ["Config", "CandidateDataset", "GeneAnnotation", "GeneEmbedding", "PanelQuery",
           "ScoreResult", "Registry", "CATEGORIES"]
__version__ = "0.1.0"
