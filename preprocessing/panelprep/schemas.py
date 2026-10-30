"""Shared input/output schemas for the preprocessing team.

Every module in ``panelprep`` speaks only these types. They are deliberately
thin (dataclasses over pandas/numpy) so notebooks can build them by hand.

Conventions fixed here, and relied on everywhere else
----------------------------------------------------
Observation unit
    A *row* of any expression matrix is one measured observation: a cell for
    scRNA-seq and for segmented spatial data, a spot/bead/bin for Visium,
    Slide-seq or binned Stereo-seq. ``CandidateDataset.obs_unit`` records
    which; it is never inferred from the matrix shape. Scores that compare
    distributions across datasets with different observation units are
    comparing different things, and the report prints the units so that is
    visible rather than hidden.

Orientation
    Always observations x genes (cells as rows). Never transposed
    automatically. Gene *embeddings* are the transpose-flavoured object and
    are carried by ``GeneEmbedding`` (genes x d), not by these matrices.

Expression units
    ``units`` is a free-but-checked string. The pipeline contract is
    ``"log1p_cp10k"``: counts normalised to 10,000 per observation, then
    log1p. Loaders convert into that; anything else must say so, and
    distribution-comparing scores refuse to run across mismatched units
    unless explicitly allowed (see ``panelprep.scores``).

Gene identifiers
    Gene *symbols* are the join key, canonicalised by
    ``panelprep.normalize.gene_key`` (upper-case, Ensembl version suffix
    stripped). Symbols are not aliased or lifted over; a gene absent from a
    candidate is reported as missing, never silently imputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

Category = Literal["spatial", "single_cell", "non_transcriptomic", "causal"]
CATEGORIES: tuple[str, ...] = ("spatial", "single_cell", "non_transcriptomic", "causal")

ObsUnit = Literal["cell", "spot", "bin", "region", "nucleus"]

#: The normalisation the pipeline standardises on. See module docstring.
CANONICAL_UNITS = "log1p_cp10k"


@dataclass
class PanelQuery:
    """The measured experiment the preprocessing team is handed.

    Parameters
    ----------
    expression
        ``(N, S)`` observations x panel genes. Columns are gene symbols; the
        panel is small (the brief says ~4).
    coords
        ``(N, 2)`` tissue coordinates for the same rows, in the source's own
        units. Required: the brief states 2D locations are part of the input.
    descriptor
        Free text: the biology story of the experiment. Used by
        ``panelprep.match`` to shortlist candidate datasets.
    organism
        Used to pick the transcriptome-wide gene universe (item 0 of the brief).
    """

    expression: pd.DataFrame
    coords: np.ndarray
    descriptor: str
    organism: str = "mus_musculus"
    units: str = CANONICAL_UNITS
    obs_unit: ObsUnit = "cell"
    obs: Optional[pd.DataFrame] = None
    name: str = "query"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expression = pd.DataFrame(self.expression)
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] < 2:
            raise ValueError(f"coords must be (N, >=2); got {self.coords.shape}")
        if self.coords.shape[0] != self.expression.shape[0]:
            raise ValueError(
                f"coords rows ({self.coords.shape[0]}) != expression rows "
                f"({self.expression.shape[0]}); rows must be the same observations"
            )
        if self.expression.columns.duplicated().any():
            raise ValueError("duplicate gene columns in PanelQuery.expression")
        if self.obs is not None and len(self.obs) != self.expression.shape[0]:
            raise ValueError("obs rows must match expression rows")

    @property
    def panel(self) -> list[str]:
        """The measured gene subset, in column order."""
        return list(self.expression.columns)

    @property
    def n_obs(self) -> int:
        return int(self.expression.shape[0])


@dataclass
class CandidateDataset:
    """A reference dataset that may match the descriptor.

    ``expression`` is optional on purpose: ``non_transcriptomic`` and ``causal``
    candidates (a PPI graph, a curated signed regulatory network, a protein
    language model) carry no cell x gene matrix. Scores declare which
    categories they apply to, and the report marks the rest "not applicable"
    rather than producing a number that does not mean anything.
    """

    name: str
    category: Category
    expression: Optional[pd.DataFrame] = None
    coords: Optional[np.ndarray] = None
    obs: Optional[pd.DataFrame] = None
    units: str = CANONICAL_UNITS
    obs_unit: ObsUnit = "cell"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}; got {self.category!r}")
        if self.expression is not None:
            self.expression = pd.DataFrame(self.expression)
            if self.expression.columns.duplicated().any():
                raise ValueError(f"{self.name}: duplicate gene columns")
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.expression is not None and self.coords.shape[0] != self.expression.shape[0]:
                raise ValueError(f"{self.name}: coords rows != expression rows")

    @property
    def has_expression(self) -> bool:
        return self.expression is not None

    @property
    def has_coords(self) -> bool:
        return self.coords is not None

    @property
    def genes(self) -> list[str]:
        return [] if self.expression is None else list(self.expression.columns)

    @property
    def n_obs(self) -> int:
        return 0 if self.expression is None else int(self.expression.shape[0])

    def describe(self) -> str:
        """One-line text used for descriptor matching."""
        bits = [self.name, self.category, str(self.meta.get("modality", "")),
                str(self.meta.get("stage", "")), str(self.meta.get("tissue", "")),
                str(self.meta.get("organism", "")), str(self.meta.get("description", ""))]
        return " ".join(b for b in bits if b and b != "nan")


@dataclass
class ScoreResult:
    """One (score, dataset) cell of the comparison table.

    ``orientation`` is carried on the result, not assumed by the renderer: KL
    and Frobenius are divergences (lower = more agreement) while ARI is an
    agreement (higher = more agreement). ``panelprep.viz`` uses it to map every
    score onto a common "agreement" direction before colouring.
    """

    score_name: str
    dataset_name: str
    value: Optional[float]
    orientation: Literal["higher_is_better", "lower_is_better"]
    status: str = "ok"           # ok | insufficient_genes | not_applicable | units_mismatch | error
    n_genes_used: int = 0
    n_genes_requested: int = 0
    genes_missing: tuple = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "score": self.score_name,
            "value": self.value,
            "orientation": self.orientation,
            "status": self.status,
            "n_genes_used": self.n_genes_used,
            "n_genes_requested": self.n_genes_requested,
            "genes_missing": ",".join(self.genes_missing),
        }


@dataclass
class GeneEmbedding:
    """``(G, d)`` gene embedding: the deliverable of items 0, 2 and 3.

    Index is gene symbols (canonicalised keys are available via
    ``panelprep.normalize.gene_key``). ``category`` is the brief's four-way tag
    so that downstream code can ask for "all spatial gene embeddings".
    """

    matrix: pd.DataFrame
    method: str
    category: Category
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = pd.DataFrame(self.matrix)
        if self.matrix.index.duplicated().any():
            raise ValueError(f"{self.method}: duplicate genes in embedding index")
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}; got {self.category!r}")

    @property
    def genes(self) -> list[str]:
        return list(self.matrix.index)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def to_parquet(self, path) -> None:
        self.matrix.to_parquet(path)

    def __repr__(self) -> str:
        return (f"GeneEmbedding(method={self.method!r}, category={self.category!r}, "
                f"shape={self.matrix.shape})")


@dataclass
class GeneAnnotation:
    """Item 4: per-gene pathway membership and functional role flags.

    ``table`` is indexed by gene symbol with at least the columns
    ``is_ligand``, ``is_receptor``, ``is_tf`` (boolean) and ``pathways``
    (semicolon-joined pathway names), plus ``top_pathway`` used for colouring.
    """

    table: pd.DataFrame
    sources: Dict[str, Any] = field(default_factory=dict)

    REQUIRED = ("is_ligand", "is_receptor", "is_tf", "pathways", "top_pathway")

    def __post_init__(self) -> None:
        self.table = pd.DataFrame(self.table)
        missing = [c for c in self.REQUIRED if c not in self.table.columns]
        if missing:
            raise ValueError(f"GeneAnnotation.table missing columns: {missing}")
