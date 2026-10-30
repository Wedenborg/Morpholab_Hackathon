"""The single configuration entry point.

Everything the pipeline does is driven by one ``Config`` object, built either
in Python or from ``configs/default.yaml``. There is no second place to look.

MODIFY HERE
-----------
* ``scores`` / ``embedders`` -- add a registered name to turn it on. The
  comparison report is generated from whatever ran, so adding a score here is
  the only step needed for it to appear in the table and the report.
* ``score_params`` / ``embedder_params`` -- per-method keyword overrides.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # --- run identity -----------------------------------------------------
    run_name: str = "synthetic"
    seed: int = 0
    outdir: str = "outputs/synthetic"

    # --- data -------------------------------------------------------------
    organism: str = "mus_musculus"
    data_dir: str = "data"
    #: Candidate datasets to consider; names resolved by panelprep.io.catalog.
    #: null/None means "everything in the catalogue".
    candidates: Optional[List[str]] = None
    #: Cap on observations sampled per dataset for scoring (keeps OT tractable).
    max_obs_per_dataset: int = 1500

    # --- descriptor matching ---------------------------------------------
    match_method: str = "tfidf"
    match_top_k: int = 0          # 0 = keep all candidates, just rank them

    # --- scoring ----------------------------------------------------------
    scores: List[str] = field(default_factory=lambda: [
        "avg_kl", "avg_js", "corr_frobenius", "ot_cluster_ari",
        "spatial_autocorr_frobenius",
    ])
    score_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allow_unit_mismatch: bool = False

    # --- gene embeddings --------------------------------------------------
    embedders: List[str] = field(default_factory=lambda: [
        "pca_baseline", "ae_separate", "vae_separate", "joint_ae", "dcca", "contrastive_ae",
    ])
    embedder_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    embedding_dim: int = 32
    #: Gene universe for embeddings. "union" = all genes seen in any expression
    #: candidate (the brief's "all genes in the transcriptome"); "intersection"
    #: = genes present in every expression candidate (required by joint methods).
    gene_universe: str = "union"
    max_genes: Optional[int] = None   # cap for cheap smoke runs

    # --- annotations ------------------------------------------------------
    annotation_sources: List[str] = field(default_factory=lambda: ["kegg", "roles"])
    offline: bool = False     # True -> no network; cached/bundled data only

    def __post_init__(self) -> None:
        if self.gene_universe not in ("union", "intersection"):
            raise ValueError("gene_universe must be 'union' or 'intersection'")

    # --- plumbing ---------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    @property
    def out(self) -> Path:
        p = Path(self.outdir)
        p.mkdir(parents=True, exist_ok=True)
        return p
