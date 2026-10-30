"""The candidate-dataset catalogue (item 1 of the brief).

Each entry declares *what the dataset is* (for descriptor matching) and *how to
load it* (a loader name plus kwargs resolved by ``panelprep.io.loaders``).
Entries carry the brief's four-way category tag.

MODIFY HERE
-----------
Append a dict to ``CATALOG``. The only required keys are ``name``, ``category``
and ``loader``; everything else is metadata used for descriptor matching and
for the provenance columns of the report. Nothing else in the package needs to
change for a new dataset to be scored and embedded.

Categories
----------
spatial
    Expression measured with tissue position retained.
single_cell
    Dissociated cells / nuclei; no tissue coordinates. UMAP is not a coordinate.
non_transcriptomic
    Not a cell x gene matrix at all: protein language models, PPI graphs.
    These carry no expression, so expression scores report "not applicable".
causal
    Experimental perturbation (knockout / chimera) where a causal contrast
    exists between conditions, or a curated signed regulatory network.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

#: Local relative paths assume ``Config.data_dir`` (default ``data/``).
CATALOG: List[Dict[str, Any]] = [
    # ---------------- spatial: Xie et al. 2025 digital embryo (GSE278603) ----
    dict(
        name="seu3d_E7.5_rep2", category="spatial", loader="seu3d",
        loader_kwargs=dict(filename="GSM9046244_Embryo_E7.5_stereo_rep2.h5ad"),
        accession="GSE278603", stage="E7.5", organism="mus_musculus",
        modality="Stereo-seq spatial transcriptomics, single-cell assignment, 3D reconstruction",
        tissue="whole mouse embryo",
        obs_unit="cell",
        description=("Spatial transcriptomics of a whole mouse embryo at embryonic day 7.5 during "
                     "gastrulation and early organogenesis; germ layer and cell type annotation with "
                     "tissue coordinates for every cell."),
        doi="10.1016/j.cell.2025.05.035",
    ),
    dict(
        name="seu3d_E7.75_rep2", category="spatial", loader="seu3d",
        loader_kwargs=dict(filename="GSM9046246_Embryo_E7.75_stereo_rep2.h5ad"),
        accession="GSE278603", stage="E7.75", organism="mus_musculus",
        modality="Stereo-seq spatial transcriptomics, single-cell assignment, 3D reconstruction",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Spatial transcriptomics of a whole mouse embryo at embryonic day 7.75, early "
                     "organogenesis, germ layer patterning with tissue coordinates."),
        doi="10.1016/j.cell.2025.05.035",
    ),
    dict(
        name="seu3d_E8.0_rep1", category="spatial", loader="seu3d",
        loader_kwargs=dict(filename="GSM9046247_Embryo_E8.0_stereo_rep1.h5ad"),
        accession="GSE278603", stage="E8.0", organism="mus_musculus",
        modality="Stereo-seq spatial transcriptomics, single-cell assignment, 3D reconstruction",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Spatial transcriptomics of a whole mouse embryo at embryonic day 8.0, somite "
                     "stage organogenesis, with tissue coordinates."),
        doi="10.1016/j.cell.2025.05.035",
    ),
    dict(
        name="seu3d_E8.0_rep2", category="spatial", loader="seu3d",
        loader_kwargs=dict(filename="GSM9046248_Embryo_E8.0_stereo_rep2.h5ad"),
        accession="GSE278603", stage="E8.0", organism="mus_musculus",
        modality="Stereo-seq spatial transcriptomics, single-cell assignment, 3D reconstruction",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Second embryo replicate of spatial transcriptomics at embryonic day 8.0 with "
                     "tissue coordinates."),
        doi="10.1016/j.cell.2025.05.035",
    ),
    # ---------------- single cell: Mittnenzweig et al. 2021 (GSE169210) ------
    # Two entries from the same study at different developmental ages. This is a
    # deliberate positive/negative control for the real-data pilot: with an E7.5
    # query, the stage-matched entry should agree better than the E6.5-6.8 one.
    dict(
        name="mittnenzweig_wt_E7.5", category="single_cell", loader="mittnenzweig",
        loader_kwargs=dict(groups=("wt_bl6", "wt_icr"), time_range=(7.4, 7.7)),
        accession="GSE169210", stage="E7.5", organism="mus_musculus",
        modality="MARS-seq single cell RNA-seq, dissociated cells, single embryo resolution",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Single-cell RNA-seq of mouse gastrulation and early organogenesis around "
                     "embryonic day 7.5, wild-type embryos, individual embryo identity retained, "
                     "germ layer and cell type structure. Dissociated cells with no tissue "
                     "coordinates."),
        doi="10.1016/j.cell.2021.04.004",
    ),
    dict(
        name="mittnenzweig_wt_E6.5", category="single_cell", loader="mittnenzweig",
        loader_kwargs=dict(groups=("wt_bl6", "wt_icr"), time_range=(6.5, 6.8)),
        accession="GSE169210", stage="E6.5", organism="mus_musculus",
        modality="MARS-seq single cell RNA-seq, dissociated cells, single embryo resolution",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Single-cell RNA-seq of early mouse gastrulation around embryonic day 6.5, "
                     "wild-type embryos, before germ layer diversification is complete. "
                     "Dissociated cells with no tissue coordinates."),
        doi="10.1016/j.cell.2021.04.004",
    ),
    # ---------------- causal: perturbation arm of the same study ------------
    dict(
        name="mittnenzweig_foxc_dko", category="causal", loader="mittnenzweig",
        loader_kwargs=dict(groups=("chimera_foxc12_dko", "tetraploid_foxc_dko"),
                           control_groups=("chimera_control", "tetraploid_control")),
        accession="GSE169210", stage="E6.5-E8.25", organism="mus_musculus",
        modality="MARS-seq single cell RNA-seq of Foxc1/Foxc2 double-knockout chimeras with "
                 "matched wild-type control cells; experimental loss-of-function perturbation",
        tissue="whole mouse embryo", obs_unit="cell",
        description=("Causal perturbation experiment: Foxc1 and Foxc2 double knockout cells in "
                     "chimeric mouse embryos with matched control cells, reading out the "
                     "downstream transcriptional consequence of removing a transcription factor "
                     "during gastrulation and early organogenesis."),
        doi="10.1016/j.cell.2021.04.004",
    ),
    # ---------------- non-transcriptomic embedding sources ------------------
    dict(
        name="esm2_protein", category="non_transcriptomic", loader="none",
        organism="mus_musculus", modality="protein language model (ESM-2) over UniProt sequences",
        description=("Amino acid sequence representation of each gene product from a protein "
                     "language model. No cell by gene matrix."),
        embedding_only=True,
    ),
    dict(
        name="string_ppi", category="non_transcriptomic", loader="none",
        organism="mus_musculus", modality="STRING protein-protein interaction network",
        description=("Protein-protein interaction and functional association network. No cell by "
                     "gene matrix."),
        embedding_only=True,
    ),
    dict(
        name="kegg_regulatory", category="causal", loader="none",
        organism="mus_musculus", modality="KEGG KGML signed directed gene regulatory relations",
        description=("Curated signed, directed activation and inhibition relations between genes "
                     "from KEGG pathway maps, used as a causal-graph embedding source. Curated "
                     "literature relations, not an experimental causal estimate."),
        embedding_only=True,
    ),
]

_BY_NAME = {e["name"]: e for e in CATALOG}


def catalog_table() -> pd.DataFrame:
    cols = ["name", "category", "accession", "stage", "organism", "obs_unit", "modality", "doi"]
    df = pd.DataFrame(CATALOG)
    for c in cols:
        if c not in df:
            df[c] = None
    return df[cols].set_index("name")


def resolve_candidates(names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Catalogue entries for ``names`` (``None`` -> the whole catalogue)."""
    if names is None:
        return list(CATALOG)
    missing = [n for n in names if n not in _BY_NAME]
    if missing:
        raise KeyError(f"unknown catalogue entries {missing}; known: {sorted(_BY_NAME)}")
    return [_BY_NAME[n] for n in names]


def entry(name: str) -> Dict[str, Any]:
    return _BY_NAME[name]
