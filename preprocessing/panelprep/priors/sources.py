"""Multi-source genome-wide gene *priors*, after ITERPERT part 1.

Reference
---------
Huang, Y. et al. "Sequential Optimal Experimental Design of Perturbation
Screens Guided by Multi-modal Priors" (bioRxiv 2023.12.12.571389). This module
implements the paper's **prior feature maps** only. Their kernelisation and
fusion live in :mod:`panelprep.priors.kernels`; the sequential/active-learning
loop they wrap around those kernels is explicitly out of scope for this
package.

What a prior is here
--------------------
Observation unit
    **The gene is the observation.** Every matrix in this module is
    ``genes x features`` -- the transpose of the package's expression contract
    (``panelprep.schemas``: observations x genes). A prior is defined for the
    whole mouse protein-coding transcriptome, not for a measured panel; that is
    the entire point of a prior, and coverage below 100% is reported, never
    hidden.
Feature map
    ``phi(gene) -> R^f``. The paper keeps the feature map and the kernel
    separate on purpose: the kernel ``K = phi phi^T`` (or a cosine/RBF variant)
    is what enters the downstream model, and fusing kernels rather than
    concatenating features is what makes sources with wildly different
    dimensionality comparable.
Missing data
    A gene with no information in a source is an **all-NaN row**, never a zero
    row. Zero is a claim ("this gene's features are at the origin"); NaN is the
    absence of a claim. :func:`panelprep.priors.kernels.fuse_kernels` consumes
    the NaN pattern through its ``missing=`` policy.
Normalisation
    Per source, and documented per source. The co-expression prior is
    standardised **per gene** because that is what makes its linear kernel the
    gene-gene correlation matrix (the paper's own identification of this
    feature map with the co-expression matrix). Sequence/text/graph priors are
    not re-standardised; the kernel's ``kind='cosine'`` handles their scale.

Organism
--------
Mouse (*Mus musculus*, NCBI taxon 10090). Symbols are Title case (``Sox2``) and
are matched through :func:`panelprep.normalize.gene_key` (case-folded), so a
source that ships upper-case symbols still joins.

Status, not exceptions
----------------------
Every source returns a :class:`PriorFeatures` whose ``provenance['status']`` is
one of

``ok``
    features computed for at least one gene.
``ok_fallback_*``
    computed, but by a documented substitute path (named in the status).
``deferred_*``
    the source cannot exist for this organism / this project's data. The matrix
    is all-NaN and the reason is in ``provenance['reason']``. A deferred source
    is *not* silently replaced by a different quantity.
``blocked_*``
    a transient or environmental obstruction (network denied, no cache in
    offline mode, missing optional dependency).

MODIFY HERE -- adding a seventh prior source
--------------------------------------------
1. Write a function in this file with exactly the registry signature::

       @PRIORS.register(
           "my_source",
           modality="what it measures, one phrase",
           category="non_transcriptomic",      # one of schemas.CATEGORIES
           needs_network=True,
           description="one line for the fusion report",
       )
       def my_source(genes, *, dim, rng, cache_dir, offline, datasets=None,
                     **params) -> PriorFeatures: ...

2. Return ``PriorFeatures`` built with :func:`make_features` (it enforces the
   genes x f orientation, the NaN rows and the provenance keys).
3. Nothing else changes: ``build_priors`` runs whatever names it is given,
   ``to_gene_embedding`` puts the source into the existing embedding inventory,
   and ``fuse_report`` picks it up automatically, so the comparison table
   updates itself.

Other MODIFY HERE points are marked inline next to the thing to change.
"""
from __future__ import annotations

import gzip
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..normalize import gene_key
from ..registry import Registry
from ..schemas import CATEGORIES, CandidateDataset, GeneEmbedding

# Hugging Face's "xet" transfer backend opens its own connections and ignores a
# SOCKS proxy, so in a proxied sandbox a checkpoint download hangs indefinitely
# instead of failing. The classic HTTP path works. Set before any
# ``huggingface_hub`` import; ``setdefault`` so an explicit choice wins.
# MODIFY HERE: drop this line on a machine with direct internet access.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PRIORS = Registry("prior")

#: Default on-disk cache. Every source must work offline from this directory.
DEFAULT_CACHE = "data/external/priors"

MOUSE_TAXON = 10090

# MODIFY HERE: pinned release versions. Bump deliberately -- a STRING or
# UniProt release change moves every kernel built from it.
STRING_VERSION = "v12.0"
STRING_LINKS_URL = (
    f"https://stringdb-downloads.org/download/protein.links.{STRING_VERSION}/"
    f"{MOUSE_TAXON}.protein.links.{STRING_VERSION}.txt.gz"
)
STRING_INFO_URL = (
    f"https://stringdb-downloads.org/download/protein.info.{STRING_VERSION}/"
    f"{MOUSE_TAXON}.protein.info.{STRING_VERSION}.txt.gz"
)
NCBI_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz"
)
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
#: Mouse reference proteome, used as the fallback route when the REST stream
#: stalls (see ``_uniprot_mouse_sequences``).
UNIPROT_PROTEOME_ID = "UP000000589"
UNIPROT_FASTA_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/"
    f"reference_proteomes/Eukaryota/{UNIPROT_PROTEOME_ID}/"
    f"{UNIPROT_PROTEOME_ID}_{MOUSE_TAXON}.fasta.gz"
)
KEGG_REST = "https://rest.kegg.jp"


# ---------------------------------------------------------------------------
# the carrier type
# ---------------------------------------------------------------------------

@dataclass
class PriorFeatures:
    """One prior feature map: ``genes x f``, plus where it came from.

    Attributes
    ----------
    matrix
        ``(G, f)`` DataFrame. Index is the requested gene symbols **in the
        requested order** (so two priors align positionally); rows for genes the
        source does not cover are entirely NaN.
    name, modality, category
        Registry name, a human phrase for the report, and the brief's four-way
        tag (``schemas.CATEGORIES``).
    provenance
        At minimum ``status``, ``n_requested``, ``n_covered``, ``coverage``, and
        the source URL / accession / model checkpoint actually used, plus any
        threshold or truncation applied.
    """

    matrix: pd.DataFrame
    name: str
    modality: str
    category: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = pd.DataFrame(self.matrix)
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}; got {self.category!r}")
        if self.matrix.index.duplicated().any():
            raise ValueError(f"{self.name}: duplicate genes in prior index")

    # -- coverage ---------------------------------------------------------
    @property
    def covered(self) -> np.ndarray:
        """Boolean mask over rows: True where the source has features."""
        if self.matrix.shape[1] == 0:
            return np.zeros(self.matrix.shape[0], dtype=bool)
        return np.isfinite(self.matrix.to_numpy(dtype=float)).any(axis=1)

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())

    @property
    def genes(self) -> List[str]:
        return list(self.matrix.index)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def status(self) -> str:
        return str(self.provenance.get("status", "unknown"))

    @property
    def ok(self) -> bool:
        return self.status.startswith("ok")

    def __repr__(self) -> str:
        return (f"PriorFeatures(name={self.name!r}, status={self.status!r}, "
                f"shape={self.matrix.shape}, covered={self.n_covered})")


def make_features(
    genes: Sequence[str],
    values: Optional[Dict[str, np.ndarray]] = None,
    *,
    name: str,
    modality: str,
    category: str,
    status: str,
    dim: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    **provenance,
) -> PriorFeatures:
    """Assemble a :class:`PriorFeatures` with the orientation/NaN contract enforced.

    Parameters
    ----------
    genes
        The requested universe, in order. The output index is exactly this.
    values
        ``{gene_symbol: feature_vector}`` for covered genes only. Keys are
        matched case-insensitively via ``normalize.gene_key``. ``None`` or
        ``{}`` produces an all-NaN matrix with ``dim`` columns (the deferred /
        blocked case).
    """
    genes = [str(g) for g in genes]
    vals = {gene_key(k): np.asarray(v, dtype=np.float32) for k, v in (values or {}).items()}
    if vals:
        f = int(next(iter(vals.values())).shape[0])
        bad = {k: v.shape for k, v in vals.items() if v.shape != (f,)}
        if bad:
            raise ValueError(f"{name}: ragged feature vectors: {list(bad)[:3]}")
    else:
        f = int(dim or 0)
    cols = list(feature_names) if feature_names is not None else [f"{name}_{i}" for i in range(f)]
    if len(cols) != f:
        raise ValueError(f"{name}: {len(cols)} feature names for {f} features")
    mat = np.full((len(genes), f), np.nan, dtype=np.float32)
    n_cov = 0
    for i, g in enumerate(genes):
        v = vals.get(gene_key(g))
        if v is not None:
            mat[i] = v
            n_cov += 1
    prov = {
        "status": status,
        "n_requested": len(genes),
        "n_covered": int(n_cov),
        "coverage": float(n_cov / len(genes)) if genes else 0.0,
        "n_features": f,
        **provenance,
    }
    return PriorFeatures(pd.DataFrame(mat, index=genes, columns=cols),
                         name=name, modality=modality, category=category, provenance=prov)


# ---------------------------------------------------------------------------
# cache / download plumbing
# ---------------------------------------------------------------------------

class PriorFetchError(RuntimeError):
    """Internal: a download failed. Sources convert this into a ``blocked_*`` status."""


def cache_path(cache_dir: str | os.PathLike, filename: str) -> Path:
    p = Path(cache_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_to_cache(
    url: str,
    dest: Path,
    *,
    offline: bool,
    params: Optional[dict] = None,
    timeout: int = 180,
    chunk: int = 1 << 20,
    max_seconds: Optional[float] = None,
) -> Path:
    """Download ``url`` to ``dest`` unless already cached. Never re-downloads.

    Missing-data behaviour: in ``offline`` mode a cache miss raises
    :class:`PriorFetchError`, which the calling source turns into
    ``status='blocked_offline_no_cache'`` -- it does not fabricate features.

    ``timeout`` is the per-read socket timeout; ``max_seconds`` is a wall-clock
    budget for the whole transfer. The second one matters because a server that
    trickles a few bytes keeps resetting the socket timeout: UniProt's
    ``/stream`` endpoint was observed to stall for >20 minutes mid-body on the
    mouse query, which a read timeout alone will never catch.

    MODIFY HERE: delete the cached file to force a refresh; there is no TTL on
    purpose, so a run is reproducible from the cache alone. Raise
    ``max_seconds`` for a slow link.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise PriorFetchError(f"offline=True and no cache at {dest}")
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    try:
        with requests.get(url, params=params, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for block in r.iter_content(chunk_size=chunk):
                    if block:
                        fh.write(block)
                    if max_seconds is not None and time.time() - t0 > max_seconds:
                        raise TimeoutError(
                            f"transfer exceeded max_seconds={max_seconds} "
                            f"({fh.tell()} bytes received)")
        os.replace(tmp, dest)
    except Exception as exc:  # noqa: BLE001 - converted to a status upstream
        tmp.unlink(missing_ok=True)
        raise PriorFetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
    return dest


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


# ---------------------------------------------------------------------------
# the gene universe (what a "genome-wide" prior is evaluated over)
# ---------------------------------------------------------------------------

def mouse_protein_coding_universe(
    *,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    max_genes: Optional[int] = None,
    seed: int = 0,
) -> List[str]:
    """The mouse protein-coding transcriptome as Title-case symbols (~22k).

    Observation unit: one gene. Source is the NCBI *Mus musculus* ``gene_info``
    table filtered to ``type_of_gene == 'protein-coding'``; symbols are taken as
    deposited (Title case) and de-duplicated.

    ``max_genes`` takes a *deterministic* subsample (sorted, seeded) for runtime
    budgeting -- a capped universe must be reported as capped, because coverage
    fractions are only comparable within one universe.

    MODIFY HERE: swap the filter to include lncRNA/pseudogenes, or replace the
    source with Ensembl/MGI; everything downstream only needs a list of symbols.
    """
    dest = cache_path(cache_dir, "Mus_musculus.gene_info.gz")
    fetch_to_cache(NCBI_GENE_INFO_URL, dest, offline=offline)
    syms: List[str] = []
    with _open_text(dest) as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
        i_sym, i_type = header.index("Symbol"), header.index("type_of_gene")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > i_type and f[i_type] == "protein-coding":
                syms.append(f[i_sym])
    out = sorted(dict.fromkeys(syms))
    if max_genes is not None and len(out) > max_genes:
        rng = np.random.default_rng(seed)
        out = sorted(np.asarray(out)[rng.choice(len(out), size=max_genes, replace=False)].tolist())
    return out


# ---------------------------------------------------------------------------
# small shared numerics
# ---------------------------------------------------------------------------

def _pca_rows(X: np.ndarray, dim: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """PCA over the *feature* axis of a dense, fully observed ``rows x f`` block."""
    from sklearn.decomposition import PCA

    k = int(min(dim, X.shape[0], X.shape[1]))
    if k < 1:
        return X, {"pca": False, "reason": "no components available"}
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X)
    return Z.astype(np.float32), {"pca": True, "n_components": k,
                                  "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}


def zscore_rows(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise each row (gene) to mean 0, sd 1. Returns ``(Z, usable_mask)``.

    Rows with zero variance cannot be standardised; they are returned as NaN and
    flagged ``False``, because a constant gene has no co-expression information
    and imputing zeros would make it look maximally average instead of unknown.
    """
    M = np.asarray(M)
    # Accumulate in float64 -- the linear kernel of the output is compared with
    # ``np.corrcoef`` to 1e-10 in the tests, which float32 cannot deliver -- and
    # return float64 only when the caller supplied it (the atlas path keeps
    # float32 to hold ~20k genes x cells in memory).
    W = M.astype(np.float64, copy=False)
    out_dtype = np.float64 if M.dtype == np.float64 else np.float32
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected
        mu = np.nanmean(W, axis=1, keepdims=True)
        sd = np.nanstd(W, axis=1, keepdims=True)
    usable = np.isfinite(sd[:, 0]) & (sd[:, 0] > 0)
    Z = np.full(W.shape, np.nan, dtype=np.float64)
    np.divide(W - mu, sd, out=Z, where=(sd > 0) & np.isfinite(sd))
    Z[~usable] = np.nan
    return Z.astype(out_dtype, copy=False), usable


# ===========================================================================
# 1. Additional Perturb-seq  (paper source 1)
# ===========================================================================

@PRIORS.register(
    "perturbation_pseudobulk",
    modality="pseudo-bulk expression change per perturbed gene (Perturb-seq)",
    category="causal",
    needs_network=False,
    description="control-centred pseudobulk delta per perturbed gene; needs a genome-scale screen",
)
def perturbation_pseudobulk(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    condition_col: str = "condition",
    control_label: str = "control",
    perturbation_col: Optional[str] = None,
    min_perturbations: int = 100,
    pca: bool = True,
    **_,
) -> PriorFeatures:
    """phi(x) = mean pseudobulk expression change of perturbation *x* vs control.

    The paper
    ---------
    "for each perturbation x, phi(x) is the mean of pseudo-bulk expression change
    from the non-targeting control cells".

    Formulation point that decides whether this source can exist
    ------------------------------------------------------------
    The observation unit of the *feature map* is a **perturbed gene**, and the
    feature axis is the measured transcriptome. A genome-wide prior therefore
    requires a screen in which (nearly) every gene has been perturbed: one
    pseudobulk delta vector per gene. A single knockout experiment yields **one**
    delta vector, which is a feature of that experiment, not a per-gene feature
    map -- turning it into a one-column prior would silently give all ~22,000
    genes the same value and would not be the paper's source.

    This project's only mouse perturbation data is the Foxc1/Foxc2 double
    knockout arm of GSE169210 (``mittnenzweig_foxc_dko``), i.e. a single
    perturbation, and a *double* knockout at that, so even the one delta cannot
    be attributed to an individual gene. The function therefore computes the
    control contrast (and reports its size, as evidence the computation is
    real), then returns an all-NaN matrix with
    ``status='deferred_insufficient_perturbations'``.

    What would populate it, unchanged
    ---------------------------------
    A genome-scale mouse Perturb-seq (the mouse analogue of Replogle et al.
    2022): a ``CandidateDataset`` whose ``obs`` carries ``condition_col`` with
    ``control_label`` for non-targeting cells and the perturbed gene symbol for
    the rest. With ``>= min_perturbations`` distinct perturbations the real
    branch below runs and the source reports ``status='ok'``. That branch is
    exercised by ``tests/test_priors.py::test_perturbation_real_branch_synthetic``
    on a synthetic screen.

    Normalisation
    -------------
    Deltas are differences of per-group means of the dataset's own
    ``log1p_cp10k`` values (so the delta is a log fold change relative to
    control), with no further scaling; PCA over the gene axis is optional.

    Missing data
    ------------
    Genes never perturbed are NaN rows. Perturbations with fewer than two cells
    are dropped and listed in provenance.

    MODIFY HERE: ``condition_col`` / ``control_label`` / ``perturbation_col`` to
    match a different screen's obs schema; ``min_perturbations`` to set what
    counts as "genome-scale enough"; ``pca=False`` to keep the full
    gene-by-gene delta matrix.
    """
    datasets = datasets or {}
    prov_base = dict(
        name="perturbation_pseudobulk",
        modality="pseudo-bulk expression change per perturbed gene (Perturb-seq)",
        category="causal",
    )
    usable = {k: d for k, d in datasets.items()
              if d.has_expression and d.obs is not None and condition_col in d.obs.columns}
    if not usable:
        return make_features(
            genes, None, status="deferred_no_perturbation_dataset", dim=0, **prov_base,
            reason=(f"no CandidateDataset with expression and an obs['{condition_col}'] column was "
                    "supplied; a Perturb-seq-style screen is required"),
            n_perturbations=0, datasets_seen=sorted(datasets),
        )

    deltas: Dict[str, np.ndarray] = {}
    feature_axis: Dict[str, List[str]] = {}
    diag: Dict[str, Any] = {"per_dataset": {}}
    for dname, ds in usable.items():
        obs, expr = ds.obs, ds.expression
        cond = obs[condition_col].astype(str)
        is_ctrl = (cond.str.lower() == control_label.lower()).to_numpy()
        pert_key = (perturbation_col if (perturbation_col and perturbation_col in obs.columns)
                    else condition_col)
        pert_all = obs[pert_key].astype(str).to_numpy()
        labels = sorted(set(pert_all[~is_ctrl]) - {control_label})
        ctrl_n = int(is_ctrl.sum())
        d: Dict[str, Any] = {"n_control_cells": ctrl_n, "n_perturbed_cells": int((~is_ctrl).sum()),
                             "perturbation_labels": labels, "n_perturbations": len(labels),
                             "perturbation_column": pert_key}
        if ctrl_n >= 2 and labels:
            ctrl_mean = expr.to_numpy(dtype=np.float64)[is_ctrl].mean(axis=0)
            X = expr.to_numpy(dtype=np.float64)
            small, norms = [], {}
            for lab in labels:
                rows = (pert_all == lab) & (~is_ctrl)
                if rows.sum() < 2:
                    small.append(lab)
                    continue
                delta = X[rows].mean(axis=0) - ctrl_mean
                if lab not in deltas:
                    deltas[lab] = delta.astype(np.float32)
                    feature_axis[lab] = list(expr.columns)
                norms[lab] = float(np.linalg.norm(delta))
            if small:
                d["dropped_lt2_cells"] = small
            d["delta_l2_norm"] = norms
            d["feature_axis_n_genes"] = int(expr.shape[1])
        diag["per_dataset"][dname] = d

    universe_keys = {gene_key(g) for g in genes}
    gene_labels = {lab: lab for lab in deltas if gene_key(lab) in universe_keys}
    n_pert = len(deltas)

    if n_pert < int(min_perturbations):
        return make_features(
            genes, None, status="deferred_insufficient_perturbations", dim=0, **prov_base,
            reason=(f"{n_pert} distinct perturbation(s) found across {len(usable)} dataset(s); a "
                    f"per-gene genome-wide feature map needs >= {min_perturbations}. One knockout "
                    "gives one delta vector, not one per gene; a double knockout cannot even be "
                    "attributed to a single gene. Marked deferred rather than returning a "
                    "constant one-column feature for every gene."),
            n_perturbations=n_pert, perturbation_labels=sorted(deltas),
            labels_matching_universe=sorted(gene_labels),
            min_perturbations=int(min_perturbations), diagnostics=diag,
        )

    # --- the real branch: one delta vector per perturbed gene ---------------
    feat_genes = sorted(gene_labels)
    axis0 = feature_axis[feat_genes[0]]
    same_axis = all(feature_axis[g] == axis0 for g in feat_genes)
    if not same_axis:                    # align on the shared measured genes
        shared = sorted(set.intersection(*[set(feature_axis[g]) for g in feat_genes]))
        pos = {g: [feature_axis[g].index(s) for s in shared] for g in feat_genes}
        X = np.vstack([deltas[gene_labels[g]][pos[g]] for g in feat_genes]).astype(np.float32)
        axis_note = f"aligned on {len(shared)} genes shared by all screens"
    else:
        X = np.vstack([deltas[gene_labels[g]] for g in feat_genes]).astype(np.float32)
        axis_note = f"{len(axis0)} measured genes"
    info: Dict[str, Any] = {"pca": False}
    if pca and X.shape[1] > dim:
        X, info = _pca_rows(X, dim, seed=0)
    values = {g: X[i] for i, g in enumerate(feat_genes)}
    return make_features(
        genes, values, status="ok", **prov_base,
        n_perturbations=n_pert, perturbation_labels=sorted(deltas),
        min_perturbations=int(min_perturbations), reduction=info, feature_axis=axis_note,
        normalisation="difference of per-group means of log1p_cp10k (log fold change vs control)",
        diagnostics=diag,
    )


# ===========================================================================
# 2. Optical pooled screens  (paper source 2)
# ===========================================================================

@PRIORS.register(
    "ops_morphology",
    modality="CellProfiler morphology profiles from optical pooled screening (PERISCOPE)",
    category="non_transcriptomic",
    needs_network=False,
    description="image-based morphology prior; no mouse genome-wide OPS exists (deferred)",
)
def ops_morphology(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    **_,
) -> PriorFeatures:
    """phi(x) = guide-normalised CellProfiler morphology profile of perturbing *x*.

    The paper uses PERISCOPE (Ramezani et al.), a **human** genome-wide optical
    pooled screen in A549 and HeLa cells. There is no mouse genome-wide optical
    pooled screen, so for *Mus musculus* this feature map has no data: it is
    returned all-NaN with ``status='deferred_no_mouse_resource'``.

    Downloading the human PERISCOPE profiles and indexing them by mouse symbol
    would not be this prior -- it would be a human morphology prior wearing a
    mouse label -- so it is not done here.

    MODIFY HERE -- the one defensible route to this prior for mouse
    ---------------------------------------------------------------
    1. Map each mouse symbol to its human orthologue (MGI
       ``HOM_MouseHumanSequence.rpt`` or Ensembl Compara); keep only 1:1 pairs
       and record the mapping rate (~80% of protein-coding genes).
    2. Pull the PERISCOPE CellProfiler features and take the **guide-normalised
       median profile per gene**. Their per-plate, per-well normalisation is not
       optional: raw CellProfiler features carry plate and batch effects larger
       than the gene effect.
    3. Feed those profiles in through ``values={mouse_symbol: profile}`` below
       and set ``status='ok_human_orthologue_morphology'`` -- a status that names
       the substitution, so the fusion report shows what was actually used.

    Observation unit: one gene (one guide-aggregated perturbation).
    Normalisation: see step 2 -- guide/plate normalisation upstream.
    Missing data: all genes, by construction, until the route above is built.
    """
    return make_features(
        genes, None, status="deferred_no_mouse_resource", dim=0,
        name="ops_morphology",
        modality="CellProfiler morphology profiles from optical pooled screening (PERISCOPE)",
        category="non_transcriptomic",
        reason=("PERISCOPE is a human genome-wide optical pooled screen (A549/HeLa); no mouse "
                "genome-wide OPS resource exists. Deferred rather than substituting human data "
                "under a mouse label. See the docstring for the orthologue route."),
        organism="mus_musculus", would_need="mouse->human 1:1 orthologues + PERISCOPE profiles",
    )


# ===========================================================================
# 3. scRNA-seq atlas co-expression  (paper source 3)  -- IMPLEMENTED
# ===========================================================================

@PRIORS.register(
    "coexpression_atlas",
    modality="per-gene expression vector across atlas cells (co-expression)",
    category="single_cell",
    needs_network=False,
    description="z-scored gene expression across atlas cells; its linear kernel is the "
                "gene-gene correlation matrix",
)
def coexpression_atlas(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    max_obs: Optional[int] = 4000,
    pca: bool = True,
    min_detected_cells: int = 3,
    category: str = "single_cell",
    **_,
) -> PriorFeatures:
    """phi(x) = gene *x*'s expression across a collection of scRNA-seq cells, z-scored.

    The paper
    ---------
    "phi(x) is the list of normalized gene expression measurements for gene x
    across a collection of scRNA-seq experiments ... the kernel matrix derived
    from this feature map corresponds to the co-expression matrix".

    Why per-gene standardisation is not optional
    --------------------------------------------
    For a row-standardised matrix ``Z`` (each gene mean 0, sd 1 across the N
    cells), the linear kernel is ``Z Z^T / N``, whose ``(i, j)`` entry is exactly
    the Pearson correlation of genes ``i`` and ``j`` across those cells. That
    identity *is* the paper's "kernel ... corresponds to the co-expression
    matrix". Without per-gene centring and scaling the linear kernel is
    dominated by mean expression level (a housekeeping gene would look similar
    to everything), which is a different quantity. ``kernels.prior_kernel`` keeps
    the ``1/N`` convention, so ``kind='linear'`` on this feature map reproduces
    ``np.corrcoef`` to numerical precision -- asserted in the tests.

    Observation unit
    ----------------
    One gene is one observation (one row). The *feature* axis is cells, and
    cells from different datasets are different observations of the same gene,
    so datasets are concatenated along the feature axis -- after per-dataset
    standardisation, so a platform with a larger dynamic range (Stereo-seq bins
    vs MARS-seq wells) cannot dominate the kernel. Mixing observation units
    (cell / spot / bin) is a modelling choice and is recorded per dataset in
    provenance.

    Normalisation
    -------------
    Expects the package's canonical ``log1p_cp10k``; the loaders guarantee it.
    Then per gene, per dataset: z-score across cells. If ``pca=True`` the
    z-scored block is reduced to ``dim`` components over the cell axis; PCA is
    fitted on covered genes only, and the resulting kernel is the correlation
    matrix projected onto its top-``dim`` eigen-directions (set ``pca=False``
    for the exact correlation kernel).

    Missing data
    ------------
    A gene absent from a dataset contributes NaN to that dataset's block; a gene
    absent from *every* dataset, or detected in fewer than ``min_detected_cells``
    cells everywhere, or constant everywhere, is an all-NaN row. When several
    datasets are combined, a gene covered by some but not all of them has NaN in
    the uncovered blocks; those entries are filled with 0 -- the gene's own mean
    *after z-scoring*, i.e. "no deviation information from that dataset" -- and
    the number of genes affected is reported as ``n_block_imputed``.

    MODIFY HERE: ``max_obs`` (cells sampled per dataset -- raise for a smoother
    correlation estimate, costs memory), ``pca``/``dim``, ``min_detected_cells``
    (detection filter), ``category`` (set ``'spatial'`` when the collection is
    spatial, so the embedding inventory tags it correctly).
    """
    datasets = datasets or {}
    expr_sets = {k: d for k, d in datasets.items() if d.has_expression}
    base = dict(name="coexpression_atlas",
                modality="per-gene expression vector across atlas cells (co-expression)",
                category=category)
    if not expr_sets:
        return make_features(genes, None, status="deferred_no_expression_dataset", dim=0, **base,
                             reason="no CandidateDataset with an expression matrix was supplied",
                             datasets_seen=sorted(datasets))
    rng = np.random.default_rng(0) if rng is None else rng
    genes = [str(g) for g in genes]
    blocks: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    per_ds: Dict[str, Any] = {}
    for dname, ds in expr_sets.items():
        expr = ds.expression
        if max_obs is not None and expr.shape[0] > max_obs:
            idx = np.sort(rng.choice(expr.shape[0], size=int(max_obs), replace=False))
            expr = expr.iloc[idx]
        # positional gene resolution; ambiguous (case-collapsing) symbols dropped
        lookup: Dict[str, str] = {}
        dupes: set = set()
        for c in expr.columns:
            k = gene_key(c)
            if k in lookup:
                dupes.add(k)
            lookup[k] = c
        for k in dupes:
            lookup.pop(k, None)
        hit = [lookup.get(gene_key(g)) for g in genes]
        present = [h for h in hit if h is not None]
        M = np.full((len(genes), expr.shape[0]), np.nan, dtype=np.float32)
        if present:
            rows = np.array([i for i, h in enumerate(hit) if h is not None])
            M[rows] = expr.loc[:, present].to_numpy(dtype=np.float32).T
        detected = np.nansum(M > 0, axis=1)
        M[detected < int(min_detected_cells)] = np.nan
        Z, usable = zscore_rows(M)
        blocks.append(Z)
        masks.append(usable)
        per_ds[dname] = {
            "n_obs_used": int(expr.shape[0]), "obs_unit": ds.obs_unit, "units": ds.units,
            "n_genes_in_dataset": int(expr.shape[1]),
            "n_universe_genes_present": int(sum(h is not None for h in hit)),
            "n_usable_after_filters": int(usable.sum()),
            "ambiguous_symbols_dropped": len(dupes),
        }
    Zcat = np.hstack(blocks)
    covered = np.any(np.vstack(masks), axis=0)
    if covered.sum() == 0:
        return make_features(genes, None, status="deferred_no_usable_genes", dim=0, **base,
                             reason="no universe gene survived detection/variance filters",
                             per_dataset=per_ds)
    X = Zcat[covered]
    n_block_imputed = 0
    if np.isnan(X).any():
        n_block_imputed = int(np.isnan(X).any(axis=1).sum())
        X = np.nan_to_num(X, nan=0.0)      # 0 == the gene's own mean after z-scoring
    info: Dict[str, Any] = {"pca": False, "note": "exact z-scored cell features retained"}
    if pca and X.shape[1] > dim:
        X, info = _pca_rows(X, dim, seed=0)
    cov_genes = [g for g, c in zip(genes, covered) if c]
    values = {g: X[i] for i, g in enumerate(cov_genes)}
    return make_features(
        genes, values, status="ok", **base,
        datasets_used=sorted(expr_sets), per_dataset=per_ds,
        normalisation="log1p_cp10k, then per-gene z-score across cells within each dataset",
        max_obs=max_obs, min_detected_cells=int(min_detected_cells),
        n_block_imputed=n_block_imputed, reduction=info,
        kernel_note="linear kernel of this feature map = gene-gene Pearson correlation",
    )


# ===========================================================================
# 4. Protein sequence / structure  (paper source 4)
# ===========================================================================

def _parse_uniprot_fasta(path: Path) -> pd.DataFrame:
    """Parse a UniProt FASTA into ``accession, gene, sequence, length``.

    Gene symbol comes from the ``GN=`` field of the header; entries without one
    (uncharacterised ORFs) are dropped rather than guessed at.
    """
    recs, acc, gene, buf = [], None, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None and gene and buf:
                    recs.append((acc, gene, "".join(buf)))
                head = line[1:].rstrip("\n")
                parts = head.split("|")
                acc = parts[1] if len(parts) > 2 else head.split()[0]
                gene = None
                for tokidx in head.split(" "):
                    if tokidx.startswith("GN="):
                        gene = tokidx[3:].strip()
                        break
                buf = []
            else:
                buf.append(line.strip())
    if acc is not None and gene and buf:
        recs.append((acc, gene, "".join(buf)))
    df = pd.DataFrame(recs, columns=["accession", "gene", "sequence"])
    df["length"] = df["sequence"].str.len()
    return df


def _uniprot_mouse_sequences(
    *, cache_dir: str | os.PathLike, offline: bool, reviewed_only: bool = True,
    rest_max_seconds: float = 240.0,
) -> tuple[pd.DataFrame, str]:
    """Cached UniProt mouse sequence table indexed by primary gene symbol.

    One sequence per symbol: the longest, which is the canonical isoform of a
    Swiss-Prot entry.

    Two routes, in order, because the first one is not reliable at this size:

    1. ``rest.uniprot.org/uniprotkb/stream`` with the query
       ``(organism_id:10090) AND (reviewed:true)`` -- the documented API, and
       the one that lets ``reviewed_only`` mean what it says.
    2. The UniProt **reference proteome** FASTA for *Mus musculus*
       (``UP000000589``) from the EBI mirror. Used only if route 1 fails or
       exceeds ``rest_max_seconds``; it is one static file and downloads in
       seconds. The reference proteome is one protein per gene for the whole
       proteome, so it includes unreviewed (TrEMBL) entries: with this route
       ``reviewed_only`` cannot be honoured, and the returned source string
       records that the route was used so ``provenance['source']`` shows it.

    MODIFY HERE: ``reviewed_only=False`` to ask route 1 for TrEMBL as well;
    ``rest_max_seconds`` to give the REST route more time before falling back.
    """
    tag = "reviewed" if reviewed_only else "all"
    dest = cache_path(cache_dir, f"uniprot_mouse_{tag}.tsv.gz")
    if dest.exists() and dest.stat().st_size > 0:
        df = pd.read_csv(dest, sep="\t", compression="gzip")
        src = str(dest)
    else:
        query = f"(organism_id:{MOUSE_TAXON})" + (" AND (reviewed:true)" if reviewed_only else "")
        try:
            fetch_to_cache(
                UNIPROT_STREAM_URL, dest, offline=offline,
                params={"query": query, "format": "tsv", "compressed": "true",
                        "fields": "accession,gene_primary,sequence,length"},
                timeout=120, max_seconds=rest_max_seconds,
            )
            df = pd.read_csv(dest, sep="\t", compression="gzip")
            src = f"{UNIPROT_STREAM_URL}?query={query}"
        except PriorFetchError:
            fa = cache_path(cache_dir, f"{UNIPROT_PROTEOME_ID}_{MOUSE_TAXON}.fasta.gz")
            fetch_to_cache(UNIPROT_FASTA_URL, fa, offline=offline, timeout=300,
                           max_seconds=900)
            df = _parse_uniprot_fasta(fa)
            df.to_csv(dest, sep="\t", index=False, compression="gzip")
            src = UNIPROT_FASTA_URL + " (reference proteome FASTA; REST stream unavailable, "
            src += "so this route includes unreviewed entries)"
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    gcol = next((c for c in df.columns if c.startswith("gene")), df.columns[1])
    df = df.rename(columns={gcol: "gene"})
    df = df[df["gene"].notna() & df["sequence"].notna()].copy()
    df["gene"] = df["gene"].astype(str).str.split(";").str[0].str.strip()
    df["length"] = df["sequence"].astype(str).str.len()
    df = df.sort_values("length", ascending=False).drop_duplicates("gene", keep="first")
    return df.set_index("gene"), src


@PRIORS.register(
    "protein_sequence",
    modality="protein language model embedding of the UniProt coding sequence (ESM-2)",
    category="non_transcriptomic",
    needs_network=True,
    description="ESM-2 residue-pooled embedding per gene product",
)
def protein_sequence(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    checkpoint: str = "facebook/esm2_t12_35M_UR50D",
    pooling: str = "mean",
    max_len: int = 1022,
    batch_size: int = 8,
    reviewed_only: bool = True,
    reduce_dim: bool = False,
    prefer_registry: bool = True,
    max_genes: Optional[int] = None,
    n_threads: Optional[int] = None,
    progress_every: int = 0,
    **_,
) -> PriorFeatures:
    """phi(x) = ESM-2 embedding of gene *x*'s protein sequence.

    The paper feeds the UniProt coding sequence into ESM-2 and takes "the [CLS]
    token"; they used the 15B checkpoint. This package is CPU-bound, so the
    default is ``facebook/esm2_t12_35M_UR50D`` and the checkpoint is a
    parameter -- whichever ran is recorded in ``provenance['checkpoint']``.

    Pooling, and why the default differs from the paper
    ---------------------------------------------------
    ESM-2 prepends a ``<cls>`` BOS token, but unlike BERT's ``[CLS]`` it is not
    trained with any sentence-level objective (ESM-2 is masked-LM only), so that
    position is not a trained sequence summary; mean-over-residues is the
    better-founded sequence representation and is the default here.
    ``pooling='bos'`` reproduces the paper's variant. Mean pooling **excludes**
    the BOS and EOS positions and all padding -- including them mixes
    non-residue vectors into the average and makes the embedding depend on the
    batch composition.

    Preference order
    ----------------
    If ``'esm2_sequence'`` is registered in ``panelprep.embed.EMBEDDERS`` it is
    called through the registry and its output is wrapped (so the two modules
    cannot drift); otherwise the local implementation runs.
    ``provenance['path']`` says which one did.

    Observation unit: one gene, represented by one canonical protein (the
    longest Swiss-Prot sequence for that symbol).
    Normalisation: none applied to the raw model output; use
    ``prior_kernel(kind='cosine')`` downstream, which is scale-free.
    Missing data: genes with no Swiss-Prot sequence (non-coding, un-reviewed, or
    a symbol UniProt does not use) are NaN rows. Sequences longer than
    ``max_len`` residues are **truncated**, counted in
    ``provenance['n_truncated']``.

    MODIFY HERE: ``checkpoint`` (e.g. ``facebook/esm2_t33_650M_UR50D`` for the
    paper's scale of model), ``pooling``, ``max_len``, ``batch_size``,
    ``reviewed_only`` (add TrEMBL for higher coverage), ``max_genes`` (runtime
    cap -- it is reported in ``provenance['max_genes']``, and the full
    transcriptome path is simply ``max_genes=None``).
    """
    base = dict(name="protein_sequence",
                modality="protein language model embedding of the UniProt coding sequence (ESM-2)",
                category="non_transcriptomic")
    genes = [str(g) for g in genes]
    t0 = time.time()
    base_err = None

    # --- prefer the other worker's registered embedder ---------------------
    if prefer_registry:
        try:
            from ..embed import EMBEDDERS

            if "esm2_sequence" in EMBEDDERS:
                emb = EMBEDDERS["esm2_sequence"].fn(
                    datasets or {}, genes=genes, dim=dim,
                    rng=np.random.default_rng(0) if rng is None else rng,
                )
                if isinstance(emb, dict):
                    emb = next(iter(emb.values()))
                M = pd.DataFrame(emb.matrix).astype(np.float32)
                arr = M.to_numpy(dtype=float)
                vals = {g: arr[i] for i, g in enumerate(M.index)
                        if np.isfinite(arr[i]).any()}
                return make_features(
                    genes, vals, status="ok", **base,
                    path="panelprep.embed.EMBEDDERS['esm2_sequence']",
                    delegated_provenance=getattr(emb, "provenance", {}),
                    wall_seconds=round(time.time() - t0, 2),
                )
        except Exception as exc:  # noqa: BLE001 - recorded, then fall back
            base_err = f"{type(exc).__name__}: {exc}"

    # --- local implementation ---------------------------------------------
    try:
        seqs, src = _uniprot_mouse_sequences(cache_dir=cache_dir, offline=offline,
                                             reviewed_only=reviewed_only)
    except PriorFetchError as exc:
        return make_features(genes, None, status="blocked_sequence_source_unavailable", dim=0,
                             **base, reason=str(exc), checkpoint=checkpoint)
    by_key = {gene_key(g): g for g in seqs.index}
    want = [(g, by_key[gene_key(g)]) for g in genes if gene_key(g) in by_key]
    n_available = len(want)
    if max_genes is not None and len(want) > max_genes:
        want = want[: int(max_genes)]
    if not want:
        return make_features(genes, None, status="deferred_no_sequences", dim=0, **base,
                             reason="no requested symbol matched a UniProt mouse entry",
                             source=src, checkpoint=checkpoint)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return make_features(genes, None, status="blocked_missing_dependency", dim=0, **base,
                             reason=f"torch/transformers unavailable: {exc}", checkpoint=checkpoint)
    if n_threads:
        torch.set_num_threads(int(n_threads))
    torch.manual_seed(0)
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
        try:
            # ESM's pooler head is not in the checkpoint; loading it would add a
            # randomly initialised layer. We only read last_hidden_state, but an
            # un-initialised parameter in the graph is worth not having.
            model = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False).eval()
        except TypeError:
            model = AutoModel.from_pretrained(checkpoint).eval()
    except Exception as exc:  # noqa: BLE001
        return make_features(genes, None, status="blocked_model_unavailable", dim=0, **base,
                             reason=f"could not load {checkpoint}: {type(exc).__name__}: {exc}",
                             checkpoint=checkpoint, source=src)

    n_trunc = 0
    values: Dict[str, np.ndarray] = {}
    # Sort by length so a batch is not padded to the longest protein in the
    # proteome; pooling excludes padding, so this changes runtime only.
    want = sorted(want, key=lambda gu: int(seqs.loc[gu[1], "length"]))
    with torch.no_grad():
        for s in range(0, len(want), int(batch_size)):
            chunk = want[s: s + int(batch_size)]
            raw = []
            for _, uni in chunk:
                seq = str(seqs.loc[uni, "sequence"])
                if len(seq) > max_len:
                    seq = seq[:max_len]
                    n_trunc += 1
                raw.append(seq)
            enc = tok(raw, return_tensors="pt", padding=True, truncation=True,
                      max_length=int(max_len) + 2)
            out = model(**enc).last_hidden_state            # (B, L, H)
            if pooling == "bos":
                pooled = out[:, 0, :]
            elif pooling == "mean":
                am = enc["attention_mask"].clone()
                am[:, 0] = 0                                 # drop BOS
                last = enc["attention_mask"].sum(1) - 1      # per-row EOS position
                am[torch.arange(am.shape[0]), last] = 0      # drop EOS
                w = am.unsqueeze(-1).to(out.dtype)
                pooled = (out * w).sum(1) / w.sum(1).clamp(min=1)
            else:
                raise ValueError("pooling must be 'mean' or 'bos'")
            for (g, _), v in zip(chunk, pooled.cpu().numpy().astype(np.float32)):
                values[g] = v
            if progress_every and (s // int(batch_size)) % int(progress_every) == 0:
                print(f"[protein_sequence] {len(values)}/{len(want)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    info: Dict[str, Any] = {"pca": False}
    if reduce_dim and values:
        keys = list(values)
        X, info = _pca_rows(np.vstack([values[k] for k in keys]), dim, seed=0)
        values = {k: X[i] for i, k in enumerate(keys)}
    return make_features(
        genes, values, status="ok", **base,
        path="panelprep.priors.sources.protein_sequence (local; 'esm2_sequence' embedder "
             + (f"errored: {base_err}" if base_err else "not registered") + ")",
        checkpoint=checkpoint, pooling=pooling, max_len=int(max_len),
        n_truncated=int(n_trunc), source=src, reviewed_only=bool(reviewed_only),
        n_symbols_with_sequence=int(n_available), max_genes=max_genes, reduction=info,
        normalisation="none (raw pooled hidden state); use a cosine kernel downstream",
        wall_seconds=round(time.time() - t0, 2),
    )


# ===========================================================================
# 5. PPI network  (paper source 5)
# ===========================================================================

def string_mouse_graph(
    *,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    score_threshold: int = 400,
):
    """STRING mouse functional-association graph as a ``networkx.Graph`` of symbols.

    Edges are kept at ``combined_score >= score_threshold`` (STRING's own
    "medium confidence" cut is 400; 700 is "high"). Scores are carried as
    ``weight`` on a 0-1 scale, which node2vec uses to bias walks.

    MODIFY HERE: ``score_threshold``; or filter on a specific evidence channel
    by downloading ``protein.links.detailed`` instead -- note that the combined
    score includes the ``textmining`` and ``database`` channels, so a kernel
    built from it is not a pure physical-interaction prior.
    """
    import networkx as nx

    info_p = cache_path(cache_dir, f"{MOUSE_TAXON}.protein.info.{STRING_VERSION}.txt.gz")
    links_p = cache_path(cache_dir, f"{MOUSE_TAXON}.protein.links.{STRING_VERSION}.txt.gz")
    fetch_to_cache(STRING_INFO_URL, info_p, offline=offline)
    fetch_to_cache(STRING_LINKS_URL, links_p, offline=offline, timeout=1800)
    info = pd.read_csv(info_p, sep="\t", compression="gzip")
    sym = dict(zip(info.iloc[:, 0].astype(str), info["preferred_name"].astype(str)))
    G = nx.Graph()
    n_lines = 0
    with gzip.open(links_p, "rt") as fh:
        fh.readline()
        for line in fh:
            a, b, s = line.split()
            n_lines += 1
            sc = int(s)
            if sc < score_threshold:
                continue
            ga, gb = sym.get(a), sym.get(b)
            if ga is None or gb is None or ga == gb:
                continue
            w = sc / 1000.0
            if G.has_edge(ga, gb):
                if G[ga][gb]["weight"] < w:
                    G[ga][gb]["weight"] = w
            else:
                G.add_edge(ga, gb, weight=w)
    meta = {"string_version": STRING_VERSION, "score_threshold": int(score_threshold),
            "n_link_rows": n_lines, "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "source": [STRING_INFO_URL, STRING_LINKS_URL]}
    return G, meta


def node2vec_embedding(
    G,
    *,
    dim: int = 64,
    walk_length: int = 40,
    n_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    epochs: int = 3,
    min_count: int = 0,
    seed: int = 0,
    workers: int = 1,
) -> tuple[Dict[str, np.ndarray], dict]:
    """node2vec (Grover & Leskovec) over a weighted graph: biased walks + skip-gram.

    Determinism: gensim's Word2Vec is only reproducible with ``workers=1`` (with
    more workers the order in which threads consume sentences changes the
    updates), so that is the default here, and the walks use an explicitly
    seeded ``numpy`` generator. ``tests/test_priors.py::test_determinism_*``
    asserts bitwise equality across two runs.

    ``p`` (return) and ``q`` (in-out) are the paper's parameters: ``q < 1``
    explores outward (community-scale structure), ``q > 1`` stays local.
    ``p = q = 1`` is an unbiased weighted walk (DeepWalk).

    MODIFY HERE: ``dim``, ``walk_length``, ``n_walks`` (runtime scales as
    ``n_nodes * n_walks * walk_length``), ``p``/``q``, ``window``, ``epochs``.
    """
    from gensim.models import Word2Vec

    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    nbrs = [np.array([idx[m] for m in G.neighbors(n)], dtype=np.int64) for n in nodes]
    wts = [np.array([G[n][m].get("weight", 1.0) for m in G.neighbors(n)], dtype=np.float64)
           for n in nodes]
    probs = [(w / w.sum() if w.size and w.sum() > 0 else w) for w in wts]
    nbr_sets = [set(a.tolist()) for a in nbrs]

    walks: List[List[str]] = []
    unbiased = (p == 1.0 and q == 1.0)
    for _ in range(int(n_walks)):
        order = rng.permutation(len(nodes))
        for start in order:
            if nbrs[start].size == 0:
                continue
            walk = [int(start)]
            while len(walk) < int(walk_length):
                cur = walk[-1]
                if nbrs[cur].size == 0:
                    break
                if len(walk) == 1 or unbiased:
                    nxt = int(rng.choice(nbrs[cur], p=probs[cur]))
                else:
                    prev = walk[-2]
                    cand, w = nbrs[cur], wts[cur].copy()
                    for i, c in enumerate(cand):
                        if c == prev:
                            w[i] /= p
                        elif c not in nbr_sets[prev]:
                            w[i] /= q
                    tot = w.sum()
                    nxt = int(rng.choice(cand, p=w / tot)) if tot > 0 else int(rng.choice(cand))
                walk.append(nxt)
            walks.append([nodes[i] for i in walk])
    model = Word2Vec(
        sentences=walks, vector_size=int(dim), window=int(window), min_count=int(min_count),
        sg=1, workers=int(workers), epochs=int(epochs), seed=int(seed), negative=5, sample=0,
    )
    vecs = {n: np.asarray(model.wv[n], dtype=np.float32) for n in nodes if n in model.wv}
    meta = {"algorithm": "node2vec", "dim": int(dim), "walk_length": int(walk_length),
            "n_walks": int(n_walks), "p": p, "q": q, "window": int(window),
            "epochs": int(epochs), "seed": int(seed), "workers": int(workers),
            "n_walks_generated": len(walks), "n_nodes_embedded": len(vecs)}
    return vecs, meta


def spectral_embedding(G, *, dim: int = 64, seed: int = 0) -> tuple[Dict[str, np.ndarray], dict]:
    """Laplacian eigenmaps on the largest connected component (the non-stochastic option).

    Nodes outside the largest component are left uncovered (NaN) rather than
    assigned an arbitrary zero vector.
    """
    import networkx as nx
    from scipy.sparse.linalg import eigsh

    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    nodes = sorted(H.nodes())
    L = nx.normalized_laplacian_matrix(H, nodelist=nodes).asfptype()
    k = int(min(dim + 1, L.shape[0] - 2))
    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    V = vecs[:, order][:, 1: dim + 1]          # drop the trivial constant eigenvector
    out = {n: np.asarray(V[i], dtype=np.float32) for i, n in enumerate(nodes)}
    return out, {"algorithm": "laplacian_eigenmaps", "dim": int(V.shape[1]),
                 "largest_component_nodes": len(nodes),
                 "eigenvalues_head": [float(v) for v in np.sort(vals)[:5]]}


@PRIORS.register(
    "ppi_network",
    modality="node embedding of the STRING protein-protein association network",
    category="non_transcriptomic",
    needs_network=True,
    description="node2vec (or spectral) embedding of STRING mouse PPI",
)
def ppi_network(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    score_threshold: int = 400,
    method: str = "node2vec",
    walk_length: int = 40,
    n_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    epochs: int = 3,
    seed: int = 0,
    restrict_to_universe: bool = False,
    prefer_registry_graph: bool = True,
    graph: Optional[Any] = None,
    **_,
) -> PriorFeatures:
    """phi(x) = node2vec embedding of gene *x* in the STRING mouse PPI graph.

    The paper uses "a node embedding such as node2vec on a PPI knowledge graph".
    Here that graph is STRING ``v12.0`` for taxon 10090 at
    ``combined_score >= score_threshold`` (default 400 = STRING "medium
    confidence"). ``method='spectral'`` swaps in Laplacian eigenmaps, which is
    deterministic without a seed and is a useful non-stochastic baseline.

    Observation unit: one gene, identified with one STRING protein through
    STRING's ``preferred_name``.
    Normalisation: none; node2vec vectors are compared with a cosine kernel.
    Missing data: genes that are not STRING nodes (or are isolated after
    thresholding, or -- for ``spectral`` -- outside the largest connected
    component) are NaN rows. ``provenance`` reports how many of the requested
    universe are STRING nodes, which is the honest ceiling on coverage for this
    prior.

    By default the graph is built over **all** mouse STRING proteins, not only
    the requested universe, because a node's embedding depends on its whole
    neighbourhood; ``restrict_to_universe=True`` induces the subgraph first,
    which is faster and strictly less informative. ``graph=`` accepts a
    pre-built ``networkx`` graph (used by the tests, and by any caller that
    wants to embed the same graph twice without re-parsing 14M STRING edges).

    MODIFY HERE: ``score_threshold``, ``method``, the node2vec hyper-parameters
    (``walk_length``, ``n_walks``, ``p``, ``q``, ``window``, ``epochs``),
    ``restrict_to_universe``.
    """
    base = dict(name="ppi_network",
                modality="node embedding of the STRING protein-protein association network",
                category="non_transcriptomic")
    genes = [str(g) for g in genes]
    t0 = time.time()
    G = graph
    graph_path = "caller-supplied graph" if graph is not None else \
                 "panelprep.priors.sources.string_mouse_graph"
    gmeta: Dict[str, Any] = {}
    if G is None and prefer_registry_graph:
        try:
            from ..embed import EMBEDDERS

            if "string_ppi" in EMBEDDERS:
                entry = EMBEDDERS["string_ppi"]
                builder = getattr(entry.fn, "build_graph", None) or entry.meta.get("build_graph")
                if callable(builder):
                    G = builder(cache_dir=cache_dir, offline=offline,
                                score_threshold=score_threshold)
                    if isinstance(G, tuple):
                        G, gmeta = G
                    graph_path = "panelprep.embed.EMBEDDERS['string_ppi'].build_graph"
        except Exception:  # noqa: BLE001 - fall back to the local builder
            G = None
    if G is None:
        try:
            G, gmeta = string_mouse_graph(cache_dir=cache_dir, offline=offline,
                                          score_threshold=score_threshold)
        except PriorFetchError as exc:
            return make_features(genes, None, status="blocked_string_unavailable", dim=0, **base,
                                 reason=str(exc), score_threshold=int(score_threshold))
    universe_keys = {gene_key(g) for g in genes}
    node_keys = {gene_key(n) for n in G.nodes()}
    n_universe_nodes = len(universe_keys & node_keys)
    if restrict_to_universe:
        keep = [n for n in G.nodes() if gene_key(n) in universe_keys]
        G = G.subgraph(keep).copy()
        gmeta = {**gmeta, "restricted_to_universe": True,
                 "n_nodes_after_restrict": G.number_of_nodes(),
                 "n_edges_after_restrict": G.number_of_edges()}
    if G.number_of_edges() == 0:
        return make_features(genes, None, status="deferred_empty_graph", dim=0, **base,
                             reason=f"no STRING edge survived score_threshold={score_threshold}",
                             graph=gmeta)
    if method == "node2vec":
        vecs, emeta = node2vec_embedding(G, dim=dim, walk_length=walk_length, n_walks=n_walks,
                                         p=p, q=q, window=window, epochs=epochs, seed=seed)
    elif method == "spectral":
        vecs, emeta = spectral_embedding(G, dim=dim, seed=seed)
    else:
        raise ValueError("method must be 'node2vec' or 'spectral'")
    by_key = {gene_key(k): k for k in vecs}
    values = {g: vecs[by_key[gene_key(g)]] for g in genes if gene_key(g) in by_key}
    return make_features(
        genes, values, status="ok", **base,
        graph_path=graph_path, graph=gmeta, embedding=emeta,
        n_universe_genes_that_are_string_nodes=int(n_universe_nodes),
        fraction_universe_in_string=float(n_universe_nodes / len(genes)) if genes else 0.0,
        score_threshold=int(score_threshold), method=method,
        normalisation="none (node embedding output); use a cosine kernel downstream",
        wall_seconds=round(time.time() - t0, 2),
    )


# ===========================================================================
# 6. Literature  (paper source 6)
# ===========================================================================

@PRIORS.register(
    "literature_text",
    modality="biomedical language model embedding of a templated sentence about the gene",
    category="non_transcriptomic",
    needs_network=True,
    description="BioGPT hidden state per gene symbol (last-token pooling)",
)
def literature_text(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    checkpoint: str = "microsoft/biogpt",
    pooling: str = "last",
    template: str = "{symbol} is a mouse gene.",
    batch_size: int = 16,
    reduce_dim: bool = False,
    max_genes: Optional[int] = None,
    n_threads: Optional[int] = None,
    progress_every: int = 0,
    **_,
) -> PriorFeatures:
    """phi(x) = BioGPT hidden state for a short sentence naming gene *x*.

    Two formulation corrections to the paper's description, both implemented
    -------------------------------------------------------------------------
    1. **There is no [CLS] token.** The paper says it feeds the gene name into
       BioGPT and takes "the [CLS] token embedding", but BioGPT is a
       decoder-only causal language model: it has no ``[CLS]``, and no position
       is trained to summarise the whole input. The causal analogue is the
       **last non-padding token**, the only position that has attended to every
       other token; that is the default (``pooling='last'``).
       ``pooling='mean'`` averages over non-padding tokens. Padding is excluded
       in both cases, otherwise the embedding would depend on the longest symbol
       in the batch.
    2. **A bare symbol is a weak prompt.** One or two tokens carry almost no
       context for a causal LM, so the input is built from a short template
       (``template``, default ``"{symbol} is a mouse gene."``). The template is a
       parameter precisely because it is a modelling choice.

    What this embedding is and is not
    ---------------------------------
    It reflects the model's *prior about the symbol string* learned from
    PubMed-scale text; it is **not** a retrieved document, an abstract-level
    summary, or evidence about this experiment. Genes whose symbols are rare in
    the literature (or are tokenised into unrelated word pieces) get embeddings
    driven by the template, which is why its kernel alignment with the other
    priors is the diagnostic to watch (see ``kernels.fuse_report``).

    Observation unit: one gene symbol.
    Normalisation: none; use a cosine kernel downstream.
    Missing data: none by construction -- every symbol can be tokenised, so
    coverage is 100% unless the model cannot be loaded (``blocked_*``) or
    ``max_genes`` caps the run (reported).

    MODIFY HERE: ``checkpoint`` (``microsoft/BioGPT-Large`` is the 1.5B variant;
    an encoder such as ``dmis-lab/biobert-v1.1`` is where a real ``[CLS]``
    exists), ``pooling``, ``template``, ``batch_size``, ``max_genes``.
    """
    base = dict(name="literature_text",
                modality="biomedical language model embedding of a templated sentence about the gene",
                category="non_transcriptomic")
    genes = [str(g) for g in genes]
    t0 = time.time()
    run = genes if max_genes is None else genes[: int(max_genes)]
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return make_features(genes, None, status="blocked_missing_dependency", dim=0, **base,
                             reason=f"torch/transformers unavailable: {exc}", checkpoint=checkpoint)
    if n_threads:
        torch.set_num_threads(int(n_threads))
    torch.manual_seed(0)
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).eval()
    except Exception as exc:  # noqa: BLE001
        return make_features(genes, None, status="blocked_model_unavailable", dim=0, **base,
                             reason=f"could not load {checkpoint}: {type(exc).__name__}: {exc}",
                             checkpoint=checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    values: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for s in range(0, len(run), int(batch_size)):
            chunk = run[s: s + int(batch_size)]
            texts = [template.format(symbol=g) for g in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            out = model(**enc).last_hidden_state
            am = enc["attention_mask"]
            if pooling == "last":
                last = am.sum(1) - 1
                pooled = out[torch.arange(out.shape[0]), last, :]
            elif pooling == "mean":
                w = am.unsqueeze(-1).to(out.dtype)
                pooled = (out * w).sum(1) / w.sum(1).clamp(min=1)
            else:
                raise ValueError("pooling must be 'last' or 'mean'")
            for g, v in zip(chunk, pooled.cpu().numpy().astype(np.float32)):
                values[g] = v
            if progress_every and (s // int(batch_size)) % int(progress_every) == 0:
                print(f"[literature_text] {len(values)}/{len(run)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    info: Dict[str, Any] = {"pca": False}
    if reduce_dim and values:
        keys = list(values)
        X, info = _pca_rows(np.vstack([values[k] for k in keys]), dim, seed=0)
        values = {k: X[i] for i, k in enumerate(keys)}
    return make_features(
        genes, values, status="ok", **base,
        checkpoint=checkpoint, pooling=pooling, template=template, max_genes=max_genes,
        reduction=info, hidden_size=int(model.config.hidden_size),
        caveat="model prior about the symbol string, not a retrieved document",
        normalisation="none (raw pooled hidden state); use a cosine kernel downstream",
        wall_seconds=round(time.time() - t0, 2),
    )


# ===========================================================================
# 7. Curated regulatory structure (KEGG) -- wrapper over the registered embedder
# ===========================================================================

def _kegg_pathway_membership(
    *, cache_dir: str | os.PathLike, offline: bool
) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    """``{symbol: [pathway ids]}`` for mouse from the KEGG REST API (cached)."""
    link_p = cache_path(cache_dir, "kegg_mmu_pathway_link.tsv")
    list_p = cache_path(cache_dir, "kegg_mmu_gene_list.tsv")
    fetch_to_cache(f"{KEGG_REST}/link/pathway/mmu", link_p, offline=offline)
    fetch_to_cache(f"{KEGG_REST}/list/mmu", list_p, offline=offline)
    sym: Dict[str, str] = {}
    with open(list_p) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            first = f[-1].split(";")[0].split(",")[0].strip()
            if first:
                sym[f[0]] = first
    memb: Dict[str, List[str]] = {}
    n_rows = 0
    with open(link_p) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != 2:
                continue
            n_rows += 1
            gid, pid = f[0], f[1].replace("path:", "")
            s = sym.get(gid)
            if s:
                memb.setdefault(s, []).append(pid)
    meta = {"source": [f"{KEGG_REST}/link/pathway/mmu", f"{KEGG_REST}/list/mmu"],
            "n_link_rows": n_rows, "n_genes_with_pathway": len(memb),
            "n_pathways": len({p for v in memb.values() for p in v})}
    return memb, meta


@PRIORS.register(
    "regulatory_kegg",
    modality="curated KEGG regulatory / pathway structure",
    category="causal",
    needs_network=True,
    description="thin wrapper over the registered 'kegg_causal' embedder; "
                "falls back to KEGG pathway membership",
)
def regulatory_kegg(
    genes: Sequence[str],
    *,
    dim: int = 64,
    rng=None,
    cache_dir: str | os.PathLike = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    prefer_registry: bool = True,
    min_pathway_size: int = 3,
    **_,
) -> PriorFeatures:
    """phi(x) = gene *x*'s position in curated KEGG regulatory structure.

    Preference order, and what each path actually gives you
    -------------------------------------------------------
    1. ``panelprep.embed.EMBEDDERS['kegg_causal']`` (owned by another worker):
       signed, directed activation/inhibition relations parsed from KGML. That
       is the causal-graph prior, and this function is a thin wrapper around it
       so the fusion can include curated regulatory structure without
       duplicating the parser.
    2. If that embedder is not registered, the fallback is **KEGG pathway
       membership** from ``rest.kegg.jp/link/pathway/mmu``: a gene x pathway
       indicator matrix reduced by truncated SVD. This is co-membership, **not**
       signed regulation -- a strictly weaker claim -- so it is reported as
       ``status='ok_fallback_pathway_membership'`` and the modality note says so.
       It is not presented as the signed causal graph.

    Observation unit: one gene. Normalisation: the indicator matrix is left
    unscaled before SVD (so a gene in many pathways has a larger norm, which the
    cosine kernel then removes). Missing data: genes in no KEGG mouse pathway
    (the majority of the transcriptome) are NaN rows.

    MODIFY HERE: ``min_pathway_size`` (drop tiny pathways), ``dim``,
    ``prefer_registry=False`` to force the fallback for comparison.
    """
    base = dict(name="regulatory_kegg", modality="curated KEGG regulatory / pathway structure",
                category="causal")
    genes = [str(g) for g in genes]
    t0 = time.time()
    reg_err = None
    if prefer_registry:
        try:
            from ..embed import EMBEDDERS

            if "kegg_causal" in EMBEDDERS:
                # Thread cache_dir and offline through. Omitting them let the
                # embedder read its own default cache, so an offline run with an
                # empty cache returned status 'ok' from data this call was never
                # given -- the opposite of the contract.
                emb = EMBEDDERS["kegg_causal"].fn(
                    datasets or {}, genes=genes, dim=dim,
                    rng=np.random.default_rng(0) if rng is None else rng,
                    cache_dir=str(cache_dir), offline=offline,
                )
                if isinstance(emb, dict):
                    emb = next(iter(emb.values()))
                M = pd.DataFrame(emb.matrix)
                arr = M.to_numpy(dtype=float)
                vals = {g: arr[i] for i, g in enumerate(M.index) if np.isfinite(arr[i]).any()}
                # A delegate that was itself blocked must not be reported 'ok'.
                delegated = dict(getattr(emb, "provenance", {}) or {})
                dstatus = str(delegated.get("status", "ok"))
                if dstatus.startswith("blocked") or not vals:
                    return make_features(
                        genes, None, status=dstatus if dstatus.startswith("blocked")
                        else "blocked_kegg_unavailable", dim=0, **base,
                        path="panelprep.embed.EMBEDDERS['kegg_causal']",
                        delegated_provenance=delegated,
                        reason="the kegg_causal embedder returned no usable rows "
                               f"(status={dstatus!r}, offline={offline})",
                        wall_seconds=round(time.time() - t0, 2),
                    )
                return make_features(
                    genes, vals, status="ok", **base,
                    path="panelprep.embed.EMBEDDERS['kegg_causal']",
                    delegated_provenance=delegated,
                    wall_seconds=round(time.time() - t0, 2),
                )
        except Exception as exc:  # noqa: BLE001
            reg_err = f"{type(exc).__name__}: {exc}"

    try:
        memb, kmeta = _kegg_pathway_membership(cache_dir=cache_dir, offline=offline)
    except PriorFetchError as exc:
        return make_features(genes, None, status="blocked_kegg_unavailable", dim=0, **base,
                             reason=str(exc), registry_error=reg_err)
    counts: Dict[str, int] = {}
    for v in memb.values():
        for pw in v:
            counts[pw] = counts.get(pw, 0) + 1
    paths = sorted(pw for pw, c in counts.items() if c >= int(min_pathway_size))
    pidx = {pw: i for i, pw in enumerate(paths)}
    by_key = {gene_key(k): k for k in memb}
    rows, keys = [], []
    for g in genes:
        k = by_key.get(gene_key(g))
        if k is None:
            continue
        v = np.zeros(len(paths), dtype=np.float32)
        hit = False
        for pw in memb[k]:
            if pw in pidx:
                v[pidx[pw]] = 1.0
                hit = True
        if hit:
            rows.append(v)
            keys.append(g)
    if not rows:
        return make_features(genes, None, status="deferred_no_kegg_membership", dim=0, **base,
                             reason="no requested gene is in a KEGG mouse pathway of size "
                                    f">= {min_pathway_size}", kegg=kmeta, registry_error=reg_err)
    X = np.vstack(rows)
    from sklearn.decomposition import TruncatedSVD

    k = max(1, int(min(dim, min(X.shape) - 1)))
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(X).astype(np.float32)
    values = {g: Z[i] for i, g in enumerate(keys)}
    return make_features(
        genes, values, status="ok_fallback_pathway_membership", **base,
        path="panelprep.priors.sources.regulatory_kegg (fallback; 'kegg_causal' embedder "
             + (f"errored: {reg_err}" if reg_err else "not registered") + ")",
        modality_note="KEGG pathway CO-MEMBERSHIP, not signed directed regulation",
        kegg=kmeta, n_pathways_used=len(paths), min_pathway_size=int(min_pathway_size),
        reduction={"algorithm": "TruncatedSVD", "n_components": k,
                   "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())},
        normalisation="binary indicator, unscaled, then truncated SVD",
        wall_seconds=round(time.time() - t0, 2),
    )


# ===========================================================================
# bridge into the package's embedding inventory
# ===========================================================================

def to_gene_embedding(features: PriorFeatures, dim: Optional[int] = None,
                      *, seed: int = 0) -> GeneEmbedding:
    """Wrap a prior as a :class:`panelprep.schemas.GeneEmbedding`.

    So a prior appears in the same inventory as the autoencoder embeddings and is
    plotted by ``panelprep.viz`` with no change there.

    Observation unit: one gene per row (unchanged). Missing data: NaN rows are
    preserved -- the viz layer decides whether to drop or grey them, which is
    better than this function inventing coordinates for genes with no data.
    If ``dim`` is smaller than the feature count, PCA is fitted **on covered
    rows only**; if it is larger, the matrix is returned unpadded and
    ``provenance['requested_dim']`` records the request.

    MODIFY HERE: swap PCA for another reducer (UMAP, t-SNE) if the inventory
    should carry a 2D view instead of a linear projection.
    """
    M = features.matrix
    prov = {**features.provenance, "prior_source": features.name,
            "modality": features.modality, "requested_dim": dim}
    if dim is not None and M.shape[1] > dim and features.n_covered > 1:
        cov = features.covered
        X = np.nan_to_num(M.to_numpy(dtype=np.float32)[cov], nan=0.0)
        Z, info = _pca_rows(X, dim, seed=seed)
        out = np.full((M.shape[0], Z.shape[1]), np.nan, dtype=np.float32)
        out[cov] = Z
        M = pd.DataFrame(out, index=M.index,
                         columns=[f"{features.name}_pc{i}" for i in range(Z.shape[1])])
        prov["embedding_reduction"] = info
    return GeneEmbedding(matrix=M, method=f"prior::{features.name}",
                         category=features.category, provenance=prov)


__all__ = [
    "PRIORS", "PriorFeatures", "PriorFetchError", "make_features", "to_gene_embedding",
    "mouse_protein_coding_universe", "string_mouse_graph", "node2vec_embedding",
    "spectral_embedding", "zscore_rows", "cache_path", "fetch_to_cache",
    "perturbation_pseudobulk", "ops_morphology", "coexpression_atlas", "protein_sequence",
    "ppi_network", "literature_text", "regulatory_kegg", "DEFAULT_CACHE",
]
