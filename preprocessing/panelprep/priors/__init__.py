"""Genome-wide multi-modal gene priors (ITERPERT part 1) and their kernel fusion.

Scope
-----
Implements the prior **feature maps**, their **kernelisation** and the **kernel
fusion** of Huang et al. (bioRxiv 2023.12.12.571389). The sequential /
active-learning design loop that the paper wraps around those kernels is out of
scope for this package.

Observation unit: the **gene**. Every matrix here is ``genes x features`` and
every kernel is ``genes x genes``; the gene universe is the mouse
protein-coding transcriptome (``sources.mouse_protein_coding_universe``), not a
measured panel.

Typical use::

    from panelprep.priors import build_priors, prior_kernel, fuse_kernels, fuse_report
    from panelprep.priors.sources import mouse_protein_coding_universe

    genes = mouse_protein_coding_universe(cache_dir="data/external/priors")
    feats = build_priors(genes, sources=["coexpression_atlas", "ppi_network"],
                         dim=64, datasets=datasets)
    kernels = {n: prior_kernel(f, kind="cosine") for n, f in feats.items() if f.ok}
    K, info = fuse_kernels(kernels, how="mean", missing="mask")
    report = fuse_report(kernels, features=feats)

MODIFY HERE
-----------
* A new prior source: add one function in ``sources.py`` (see its module
  docstring). ``build_priors``, ``to_gene_embedding`` and ``fuse_report`` pick it
  up with no edit here.
* Which sources a run uses: the ``sources=`` argument of ``build_priors`` -- or,
  from the package's single configuration entry point, a list in
  ``Config.embedder_params['priors']['sources']`` passed through by the caller.
  This module deliberately does not read configuration itself.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..schemas import CandidateDataset
from .kernels import (  # noqa: F401
    check_psd, fuse_kernels, fuse_report, kernel_alignment, normalise_kernel, prior_kernel,
)
from .sources import (  # noqa: F401
    DEFAULT_CACHE, PRIORS, PriorFeatures, PriorFetchError, make_features,
    mouse_protein_coding_universe, node2vec_embedding, spectral_embedding, string_mouse_graph,
    to_gene_embedding, zscore_rows,
)


def build_priors(
    genes: Sequence[str],
    *,
    sources: Sequence[str],
    dim: int = 64,
    cache_dir: str = DEFAULT_CACHE,
    offline: bool = False,
    datasets: Optional[Dict[str, CandidateDataset]] = None,
    rng=None,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Dict[str, PriorFeatures]:
    """Run the requested prior sources over one gene universe.

    Parameters
    ----------
    genes
        The gene universe, in order. Every returned ``PriorFeatures`` is indexed
        by exactly this list, so the kernels align positionally and
        ``fuse_kernels`` can refuse to reindex.
    sources
        Registry names from ``PRIORS``.
    dim
        Target feature dimension where a source reduces (PCA/SVD/node2vec).
    cache_dir, offline
        Download cache; ``offline=True`` forbids network access and turns a
        cache miss into a ``blocked_offline_no_cache`` status.
    datasets
        ``{name: CandidateDataset}`` for the data-driven sources
        (``coexpression_atlas``, ``perturbation_pseudobulk``). Ignored by the
        others.
    params
        ``{source_name: {kwarg: value}}`` per-source overrides.

    Failure behaviour
    -----------------
    A source that raises is **captured**, not propagated: the result carries
    ``status='blocked_exception'`` with the exception text in
    ``provenance['reason']``, so one broken source cannot abort a genome-scale
    run that has already paid for the others. Data problems (no mouse resource,
    too few perturbations, no network) are statuses by construction inside the
    sources themselves.

    Observation unit: gene. Normalisation and missing-data behaviour are
    per-source and documented on each source function.

    MODIFY HERE: ``params`` is the single place to pass source-specific options
    (e.g. ``{"protein_sequence": {"checkpoint": "facebook/esm2_t33_650M_UR50D"}}``).
    """
    genes = [str(g) for g in genes]
    params = params or {}
    out: Dict[str, PriorFeatures] = {}
    for name in sources:
        entry = PRIORS[name]
        kw = dict(params.get(name, {}))
        src_rng = np.random.default_rng(0) if rng is None else rng
        t0 = time.time()
        try:
            pf = entry.fn(genes, dim=dim, rng=src_rng, cache_dir=cache_dir, offline=offline,
                          datasets=datasets, **kw)
        except Exception as exc:  # noqa: BLE001 - a failure is a status, not a crash
            pf = make_features(
                genes, None, status="blocked_exception", dim=0, name=name,
                modality=str(entry.meta.get("modality", name)),
                category=str(entry.meta.get("category", "non_transcriptomic")),
                reason=f"{type(exc).__name__}: {exc}",
            )
        pf.provenance.setdefault("wall_seconds", round(time.time() - t0, 2))
        pf.provenance["registry_meta"] = dict(entry.meta)
        out[name] = pf
        if verbose:
            print(f"[build_priors] {name}: status={pf.status} covered={pf.n_covered}/{len(genes)} "
                  f"dim={pf.dim} ({pf.provenance['wall_seconds']}s)", flush=True)
    return out


def coverage_table(features: Dict[str, PriorFeatures]):
    """One row per prior source: status, coverage, feature dimension, wall time.

    This is the table the task's acceptance check reports; it is built from
    whatever ``build_priors`` returned, so a newly added source appears without
    editing anything.
    """
    import pandas as pd

    rows = []
    for name, pf in features.items():
        prov = pf.provenance
        rows.append({
            "source": name,
            "category": pf.category,
            "status": pf.status,
            "n_covered": pf.n_covered,
            "n_requested": int(prov.get("n_requested", len(pf.genes))),
            "coverage": round(float(prov.get("coverage", 0.0)), 4),
            "n_features": pf.dim,
            "wall_seconds": prov.get("wall_seconds"),
            "note": prov.get("reason") or prov.get("modality_note") or prov.get("checkpoint") or "",
        })
    return pd.DataFrame(rows).set_index("source")


__all__ = [
    "PRIORS", "PriorFeatures", "PriorFetchError", "build_priors", "coverage_table",
    "to_gene_embedding", "make_features", "mouse_protein_coding_universe",
    "string_mouse_graph", "node2vec_embedding", "spectral_embedding", "zscore_rows",
    "prior_kernel", "normalise_kernel", "fuse_kernels", "check_psd", "kernel_alignment",
    "fuse_report", "DEFAULT_CACHE",
]
