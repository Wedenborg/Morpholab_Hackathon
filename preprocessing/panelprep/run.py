"""One run, one configuration, cached embeddings: the notebook's entry point.

:func:`run_analysis` is what ``notebooks/05_run_analysis.ipynb`` calls. It takes
**one** :class:`~panelprep.config.Config` -- which already names the reference
datasets, the scores, the embedders and every parameter -- and executes the
whole pipeline for that choice:

1. load the chosen reference datasets (``config.candidates``);
2. score them against the measured panel (``config.scores``) and write the
   agreement heatmap, the raw heatmap and the markdown comparison report;
3. fix the gene universe the embeddings run over, with the panel pinned in;
4. obtain each gene embedding in ``config.embedders`` -- **loaded from the
   embedding store when it is already there**, computed only on a miss;
5. annotate the genes (pathway, ligand / receptor / TF);
6. render every figure via :func:`panelprep.viz.gallery.visualize_all`.

How "do not retrain" is enforced
-------------------------------
``esm2_sequence``, ``literature_text`` / BioGPT and ``ppi_network`` / node2vec
depend only on the gene symbol list, cost tens of minutes to hours, and return
the same answer every time. They are listed in :data:`CACHED_ONLY` and are never
recomputed by this function: if one is requested and the store does not have
it, the run reports it as ``blocked`` with the command that builds it, and
carries on with everything else. That is deliberate -- a notebook that silently
starts a two-hour protein-language-model job is worse than one that tells you
to run the builder.

The dataset-dependent expression embedders (PCA, AE, VAE, joint AE, deep CCA,
contrastive, factor analysis) are cached under the same interface but *are*
computed on a miss, because they are minutes, not hours, and they change
whenever the reference datasets or the gene universe change.

Use
---
>>> from panelprep.config import Config
>>> from panelprep.run import run_analysis
>>> config = Config.from_yaml("configs/real_pilot.yaml").replace(
...     candidates=["seu3d_E7.5_rep2"], embedders=["pca_baseline", "dcca"],
...     scores=["avg_js", "corr_frobenius", "corr_mae"])
>>> result = run_analysis(config, query=query)
>>> result["scores"]; result["figures"]; result["inventory"]

MODIFY HERE
-----------
* ``CACHED_ONLY`` -- which embedders must come from the store.
* :func:`run_analysis` ``stages`` -- skip scoring, embedding, annotation or
  figures for a faster partial run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .embed import EMBEDDERS, gene_universe, run_embedders
from .embed.store import EmbeddingStore
from .pipeline import annotate, load_candidates, match_candidates
from .schemas import GeneEmbedding, PanelQuery
from .scores import SCORES, run_scores, split_scoreable

__all__ = ["run_analysis", "CACHED_ONLY", "DEFAULT_STAGES"]

#: Embedders that must be loaded, never recomputed inside a run. MODIFY HERE.
CACHED_ONLY: frozenset[str] = frozenset({
    "esm2_sequence",      # ESM-2 over UniProt sequences      (~50 min / 2.5k genes)
    "literature_text",    # BioGPT                             (~45 min)
    "ppi_network",        # node2vec over STRING                (~35 min)
    "protein_sequence",   # prior-source alias for ESM-2
    "string_ppi",         # spectral embedding of STRING
    "kegg_causal",        # signed Laplacian of KEGG relations
    "regulatory_kegg",    # prior-source alias
})

#: Stages, in order. MODIFY HERE (or pass ``stages=`` to skip some).
DEFAULT_STAGES: tuple[str, ...] = ("match", "score", "embed", "annotate",
                                   "figures", "report")


def run_analysis(
    config: Config,
    *,
    query: Optional[PanelQuery] = None,
    store: Optional[EmbeddingStore] = None,
    candidates: Optional[Mapping[str, Any]] = None,
    annotation: Optional[Any] = None,
    stages: Sequence[str] = DEFAULT_STAGES,
    recompute: bool = False,
    highlight_panel: bool = True,
    projection: str = "umap",
    group_figures_by: Optional[str] = "store_group",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the whole analysis for one configuration and return everything it made.

    Parameters
    ----------
    config
        The single configuration entry point. Its ``candidates``, ``scores``,
        ``embedders``, ``embedding_dim``, ``max_genes``, ``seed`` and
        ``annotation_sources`` fields *are* the "particular choice of reference
        data, embedding method, similarity score and other parameters".
    query
        The measured panel. Required for the ``match`` and ``score`` stages;
        the embedding stages work without it.
    store
        Where cached embeddings live. Defaults to ``<config.outdir>/store``.
    candidates
        Pre-loaded ``{name: CandidateDataset}``; loaded from the catalog when
        omitted. Pass them in when you have already paid for the I/O.
    annotation
        Pre-computed :class:`GeneAnnotation`; fetched when omitted.
    stages
        Subset of :data:`DEFAULT_STAGES`.
    recompute
        Rebuild cached embeddings instead of loading them. Does **not** lift the
        :data:`CACHED_ONLY` rule -- use the builder script for those.
    highlight_panel
        Draw the measured panel genes as labelled stars in every figure.

    Returns
    -------
    dict
        ``config``, ``match``, ``scores``, ``score_table``, ``genes``,
        ``embeddings``, ``annotation``, ``inventory``, ``figures``,
        ``blocked``, ``timings`` -- plus the paths of everything written under
        ``config.out``.
    """
    stages = tuple(stages)
    t_all = time.time()
    timings: Dict[str, float] = {}
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    store = store or EmbeddingStore(out / "store")
    result: Dict[str, Any] = {
        "config": config.to_dict(), "stages_run": list(stages),
        "blocked": {}, "figures": {}, "timings": timings,
        "store_root": str(store.root),
    }
    _log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    _log(f"[run] {config.run_name}: stages={', '.join(stages)} -> {out}")

    # ------------------------------------------------------------- 1. load
    if candidates is None:
        t = time.time()
        candidates = load_candidates(config)
        timings["load_candidates"] = round(time.time() - t, 1)
        _log(f"[run] loaded {len(candidates)} candidate datasets "
             f"({timings['load_candidates']}s)")
    result["candidates"] = {
        k: {"category": v.category, "n_obs": int(v.n_obs) if v.has_expression else 0,
            "n_genes": len(v.genes) if v.has_expression else 0,
            "has_coords": bool(v.has_coords), "units": v.units}
        for k, v in candidates.items()}

    # ------------------------------------------------------------ 2. match
    if "match" in stages and query is not None:
        t = time.time()
        match = match_candidates(query, config)
        match.to_csv(out / "match_table.csv", index=False)
        result["match"] = match
        timings["match"] = round(time.time() - t, 1)

    # ------------------------------------------------------------ 3. score
    if "score" in stages and query is not None:
        t = time.time()
        scoreable, skipped = split_scoreable(candidates.values())
        scores = run_scores(
            query, scoreable, score_names=list(config.scores),
            genes=list(query.panel), params=config.score_params,
            seed=config.seed, max_obs=config.max_obs_per_dataset,
            allow_unit_mismatch=config.allow_unit_mismatch)
        scores.to_csv(out / "score_results.csv", index=False)
        result["scores"] = scores
        result["scores_skipped"] = {d.name: "no expression matrix" for d in skipped}
        table = scores.pivot(index="dataset", columns="score", values="value")
        result["score_table"] = table
        _log(f"[run] scored {table.shape[0]} datasets x {table.shape[1]} scores")

        # Both colourings always, because "low divergence" and "high agreement"
        # are the same fact with opposite signs and only one of them is red.
        from .viz import score_heatmap
        figures_dir.mkdir(parents=True, exist_ok=True)
        for direction, fname in (("agreement", "score_heatmap.png"),
                                 ("raw", "score_heatmap_raw.png")):
            tab = score_heatmap(
                scores, out_path=str(figures_dir / fname), direction=direction,
                title=f"{config.run_name}: panel-gene agreement"
                      if direction == "agreement" else
                      f"{config.run_name}: raw score values")
            result["figures"][f"score_heatmap_{direction}"] = str(figures_dir / fname)
            if direction == "agreement":
                result["ranking"] = getattr(tab, "ranking", None)
        timings["score"] = round(time.time() - t, 1)

    # ------------------------------------------------------------ 4. embed
    if "embed" in stages:
        t = time.time()
        panel = list(query.panel) if query is not None else []
        genes = gene_universe(
            candidates, how=config.gene_universe, max_genes=config.max_genes,
            rng=np.random.default_rng(config.seed), must_include=panel)
        result["genes"] = genes
        _log(f"[run] gene universe: {len(genes)} genes "
             f"({sum(g in set(genes) for g in panel)}/{len(panel)} panel genes pinned)")

        embeddings: Dict[str, GeneEmbedding] = {}
        for name in config.embedders:
            if name in CACHED_ONLY:
                hits = [n for n in store.names() if n == name or n.startswith(f"{name}::")]
                if not hits:
                    result["blocked"][name] = (
                        f"{name} is cache-only (it costs tens of minutes and depends "
                        f"only on the gene list). Not found in {store.root}. Build it "
                        f"with: python scripts/build_embedding_store.py --sources {name}")
                    _log(f"  [blocked] {name}: not in store")
                    continue
                for hit in hits:
                    embeddings[hit] = store.get(hit, genes=genes)
                    _log(f"  [load]  {hit} (cache-only source)")
                continue
            if name not in EMBEDDERS:
                # Not a registered embedder -- but it may still be a prior source
                # sitting in the store (those are built by the builder script and
                # have no embedder entry). Loading it is correct; refusing it
                # because of a registry miss would be an accounting error.
                hits = [n for n in store.names()
                        if n == name or n.startswith(f"{name}::")]
                if hits:
                    for hit in hits:
                        embeddings[hit] = store.get(hit, genes=genes)
                        _log(f"  [load]  {hit} (from store; no embedder entry)")
                    continue
                result["blocked"][name] = (
                    f"{name} is neither a registered embedder nor present in "
                    f"{store.root}")
                _log(f"  [blocked] {name}: not registered and not in store")
                continue
            mode = EMBEDDERS[name].meta.get("mode", "joint")
            keys = ([f"{name}::{d}" for d, v in candidates.items() if v.has_expression]
                    if mode == "separate" else [name])
            missing = [k for k in keys if not store.has(k)] if not recompute else keys
            if missing:
                _log(f"  [build] {name} ({mode}, {len(missing)} missing)")
                built = run_embedders(
                    candidates, embedder_names=[name], genes=genes,
                    dim=config.embedding_dim, params=config.embedder_params,
                    seed=config.seed, max_obs=config.max_obs_per_dataset)
                for k, v in built.items():
                    store.put(v, k, group="expression")
                    embeddings[k] = v
            for k in keys:
                if k not in embeddings and store.has(k):
                    embeddings[k] = store.get(k, genes=genes)
                    _log(f"  [load]  {k}")
        result["embeddings"] = embeddings
        timings["embed"] = round(time.time() - t, 1)
        _log(f"[run] {len(embeddings)} embeddings ready ({timings['embed']}s)")

    # --------------------------------------------------------- 5. annotate
    if "annotate" in stages and "genes" in result:
        t = time.time()
        if annotation is None:
            annotation = annotate(result["genes"], config)
        result["annotation"] = annotation
        if annotation is not None:
            annotation.table.to_csv(out / "gene_annotation.csv")
        timings["annotate"] = round(time.time() - t, 1)

    # ---------------------------------------------------------- 6. figures
    if "figures" in stages and result.get("embeddings"):
        t = time.time()
        from .viz.gallery import visualize_all
        manifest = visualize_all(
            result["embeddings"], result.get("annotation"), outdir=figures_dir,
            genes=result.get("genes"), group_by=group_figures_by,
            method=projection,
            highlight=(list(query.panel) if (highlight_panel and query is not None)
                       else None),
            seed=config.seed, verbose=verbose)
        result["figure_manifest"] = manifest
        result["figures"].update(dict(zip(manifest["figure"], manifest["path"])))
        timings["figures"] = round(time.time() - t, 1)

    # ----------------------------------------------------------- 7. report
    # Generated from the registries and the long-format score frame, so a newly
    # registered score or embedder appears here with no edit to this function.
    if "report" in stages:
        t = time.time()
        from .report import build_report
        path = out / "comparison_report.md"
        build_report(
            config, match_table=result.get("match"),
            score_results=result.get("scores"),
            embeddings=result.get("embeddings"),
            annotation=result.get("annotation"),
            figures=result.get("figures"),
            manual_status=result.get("blocked") or None,
            out_path=str(path))
        result["figures"]["comparison_report"] = str(path)
        timings["report"] = round(time.time() - t, 1)
        _log(f"[run] report -> {path}")

    # ------------------------------------------------------------ 8. close
    inv = store.inventory(write=True)
    result["inventory"] = inv
    inv.to_csv(out / "embedding_inventory.csv", index=False)
    timings["total"] = round(time.time() - t_all, 1)
    (out / "run_summary.json").write_text(json.dumps({
        "run_name": config.run_name,
        "config": config.to_dict(),
        "stages_run": list(stages),
        "n_candidates": len(candidates),
        "n_genes": len(result.get("genes", [])),
        "n_embeddings": len(result.get("embeddings", {})),
        "blocked": result["blocked"],
        "timings_seconds": timings,
        "figures": result["figures"],
    }, indent=2, default=str))
    _log(f"[run] done in {timings['total']}s; "
         f"{len(result['blocked'])} blocked; summary -> {out / 'run_summary.json'}")
    return result
