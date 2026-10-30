"""End-to-end driver: one function per stage, one ``run_pipeline`` that chains them.

Everything is driven by a single ``Config``. The notebooks call these functions;
they do not reimplement any of this.

Stages (numbered as in the brief)
---------------------------------
1a ``match_candidates``  descriptor -> ranked candidate datasets, with category tags
1b ``score_candidates``  panel-gene agreement scores -> long-format table + heatmap
2  ``embed_expression``  separate and joint autoencoder gene embeddings
3  ``embed_external``    sequence / PPI / causal gene embeddings
4  ``annotate``          pathway membership and ligand / receptor / TF roles
0  the union of 2+3 is the four-category gene-embedding deliverable

MODIFY HERE: to change *what runs*, edit ``Config`` (or the YAML), not this file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .io.catalog import resolve_candidates
from .io.loaders import load_candidate
from .match import match_descriptor
from .schemas import CandidateDataset, GeneAnnotation, GeneEmbedding, PanelQuery


# ---------------------------------------------------------------------------
# stage 1a: which datasets plausibly describe the same biology
# ---------------------------------------------------------------------------

def match_candidates(query: PanelQuery, config: Config) -> pd.DataFrame:
    entries = resolve_candidates(config.candidates)
    return match_descriptor(query.descriptor, entries,
                            method=config.match_method, top_k=config.match_top_k)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_candidates(config: Config, names: Optional[Sequence[str]] = None,
                    verbose: bool = True) -> Dict[str, CandidateDataset]:
    """Load catalogue entries into ``CandidateDataset`` objects.

    Loader failures are reported and skipped rather than aborting the run; the
    returned dict is the set that actually loaded, and the report says which
    entries were dropped and why.
    """
    out, failed = {}, {}
    rng = np.random.default_rng(config.seed)
    for e in resolve_candidates(names if names is not None else config.candidates):
        t0 = time.time()
        try:
            ds = load_candidate(e, config.data_dir, max_obs=config.max_obs_per_dataset, rng=rng)
            out[e["name"]] = ds
            if verbose:
                shape = ds.expression.shape if ds.has_expression else "(no matrix)"
                print(f"  loaded {e['name']:26s} {e['category']:18s} {shape}  {time.time()-t0:.1f}s")
        except Exception as exc:
            failed[e["name"]] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  FAILED {e['name']:26s} {failed[e['name']]}")
    if failed:
        out_failed = Path(config.outdir) / "load_failures.json"
        out_failed.parent.mkdir(parents=True, exist_ok=True)
        out_failed.write_text(json.dumps(failed, indent=1))
    return out


# ---------------------------------------------------------------------------
# stage 1b: agreement scores
# ---------------------------------------------------------------------------

def score_candidates(query: PanelQuery, candidates: Dict[str, CandidateDataset],
                     config: Config) -> pd.DataFrame:
    from .scores import run_scores

    return run_scores(query, list(candidates.values()),
                      score_names=config.scores, genes=query.panel,
                      params=config.score_params, seed=config.seed,
                      max_obs=config.max_obs_per_dataset,
                      allow_unit_mismatch=config.allow_unit_mismatch)


# ---------------------------------------------------------------------------
# stages 2 and 3: gene embeddings
# ---------------------------------------------------------------------------

def embedding_gene_universe(candidates: Dict[str, CandidateDataset], config: Config,
                            panel: Optional[Sequence[str]] = None) -> List[str]:
    """The gene list embeddings are computed over (item 0: the whole transcriptome).

    Not *only* the measured panel: the point of the deliverable is to have a
    representation for genes that were never measured. But the panel itself is
    pinned into the universe via ``must_include``, because a ``max_genes`` cap
    subsamples at random and would otherwise drop it -- and an embedding that
    does not contain the measured genes cannot show where they sit.
    """
    from .embed import gene_universe

    return gene_universe({k: v for k, v in candidates.items() if v.has_expression},
                         how=config.gene_universe, max_genes=config.max_genes,
                         rng=np.random.default_rng(config.seed),
                         must_include=panel)


def embed_genes(candidates: Dict[str, CandidateDataset], genes: Sequence[str],
                config: Config, verbose: bool = True) -> Dict[str, GeneEmbedding]:
    """Run every embedder in ``Config.embedders`` that is registered.

    Unregistered names and embedder failures are collected rather than raised,
    so one broken method cannot cost the whole run; they are reported as
    ``blocked`` in the comparison report.
    """
    from .embed import EMBEDDERS, run_embedders

    out: Dict[str, GeneEmbedding] = {}
    problems: Dict[str, str] = {}
    for name in config.embedders:
        if name not in EMBEDDERS:
            problems[name] = "not registered"
            continue
        t0 = time.time()
        try:
            got = run_embedders(candidates, embedder_names=[name], genes=list(genes),
                                dim=config.embedding_dim, params=config.embedder_params,
                                seed=config.seed, max_obs=config.max_obs_per_dataset)
            out.update(got)
            if verbose:
                for k, v in got.items():
                    nan = int(v.matrix.isna().all(axis=1).sum())
                    print(f"  {k:34s} {v.matrix.shape} nan_genes={nan}  {time.time()-t0:.1f}s")
        except Exception as exc:
            problems[name] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  FAILED {name:28s} {problems[name]}")
    if problems:
        p = Path(config.outdir) / "embedder_problems.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(problems, indent=1))
    return out


# ---------------------------------------------------------------------------
# stage 4: annotations
# ---------------------------------------------------------------------------

def annotate(genes: Sequence[str], config: Config) -> Optional[GeneAnnotation]:
    from .annotate import annotate_genes

    return annotate_genes(list(genes), sources=config.annotation_sources,
                          cache_dir=str(Path(config.data_dir) / "external"),
                          offline=config.offline)


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def run_pipeline(query: PanelQuery, config: Config, *,
                 candidates: Optional[Dict[str, CandidateDataset]] = None,
                 do_embeddings: bool = True, do_annotations: bool = True,
                 verbose: bool = True) -> dict:
    """Run every stage and write the artifacts under ``config.outdir``."""
    out = Path(config.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    result: dict = {"config": config.to_dict()}

    if verbose:
        print(f"[1a] descriptor match ({config.match_method})")
    result["match_table"] = match_candidates(query, config)
    result["match_table"].to_csv(out / "match_table.csv", index=False)

    if candidates is None:
        if verbose:
            print("[load] candidate datasets")
        candidates = load_candidates(config, verbose=verbose)
    result["candidates"] = candidates

    if verbose:
        print(f"[1b] agreement scores: {config.scores}")
    scores = score_candidates(query, candidates, config)
    scores.to_csv(out / "score_results.csv", index=False)
    result["score_results"] = scores

    figures: Dict[str, str] = {}
    try:
        from .viz import score_heatmap

        fp = out / "figures" / "score_heatmap.png"
        score_heatmap(scores, out_path=str(fp),
                      title=f"Panel-gene agreement with {query.name} "
                            f"(panel: {', '.join(query.panel)})")
        figures["Panel-gene agreement across candidate datasets"] = str(
            fp.relative_to(out))
    except Exception as exc:
        print(f"  score_heatmap unavailable: {type(exc).__name__}: {exc}")

    embeddings: Dict[str, GeneEmbedding] = {}
    annotation = None
    if do_embeddings:
        genes = embedding_gene_universe(candidates, config, panel=query.panel)
        result["gene_universe"] = genes
        if verbose:
            print(f"[2/3] gene embeddings over {len(genes)} genes "
                  f"({config.gene_universe} of expression candidates)")
        embeddings = embed_genes(candidates, genes, config, verbose=verbose)
        for name, emb in embeddings.items():
            emb.to_parquet(out / f"embedding_{name.replace('::', '__')}.parquet")
    result["embeddings"] = embeddings

    if do_annotations and embeddings:
        genes = result.get("gene_universe") or []
        if verbose:
            print(f"[4] annotations for {len(genes)} genes")
        try:
            annotation = annotate(genes, config)
            if annotation is not None:
                annotation.table.to_csv(out / "gene_annotation.csv")
        except Exception as exc:
            print(f"  annotation unavailable: {type(exc).__name__}: {exc}")
    result["annotation"] = annotation

    try:
        from .report import build_report

        rp = out / "comparison_report.md"
        build_report(config, match_table=result["match_table"], score_results=scores,
                     embeddings=embeddings, annotation=annotation, figures=figures,
                     out_path=str(rp))
        result["report_path"] = str(rp)
        if verbose:
            print(f"[report] {rp}")
    except Exception as exc:
        print(f"  report unavailable: {type(exc).__name__}: {exc}")

    result["figures"] = figures
    return result
