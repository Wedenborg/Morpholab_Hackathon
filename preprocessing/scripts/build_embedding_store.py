#!/usr/bin/env python
"""Build the expensive, gene-list-only embeddings once, into the store.

Why a separate script
---------------------
``esm2_sequence`` (ESM-2 over UniProt sequences), ``literature_text`` (BioGPT)
and ``ppi_network`` (node2vec over STRING) depend **only on the gene symbol
list**. They cost tens of minutes to hours, hit the network, and return the same
answer every time -- so they belong in a build step, not in a notebook.
``panelprep.run.run_analysis`` refuses to compute them (see
``panelprep.run.CACHED_ONLY``) and tells you to run this script instead.

Measured cost on this project's 4,006-symbol mouse universe, 8-core CPU, no GPU:

=====================  ==========  ==========  ====================================
source                 wall time   coverage    what it actually is
=====================  ==========  ==========  ====================================
coexpression_atlas         1.1 s   0.506       64 SVD factors of a cross-dataset
                                               gene-gene co-expression matrix
regulatory_kegg            0.1 s   0.351       KEGG pathway co-membership (see note)
ppi_network             2144   s   0.702       node2vec, STRING v12.0 score>=400
protein_sequence        2946   s   0.634       ESM-2 t12 35M, mean over residues
literature_text         2625   s   1.000       BioGPT, last-token pooling
=====================  ==========  ==========  ====================================

Note on ``regulatory_kegg``: the run that produced the shipped matrix fell back
to pathway **co-membership**, which is undirected and unsigned -- it is not the
signed regulatory graph the source is meant to provide. Rebuilding it costs
seconds; see ``--sources regulatory_kegg --recompute``.

Usage
-----
::

    # everything, into the default store
    python scripts/build_embedding_store.py --config configs/real_pilot.yaml

    # just one source, overwriting what is there
    python scripts/build_embedding_store.py --sources protein_sequence --recompute

    # adopt matrices computed elsewhere (no compute at all)
    python scripts/build_embedding_store.py --import-dir data/priors_export

The gene list comes from ``--genes-file`` (one symbol per line) if given,
otherwise from the configured candidate datasets' gene universe.

MODIFY HERE
-----------
* ``DEFAULT_SOURCES`` -- which prior sources to build by default.
* ``--dim`` -- feature width for the graph/co-expression sources (the language
  models return their own native width and ignore it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panelprep.config import Config                      # noqa: E402
from panelprep.embed import gene_universe                # noqa: E402
from panelprep.embed.store import EmbeddingStore         # noqa: E402
from panelprep.pipeline import load_candidates           # noqa: E402

#: Prior sources built by default. MODIFY HERE.
DEFAULT_SOURCES = ("coexpression_atlas", "protein_sequence", "ppi_network",
                   "literature_text", "regulatory_kegg")

#: Category per source, for the store sidecar.
CATEGORY = {
    "coexpression_atlas": "single_cell",
    "protein_sequence": "non_transcriptomic",
    "ppi_network": "non_transcriptomic",
    "literature_text": "non_transcriptomic",
    "regulatory_kegg": "causal",
    "perturbation_pseudobulk": "causal",
    "ops_morphology": "non_transcriptomic",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/real_pilot.yaml",
                   help="configuration file; supplies the candidates and outdir")
    p.add_argument("--store", default=None,
                   help="store root; defaults to <outdir>/store")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES),
                   help=f"prior sources to build (default: {' '.join(DEFAULT_SOURCES)})")
    p.add_argument("--genes-file", default=None,
                   help="one gene symbol per line; defaults to the dataset gene universe")
    p.add_argument("--dim", type=int, default=64,
                   help="feature width for graph/co-expression sources")
    p.add_argument("--recompute", action="store_true",
                   help="overwrite entries already in the store")
    p.add_argument("--offline", action="store_true",
                   help="use cached downloads only; a source with no cache is "
                        "reported blocked rather than attempted")
    p.add_argument("--import-dir", default=None,
                   help="adopt prior_<source>.parquet files from this directory "
                        "(expects prior_manifest.csv alongside); no compute")
    p.add_argument("--cache-dir", default="data/external/priors")
    return p.parse_args(argv)


def load_genes(args, config) -> list[str]:
    """Gene list to build over: explicit file, else the dataset gene universe."""
    if args.genes_file:
        genes = [ln.strip() for ln in Path(args.genes_file).read_text().splitlines()
                 if ln.strip()]
        print(f"[build] {len(genes)} genes from {args.genes_file}")
        return genes
    candidates = load_candidates(config, verbose=False)
    genes = gene_universe(candidates, how=config.gene_universe,
                          max_genes=config.max_genes,
                          rng=np.random.default_rng(config.seed))
    print(f"[build] {len(genes)} genes from the gene universe of "
          f"{len(candidates)} candidate datasets")
    return genes


def do_import(store: EmbeddingStore, import_dir: str) -> int:
    """Adopt already-computed prior matrices; nothing is recomputed."""
    d = Path(import_dir)
    manifest_path = d / "prior_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")
    manifest = pd.read_csv(manifest_path)
    n = 0
    for _, row in manifest.iterrows():
        source = str(row["source"])
        parquet = d / Path(str(row["parquet_file"])).name
        if not parquet.exists():
            print(f"  [skip] {source}: {parquet} missing")
            continue
        store.import_parquet(
            parquet, source, group="prior",
            category=CATEGORY.get(source, "non_transcriptomic"), method=source,
            provenance={
                "source": source,
                "model_or_db_version": row.get("model_or_db_version", ""),
                "status": row.get("status", "ok"),
                "n_genes": int(row.get("n_genes", 0)),
                "n_features": int(row.get("n_features", 0)),
                "coverage_fraction": float(row.get("coverage_fraction", float("nan"))),
                "compute_seconds": float(row.get("compute_seconds", float("nan"))),
                "provenance_note": "imported, not recomputed in this run",
            })
        print(f"  [import] {source}: {parquet.name}")
        n += 1
    return n


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    store = EmbeddingStore(args.store or (Path(config.out) / "store"))
    print(f"[build] store: {store.root}")

    if args.import_dir:
        n = do_import(store, args.import_dir)
        inv = store.inventory(write=True)
        print(f"[build] imported {n} matrices; store holds {len(inv)} embeddings")
        return 0

    from panelprep.priors import build_priors

    genes = load_genes(args, config)
    todo, already = [], []
    for source in args.sources:
        (already if (store.has(source) and not args.recompute) else todo).append(source)
    for source in already:
        print(f"  [have] {source} (use --recompute to rebuild)")
    if not todo:
        print("[build] nothing to do")
        store.inventory(write=True)
        return 0

    print(f"[build] building {len(todo)}: {', '.join(todo)}")
    t0 = time.time()
    priors = build_priors(genes, sources=todo, dim=args.dim,
                          cache_dir=args.cache_dir, offline=args.offline,
                          verbose=True)
    store.put_priors(priors)
    inv = store.inventory(write=True)
    report = {
        "built": list(priors),
        "n_genes": len(genes),
        "wall_seconds": round(time.time() - t0, 1),
        "coverage": {k: float(v.provenance.get("coverage", float("nan")))
                     for k, v in priors.items()},
        "status": {k: str(v.provenance.get("status", "")) for k, v in priors.items()},
    }
    (Path(store.root) / "build_log.json").write_text(json.dumps(report, indent=2))
    print(f"[build] done in {report['wall_seconds']}s; "
          f"store holds {len(inv)} embeddings")
    for k, v in report["coverage"].items():
        print(f"  {k}: coverage {v:.3f} ({report['status'][k]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
