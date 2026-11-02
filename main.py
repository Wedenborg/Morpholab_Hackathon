"""Recommend genes for a Stereo-seq embryo over a few measurement rounds.

Each round recommends genes given what has been measured so far, then measures them
on every section of the embryo through the logged MeasurementOracle.
"""
import os
import argparse
import json

import anndata as ad
from scipy import sparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from eval_team import load_embryo, measure
from gene_recommender import recommend_genes, ridge_impute

# Multiples of 4 fill every experiment, because one experiment measures at most 4 genes.
ROUND_SIZES = (12, 8)
SEED = 42
SPATIAL_NEIGHBORS = 5
CELL_TYPE_WEIGHT = 0.1
# Stereo-seq embryo dataset from GEO. Set MORPHO_DATA_DIR to use another folder.
DATA_DIR = Path(os.environ.get("MORPHO_DATA_DIR", Path(__file__).resolve().parent / "data" / "GSE278603_RAW"))
# E7.5 rep1 is the target and rep2 the reference. Do not score against rep2 with this setup.
TARGET_REP = 1
REFERENCE_REP = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("outputs/ridge"))
    parser.add_argument("--round-sizes", type=int, nargs="+", default=list(ROUND_SIZES))
    parser.add_argument("--max-cells", type=int, help="Limit cells per embryo for a smoke run")
    parser.add_argument("--max-genes", type=int, help="Limit aligned genes for a smoke run")
    args = parser.parse_args()
    if any(k <= 0 for k in args.round_sizes) or any(v is not None and v <= 0 for v in (args.max_cells, args.max_genes)):
        parser.error("Round sizes and data limits must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output")
    if not any(args.data_dir.glob("*.h5ad")):
        raise SystemExit(f"No .h5ad files in {args.data_dir}. Download GSE278603_RAW or set MORPHO_DATA_DIR.")
    # Subset before densifying: full embryos can require several GB per matrix.
    hidden = load_embryo(TARGET_REP, data_dir=args.data_dir)
    reference = load_embryo(REFERENCE_REP, data_dir=args.data_dir)
    genes = hidden.var_names.intersection(reference.var_names, sort=False)
    if args.max_genes:
        genes = genes[:args.max_genes]
    hidden = hidden[:args.max_cells, genes].copy()
    reference = reference[:args.max_cells, genes].copy()
    if sum(args.round_sizes) > len(genes):
        parser.error("Requested panel exceeds the aligned gene count")
    from eval_team.submission import MeasurementOracle
    oracle, gene_names = MeasurementOracle(hidden), hidden.var_names
    locations = torch.tensor(hidden.obs[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
    gene_log_counts = torch.zeros(hidden.shape, dtype=torch.float32)
    sc_gene_log_counts = [torch.from_numpy(sparse.csr_matrix(reference.X).toarray().astype(np.float32))]
    reference_locations = [torch.tensor(reference.obs[["x", "y", "z"]].to_numpy(), dtype=torch.float32)]
    cell_types = reference.obs["celltype"].astype(str).to_numpy()
    reference_labels = [torch.from_numpy(pd.factorize(cell_types)[0].astype(np.int64))]

    # recommend_genes does not use these; they keep the shared method signature.
    gene_embeddings = torch.zeros(len(gene_names), 1)
    gene_similarity = torch.zeros(0)
    # Track measured genes explicitly, including genes whose expression is all zero.
    observed_mask = torch.zeros(len(gene_names), dtype=torch.bool)

    for round_index, n_genes in enumerate(args.round_sizes):
        start = time.perf_counter()
        result = recommend_genes(
            n_genes_per_round=n_genes,
            gene_embeddings=gene_embeddings,
            gene_similarity=gene_similarity,
            locations=locations,
            gene_log_counts=gene_log_counts,
            sc_gene_log_counts=sc_gene_log_counts,
            observed_mask=observed_mask,
            reference_locations=reference_locations,
            spatial_neighbors=SPATIAL_NEIGHBORS,
            reference_labels=reference_labels,
            label_weight=CELL_TYPE_WEIGHT,
            seed=SEED,
        )
        elapsed = time.perf_counter() - start
        next_gene_indices = result.selection_order
        gene_log_counts = measure(oracle, gene_names, next_gene_indices, gene_log_counts)
        observed_mask[next_gene_indices] = True

        print(f"Round {round_index} ({elapsed:.1f} s):")
        for gene, gain in zip(next_gene_indices.tolist(), result.marginal_gains.tolist()):
            print(f"  {gene_names[gene]:20s} {gain:.2%} of target variance")

    # result.imputed predates the last round's measurements, so predict again from all of them.
    imputed = ridge_impute(
        sc_gene_log_counts, gene_log_counts, observed_mask, locations,
        reference_locations=reference_locations, spatial_neighbors=SPATIAL_NEIGHBORS,
    )
    args.output.mkdir(parents=True)
    reconstruction = ad.AnnData(imputed.detach().cpu().numpy(), obs=hidden.obs.copy(), var=hidden.var.copy())
    reconstruction.write_h5ad(args.output / "reconstruction.h5ad")
    oracle.log().to_csv(args.output / "experiments.csv", index=False)
    mse = float(np.mean((reconstruction.X - sparse.csr_matrix(hidden.X).toarray()) ** 2))
    summary = {"method": "ridge", "shape": list(hidden.shape), "measured_genes": int(observed_mask.sum()),
               "experiments": len(oracle.log()), "reconstruction_mse": mse,
               "note": "MSE includes measured genes; this is an execution check, not the held-out benchmark."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Measured {int(observed_mask.sum())} genes in {len(oracle.log())} experiments.")
    print(f"Imputed expression shape: {tuple(imputed.shape)}")


if __name__ == "__main__":
    main()
