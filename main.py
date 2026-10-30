"""Recommend genes for a Stereo-seq embryo over a few measurement rounds.

Each round recommends genes given what has been measured so far, then measures them
on every section of the embryo through the logged MeasurementOracle.
"""
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from eval_team import load_e75_inputs, load_embryo, measure
from gene_recommender import recommend_genes, ridge_impute
from mouse_embryo import get_locations

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
    if not any(DATA_DIR.glob("*.h5ad")):
        raise SystemExit(f"No .h5ad files in {DATA_DIR}. Download GSE278603_RAW or set MORPHO_DATA_DIR.")
    inputs = load_e75_inputs(hidden_rep=TARGET_REP, visible_rep=REFERENCE_REP, data_dir=DATA_DIR)
    oracle, gene_names, locations = inputs["oracle"], inputs["genes"], inputs["locations"]
    gene_log_counts = inputs["gene_log_counts"]
    sc_gene_log_counts = inputs["sc_gene_log_counts"]
    reference = load_embryo(REFERENCE_REP, data_dir=DATA_DIR)
    reference_locations = [get_locations(reference)]
    cell_types = reference.obs["celltype"].astype(str).to_numpy()
    reference_labels = [torch.from_numpy(pd.factorize(cell_types)[0].astype(np.int64))]

    # recommend_genes does not use these; they keep the shared method signature.
    gene_embeddings = torch.zeros(len(gene_names), 1)
    gene_similarity = torch.zeros(0)
    # Track measured genes explicitly, including genes whose expression is all zero.
    observed_mask = torch.zeros(len(gene_names), dtype=torch.bool)

    for round_index, n_genes in enumerate(ROUND_SIZES):
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
    print(f"Measured {int(observed_mask.sum())} genes in {len(oracle.log())} experiments.")
    print(f"Imputed expression shape: {tuple(imputed.shape)}")


if __name__ == "__main__":
    main()
