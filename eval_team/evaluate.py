"""Evaluate a submission (reconstruction + experiment log) for one hidden E7.5 embryo.

Scores, all computed against data only the benchmark sees:

  region scores (region_benchmark.py)
    peng_spearman, atlas_spearman   plausibility: does each gene's pattern across regions /
                                    cell types match published E7.5 data?
    truth_region_spearman           the same region averages against the hidden embryo
  held-out gene score (spatial_scores.py)
    heldout_corr                    fidelity: neighbourhood-level correlation with the hidden
                                    embryo, only where the method did not measure the gene;
                                    also broken down by gene sparsity
  cell-type proportions (spatial_scores.py)
    celltype_tv, celltype_balanced_accuracy

Every score is also computed on the true data ("..._true_data") where that is meaningful.
The true data is a reference point, not an upper bound.

Usage:
    python -m eval_team.evaluate --truth <hidden .h5ad> --recon <recon .h5ad> --log <experiments .csv> \
                       [--out-dir results/]
"""
import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .region_benchmark import REF_DIR, RegionBenchmark, canonical_celltype
from .spatial_scores import (celltype_proportion_score, heldout_gene_score, neighbourhood_bins,
                            sparsity_summary)
from .submission import check_log, measured_mask


class Evaluator:
    def __init__(self, truth, ref_dir=REF_DIR, levels=False, cells_per_bin=25, min_detect=0.02,
                 n_proportion_genes=2000):
        self.truth = truth
        self.regions = RegionBenchmark(truth, ref_dir=ref_dir, levels=levels, min_detect=min_detect)
        self.bins = neighbourhood_bins(truth.obs, cells_per_bin)
        X = sp.csr_matrix(truth.X)
        detect = np.asarray((X > 0).mean(0)).ravel()
        self.genes = pd.Index(truth.var_names[detect >= min_detect])
        m = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.multiply(X).mean(0)).ravel() - m ** 2
        self.proportion_genes = truth.var_names[np.argsort(-var)[:n_proportion_genes]]
        self.labels = canonical_celltype(truth.obs["celltype"])
        self.proportions_true_data = celltype_proportion_score(truth, truth, self.labels, self.proportion_genes)

    def evaluate(self, recon, log=None):
        """recon: AnnData or DataFrame (cells x genes). log: experiment log DataFrame or None.
        Returns (summary dict, dict of tables)."""
        library = pd.Index(self.truth.var_names)
        recon_genes = pd.Index(recon.var_names if hasattr(recon, "var_names") else recon.columns)
        summary = {"library_genes_missing": int((~library.isin(recon_genes)).sum())}
        mask = None
        if log is not None:
            summary["log_problems"] = check_log(log)
            summary["n_experiments"] = len(log)
            measured_genes = set(g for gs in log["genes"].astype(str) for g in gs.split(";") if g)
            summary["n_genes_measured"] = len(measured_genes)
            mask = measured_mask(log, self.truth.obs, self.genes)

        reg, reg_genes = self.regions.score(recon)
        summary.update(reg)

        held = heldout_gene_score(recon, self.truth, self.bins, self.genes, mask=mask)
        summary["heldout_corr"] = float(held["corr"].mean())
        summary["heldout_n_genes"] = int(held["corr"].notna().sum())
        by_sparsity = sparsity_summary(held)
        for label, row in by_sparsity.iterrows():
            summary[f"heldout_corr | {label}"] = float(row["mean"])

        summary.update(celltype_proportion_score(recon, self.truth, self.labels, self.proportion_genes))
        for k, v in self.proportions_true_data.items():
            summary[f"{k}_true_data"] = v
        return summary, {"region_per_gene": reg_genes, "heldout_per_gene": held, "heldout_by_sparsity": by_sparsity}


def main():
    import anndata as ad
    from .submission import load_log
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--log", help="experiment log CSV; without it every value counts as predicted")
    ap.add_argument("--out-dir", help="write summary.json and per-gene tables here")
    ap.add_argument("--levels", action="store_true", help="split regions by proximal-distal level too")
    args = ap.parse_args()
    ev = Evaluator(ad.read_h5ad(args.truth), levels=args.levels)
    summary, tables = ev.evaluate(ad.read_h5ad(args.recon), load_log(args.log) if args.log else None)
    print(json.dumps(summary, indent=2))
    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        for name, t in tables.items():
            t.to_csv(out / f"{name}.csv")


if __name__ == "__main__":
    main()
