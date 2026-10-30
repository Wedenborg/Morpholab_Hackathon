"""Compare experimental designs on the hidden embryo, scored by eval_team.

Budgets are counted in experiments (4 genes on one section). S = number of sections.
  1  gene-count curve: k marker genes on every section, k = 4..64 (k/4 * S experiments)
  2  rotating marker sets with the model; dummy method and random genes for comparison
  3  greedy information-gain design (planned in advance in effect)
  4  adaptive cell-type design
Designs 2-4 run at S and 2S experiments.

    python -m active_design.run_tests --hidden 2 --visible 1 --out results_rep2.csv
"""
import argparse
import time
import warnings

import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team import dummy_method
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import neighbourhood_bins

from . import designs
from .model import FactorModel, unit_precision


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--visible", type=int, default=1)
    ap.add_argument("--factors", type=int, default=16)
    ap.add_argument("--cells-per-unit", type=int, default=30)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--cross", type=float, default=0.25)
    ap.add_argument("--tests", default="1,2,3,4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tests = set(args.tests.split(","))

    hidden, visible = load_embryo(args.hidden), load_embryo(args.visible)
    library = hidden.var_names
    model = FactorModel(args.factors, args.cells_per_unit).fit(visible, library)
    units = neighbourhood_bins(hidden.obs, args.cells_per_unit, seed=1)
    Q = unit_precision(hidden.obs, units, lam=args.lam, cross=args.cross)
    ev = Evaluator(hidden)
    S = hidden.obs["z"].nunique()
    rows = []

    def record(test, design, recon, log, seconds):
        s, tables = ev.evaluate(recon, log)
        by = tables["heldout_by_sparsity"]["mean"]
        rows.append(dict(hidden=f"rep{args.hidden}", test=test, design=design, experiments=len(log),
                         genes=s["n_genes_measured"], heldout=s["heldout_corr"],
                         heldout_dense=by.iloc[-1], heldout_sparse=by.iloc[0],
                         peng=s["peng_spearman"], atlas=s["atlas_spearman"],
                         tv=s["celltype_tv"], bacc=s["celltype_balanced_accuracy"], seconds=seconds))
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    def run_model(test, design, fn):
        t = time.time()
        sess = designs.Session(MeasurementOracle(hidden), model, Q, units)
        fn(sess)
        record(test, design, sess.reconstruct(), sess.oracle.log(), time.time() - t)

    if "1" in tests:
        for k in [4, 8, 16, 32, 64]:
            genes = sum(choose_gene_sets(visible, library, k // 4, "markers"), [])
            run_model("1 gene count", f"{k} marker genes on every section", lambda s, g=genes: designs.fixed_panel(s, g))
    if "2" in tests:
        t = time.time()
        rec, log = dummy_method.run(MeasurementOracle(hidden), visible, n_sets=8, how="markers")
        record("2 rotation", "dummy method, 8 marker sets", rec, log, time.time() - t)
        one = choose_gene_sets(visible, library, 1, "markers")
        run_model("2 rotation", "model, 1 marker set (no rotation)", lambda s: designs.rotating(s, one, S))
        rand = choose_gene_sets(visible, library, 8, "random")
        run_model("2 rotation", "model, 8 random-gene sets", lambda s: designs.rotating(s, rand, S))
        for B in [S, 2 * S]:
            sets = choose_gene_sets(visible, library, 8 if B == S else 16, "markers")
            run_model("2 rotation", f"model, {len(sets)} marker sets", lambda s, ss=sets, b=B: designs.rotating(s, ss, b))
    if "3" in tests:
        for B in [S, 2 * S]:
            run_model("3 planned", "greedy information gain", lambda s, b=B: designs.greedy_information(s, b))
    if "4" in tests:
        for B in [S, 2 * S]:
            run_model("4 adaptive", "adaptive cell-type separation", lambda s, b=B: designs.adaptive_celltype(s, b))


if __name__ == "__main__":
    main()
