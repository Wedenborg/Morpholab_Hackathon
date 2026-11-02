"""Designs on the cell-type model: fixed vs predictive (adaptive), whole sections vs quarters.

Budget: one whole-section experiment per section, or four quarter-section experiments per
section (the same number of cell x gene measurements). Each run is scored with the posterior
mean (best for correlation) and with each cell's most likely state (keeps cells discrete).

    python -m active_design.run_step3 --hidden 2 --out step3_rep2.csv
"""
import argparse
import time
import warnings

import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.dummy_method import choose_gene_sets

from . import celltype_design as cd
from .celltype_model import CellTypeModel
from .run_step2 import as_adata


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h, v = args.hidden, 3 - args.hidden
    hidden, visible = load_embryo(h), load_embryo(v)
    genes, obs = hidden.var_names, hidden.obs
    ev = Evaluator(hidden)
    S = obs["z"].nunique()
    sets = choose_gene_sets(visible, genes, 8, "markers")
    ctm = CellTypeModel(40).fit(visible, genes)
    regions = {"sections": cd.section_regions(obs), "quarters": cd.quarter_regions(obs)}
    rows = []
    for rname, n_exp in (("sections", S), ("quarters", 4 * S)):
        for dname, run in (("rotating markers", lambda s, n: cd.rotating(s, sets, n)),
                           ("predictive", lambda s, n: cd.greedy_predictive(s, n))):
            t = time.time()
            sess = cd.StateSession(MeasurementOracle(hidden), ctm, obs, regions[rname])
            run(sess, n_exp)
            log = sess.oracle.log()
            for how in ("mean", "most likely state"):
                recon = as_adata(sess.reconstruct("mean" if how == "mean" else "map"), sess.measured, obs, genes)
                s, tb = ev.evaluate(recon, log)
                by = tb["heldout_by_sparsity"]["mean"]
                rows.append(dict(hidden=f"rep{h}", regions=rname, design=dname, output=how, n_experiments=len(log),
                                 n_genes=s["n_genes_measured"], heldout=s["heldout_corr"], heldout_dense=by.iloc[-1],
                                 heldout_sparse=by.iloc[0], peng=s["peng_spearman"], atlas=s["atlas_spearman"],
                                 truth_region=s["truth_region_spearman"], tv=s["celltype_tv"],
                                 bacc=s["celltype_balanced_accuracy"], run_s=time.time() - t))
                print(" | ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in rows[-1].items()), flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
