"""Compare where the factor model's gene weights come from, at one experiment per section.

The Pijuan-Sala atlas is the benchmark's reference, so no variant uses it (or TOME / the
Extended Mouse Atlas, which contain it). mdCLA E7.5 is the independent single-cell source.

Variants (the visible embryo always calibrates to Stereo-seq; the hidden embryo is never used):
  0 visible embryo        factors and weights from the visible embryo's neighbourhood averages
  1 mdCLA E7.5, rescale   factor directions and gene weights from mdCLA cells, rescaled per gene
Designs: rotating marker sets and greedy information gain.

    python -m active_design.run_sources --hidden 2 --out sources_rep2.csv
"""
import argparse
import time
import warnings

import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import neighbourhood_bins

from . import designs, sources
from .model import FactorModel, unit_precision


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated variant numbers to run")
    args = ap.parse_args()
    h, v = args.hidden, 3 - args.hidden
    hidden, visible = load_embryo(h), load_embryo(v)
    genes = hidden.var_names
    units = neighbourhood_bins(hidden.obs, 30, seed=1)
    Q = unit_precision(hidden.obs, units, lam=3, cross=1.0)
    ev = Evaluator(hidden)
    S = hidden.obs["z"].nunique()
    sets = choose_gene_sets(visible, genes, 8, "markers")
    K = 16

    def fit(**kw):
        return FactorModel(K, 30).fit(visible, genes, **kw)

    variants = [
        ("visible embryo", lambda: fit()),
        ("mdCLA E7.5, rescale", lambda: fit(source=sources.mdcla_cells(genes)[0])),
    ]
    only = {int(x) for x in args.only.split(",") if x}
    rows = []
    for i, (name, build) in enumerate(variants):
        if only and i not in only:
            continue
        t = time.time()
        model = build()
        t_fit = time.time() - t
        for dname, run in [("rotating markers", lambda s: designs.rotating(s, sets, S)),
                           ("greedy information", lambda s: designs.greedy_information(s, S))]:
            t = time.time()
            sess = designs.Session(MeasurementOracle(hidden), model, Q, units)
            run(sess)
            s, tb = ev.evaluate(sess.reconstruct(), sess.oracle.log())
            by = tb["heldout_by_sparsity"]["mean"]
            rows.append(dict(hidden=f"rep{h}", variant=name, design=dname, heldout=s["heldout_corr"],
                             heldout_dense=by.iloc[-1], heldout_sparse=by.iloc[0], peng=s["peng_spearman"],
                             atlas=s["atlas_spearman"], tv=s["celltype_tv"], bacc=s["celltype_balanced_accuracy"],
                             fit_s=t_fit, run_s=time.time() - t))
            print(" | ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in rows[-1].items()), flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
