"""Count model (NB / ZINB) vs the Gaussian model, at one experiment per section.

Variants (factors from an NB factor model on counts):
  visible NB / ZINB            factors from the visible embryo
  mdCLA rescale NB / ZINB      mdCLA weights kept, rescaled per gene to Stereo-seq
  mdCLA refit NB / ZINB        mdCLA factors, every gene's weights refitted on the visible embryo
Designs: rotating marker sets; --greedy adds the adaptive greedy information design.

    python -m active_design.run_counts --hidden 2 --out counts_rep2.csv
"""
import argparse
import copy
import time
import warnings

import numpy as np
import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import neighbourhood_bins

from . import counts, designs, sources
from .model import unit_precision


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated factor sources: visible,rescale,refit")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--n-experiments", type=int, default=0, help="default: one per section")
    ap.add_argument("--n-iter", type=int, default=8)
    args = ap.parse_args()
    h, v = args.hidden, 3 - args.hidden
    hidden, visible = load_embryo(h), load_embryo(v)
    genes = hidden.var_names
    units = neighbourhood_bins(hidden.obs, 30, seed=1)
    Q = unit_precision(hidden.obs, units, lam=3, cross=1.0)
    ev = Evaluator(hidden)
    S = args.n_experiments or hidden.obs["z"].nunique()
    sets = choose_gene_sets(visible, genes, 8, "markers")

    only = [x for x in args.only.split(",") if x] or ["visible", "rescale", "refit"]
    md = None
    rows = []
    for src in only:
        t = time.time()
        print(f"fitting {src}", flush=True)
        if src == "visible":
            base = counts.CountModel(16, 30).fit(visible, genes, n_iter=args.n_iter)
        else:
            if md is None:
                Xm, lib = sources.mdcla_counts(genes)
                md = (Xm, np.log(lib / np.median(lib)))
            base = counts.CountModel(16, 30).fit(visible, genes, source=md[0], source_offset=md[1],
                                                 loadings=src, n_iter=args.n_iter)
        t_fit = time.time() - t
        for zi in (False, True):
            model = copy.copy(base).calibrate(zero_inflated=zi)
            name = f"{'mdCLA ' + src if src != 'visible' else 'visible'} {'ZINB' if zi else 'NB'}"
            runs = [("rotating markers", lambda s: designs.rotating(s, sets, S))]
            if args.greedy:
                runs.append(("greedy information", lambda s: counts.greedy_information(s, S)))
            for dname, run in runs:
                t = time.time()
                sess = counts.CountSession(MeasurementOracle(hidden), model, Q, units)
                run(sess)
                s, tb = ev.evaluate(sess.reconstruct(), sess.oracle.log())
                by = tb["heldout_by_sparsity"]["mean"]
                rows.append(dict(hidden=f"rep{h}", variant=name, design=dname, heldout=s["heldout_corr"],
                                 heldout_dense=by.iloc[-1], heldout_sparse=by.iloc[0], peng=s["peng_spearman"],
                                 atlas=s["atlas_spearman"], tv=s["celltype_tv"], unit_loglik=model.unit_loglik(),
                                 fit_s=t_fit, run_s=time.time() - t))
                print(" | ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in rows[-1].items()), flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
