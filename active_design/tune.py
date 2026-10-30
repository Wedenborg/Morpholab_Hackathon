"""Tune the model's settings on the visible embryo only (it plays the hidden embryo),
so the real hidden embryo never influences them. Design used: rotating marker sets,
one experiment per section. Writes a CSV of scores per setting.

    python -m active_design.tune --visible 1 --out tune_rep1.csv
"""
import argparse
import itertools
import time
import warnings

import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import neighbourhood_bins

from . import designs
from .model import FactorModel, unit_precision


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--visible", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb = load_embryo(args.visible)
    ev = Evaluator(emb)
    sets = choose_gene_sets(emb, emb.var_names, 8, "markers")
    rows = []
    for K, cpu in [(8, 30), (16, 30), (32, 30), (16, 15)]:
        model = FactorModel(K, cpu).fit(emb, emb.var_names)
        units = neighbourhood_bins(emb.obs, cpu, seed=1)
        for lam, cross in itertools.product([0.3, 3, 30], [0.25, 1.0]):
            t = time.time()
            Q = unit_precision(emb.obs, units, lam=lam, cross=cross)
            sess = designs.Session(MeasurementOracle(emb), model, Q, units)
            designs.rotating(sess, sets, len(sess.sections))
            s, _ = ev.evaluate(sess.reconstruct(), sess.oracle.log())
            rows.append(dict(K=K, cells_per_unit=cpu, lam=lam, cross=cross, heldout=s["heldout_corr"],
                             peng=s["peng_spearman"], atlas=s["atlas_spearman"], tv=s["celltype_tv"],
                             bacc=s["celltype_balanced_accuracy"]))
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()),
                  f"({time.time() - t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
