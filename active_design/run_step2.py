"""Registration and cell-type models vs the Gaussian factor model, on the same measurements.

Each design is run once with the Gaussian model (rotating markers; greedy information gain,
which does not depend on measured values), then every model reconstructs from the same
measurements:
  gaussian                      factor model, model.py
  registration, position only   copy of the visible embryo after alignment
  registration + genes          visible cells chosen by position and measured genes
  cell states, beta=0/2/4       cell-type model, flat prior, Potts strength beta
  cell states + registration    cell-type model with the registration's position prior, beta=2

    python -m active_design.run_step2 --hidden 2 --out step2_rep2.csv
"""
import argparse
import time
import warnings

import anndata as ad
import pandas as pd

from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import neighbourhood_bins

from . import designs
from .celltype_model import CellTypeModel
from .model import FactorModel, unit_precision
from .registration import Registration


def as_adata(X, measured, obs, genes):
    """Dense reconstruction with measured values passed through."""
    cell_index, gpos = pd.Index(obs.index), {g: i for i, g in enumerate(genes)}
    for _, df in measured.items():
        rows = cell_index.get_indexer(df.index)
        for j, g in enumerate(df.columns):
            X[rows, gpos[g]] = df.to_numpy()[:, j]
    return ad.AnnData(X=X, obs=pd.DataFrame(index=cell_index), var=pd.DataFrame(index=genes))


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h, v = args.hidden, 3 - args.hidden
    hidden, visible = load_embryo(h), load_embryo(v)
    genes = hidden.var_names
    obs = hidden.obs
    units = neighbourhood_bins(obs, 30, seed=1)
    Q = unit_precision(obs, units, lam=3, cross=1.0)
    ev = Evaluator(hidden)
    S = obs["z"].nunique()
    sets = choose_gene_sets(visible, genes, 8, "markers")
    fm = FactorModel(16, 30).fit(visible, genes)
    t = time.time()
    ctm = CellTypeModel(40).fit(visible, genes)
    A = CellTypeModel.graph(obs)
    print(f"cell-state model fitted in {time.time() - t:.0f}s", flush=True)
    reg0 = Registration(visible, genes)

    rows = []

    def score(design, name, recon, log, extra=None):
        s, tb = ev.evaluate(recon, log)
        by = tb["heldout_by_sparsity"]["mean"]
        rows.append(dict(hidden=f"rep{h}", design=design, model=name, heldout=s["heldout_corr"],
                         heldout_dense=by.iloc[-1], heldout_sparse=by.iloc[0], peng=s["peng_spearman"],
                         atlas=s["atlas_spearman"], truth_region=s["truth_region_spearman"],
                         tv=s["celltype_tv"], bacc=s["celltype_balanced_accuracy"]) | (extra or {}))
        print(" | ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in rows[-1].items()), flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    for design, run in [("rotating markers", lambda s: designs.rotating(s, sets, S)),
                        ("greedy information", lambda s: designs.greedy_information(s, S))]:
        sess = designs.Session(MeasurementOracle(hidden), fm, Q, units)
        run(sess)
        log, measured = sess.oracle.log(), sess.measured
        score(design, "gaussian", sess.reconstruct(), log)

        reg = reg0.fit(obs, measured)
        extra = dict(reg_angle=reg.transform[0], reg_mirror=reg.transform[1], reg_fit=reg.fit_score)
        # diagnostic only (uses the hidden labels, never used for fitting): how often a hidden
        # cell's registered position lands among visible cells of the same annotated type
        nb = reg.neighbours(obs, measured, k=30, use_genes=False)
        lab_h = obs["celltype"].astype(str).to_numpy()
        lab_v = visible.obs["celltype"].astype(str).to_numpy()
        extra["reg_type_agreement"] = float((lab_v[nb.indices.reshape(len(obs), -1)] == lab_h[:, None]).mean())
        score(design, "registration, position only", as_adata(reg.predict(obs, measured, use_genes=False), measured, obs, genes), log, extra)
        score(design, "registration + genes", as_adata(reg.predict(obs, measured, use_genes=True), measured, obs, genes), log, extra)

        for beta in (0.0, 2.0, 4.0):
            q = ctm.posterior(obs, measured, beta=beta, A=A)
            score(design, f"cell states, beta={beta:g}", as_adata(ctm.reconstruct(q), measured, obs, genes), log)
        q = ctm.posterior(obs, measured, beta=2.0, A=A, log_prior=ctm.registration_prior(reg))
        score(design, "cell states + registration, beta=2", as_adata(ctm.reconstruct(q), measured, obs, genes), log)


if __name__ == "__main__":
    main()
