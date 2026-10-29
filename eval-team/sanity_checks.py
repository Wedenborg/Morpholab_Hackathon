"""Check that the region benchmark ranks reconstructions sensibly.

For each E7.5 embryo used as the hidden embryo, score reconstructions whose quality we
know in advance:

- true data (the reference point),
- true data with counts thinned to 50 / 20 / 5 % (shallower sequencing; should score lower
  and lower),
- true data averaged over each cell's 10 or 50 nearest neighbours in its section
  (reconstructions are usually smooth; this should cost little),
- cells shuffled (spatial information destroyed; should score near 0),
- genes shuffled (gene identities destroyed; should score near 0),
- the other E7.5 embryo's region averages copied onto this embryo's cells, using this
  embryo's region labels. No measurement of this embryo is used, so it shows how much a
  method could score by copying a reference instead of measuring,
- the dummy method (dummy_method.py) with marker, most-variable and random genes, going
  through the measurement oracle and submitting its experiment log.

Every case is scored with the full evaluator (evaluate.py). Cases without an experiment
log count every value as predicted.

Usage: python sanity_checks.py [--out results.csv]
"""
import argparse
import pathlib
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import dummy_method
from evaluate import Evaluator
from region_benchmark import assign_regions
from submission import MeasurementOracle

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "GSE278603_RAW"


def load(rep):
    return ad.read_h5ad(next(DATA.glob(f"*_Embryo_E7.5_stereo_rep{rep}.h5ad")))


def with_X(a, X):
    return ad.AnnData(X=X, obs=pd.DataFrame(index=a.obs_names), var=pd.DataFrame(index=a.var_names))


def thinned(a, p, seed=0):
    X = sp.csr_matrix(a.X).copy()
    counts = np.rint(np.expm1(X.data)).astype(np.int64)
    X.data = np.log1p(np.random.default_rng(seed).binomial(counts, p) / p)
    X.eliminate_zeros()
    return with_X(a, X)


def smoothed(a, k):
    X = sp.csr_matrix(a.X)
    z = a.obs["z"].to_numpy()
    xy = a.obs[["x", "y"]].to_numpy()
    rows, cols = [], []
    for zz in np.unique(z):
        idx = np.where(z == zz)[0]
        kk = min(k, len(idx))
        _, nb = NearestNeighbors(n_neighbors=kk).fit(xy[idx]).kneighbors(xy[idx])
        rows.append(np.repeat(idx, kk))
        cols.append(idx[nb].ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(a.n_obs, a.n_obs))
    W = sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W
    return with_X(a, W @ X)


def cells_shuffled(a, seed=0):
    perm = np.random.default_rng(seed).permutation(a.n_obs)
    return with_X(a, sp.csr_matrix(a.X)[perm])


def genes_shuffled(a, seed=0):
    perm = np.random.default_rng(seed).permutation(a.n_vars)
    return with_X(a, sp.csr_matrix(a.X)[:, perm])


def copied_from(a, other):
    """Each cell of `a` gets the mean expression of `other`'s cells in the same region
    (layer x sector). Genes `other` lacks are zero. Cells outside the regions get zeros."""
    ra = assign_regions(a.obs)["region"].str.rsplit("_", n=1).str[0]
    ro = assign_regions(other.obs)["region"].str.rsplit("_", n=1).str[0]
    Xo = sp.csr_matrix(other.X)
    genes = pd.Index(a.var_names)
    col = genes.get_indexer(other.var_names)
    out = np.zeros((a.n_obs, a.n_vars))
    for r in ro.dropna().unique():
        mo = (ro == r).to_numpy()
        ma = (ra == r).to_numpy()
        if ma.any():
            mean = np.asarray(Xo[mo].mean(0)).ravel()
            out[np.ix_(ma, col[col >= 0])] = mean[col >= 0]
    return with_X(a, out)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    emb = {1: load(1), 2: load(2)}
    rows = []
    for hidden, visible in [(2, 1), (1, 2)]:
        a = emb[hidden]
        ev = Evaluator(a)

        def dummy(n_sets, how):
            return dummy_method.run(MeasurementOracle(a), emb[visible], n_sets=n_sets, how=how)

        cases = {
            "true data": lambda: (a, None),
            "counts thinned to 50%": lambda: (thinned(a, 0.5), None),
            "counts thinned to 20%": lambda: (thinned(a, 0.2), None),
            "counts thinned to 5%": lambda: (thinned(a, 0.05), None),
            "10-neighbour average": lambda: (smoothed(a, 10), None),
            "50-neighbour average": lambda: (smoothed(a, 50), None),
            "cells shuffled": lambda: (cells_shuffled(a), None),
            "genes shuffled": lambda: (genes_shuffled(a), None),
            f"copy rep{visible} region averages": lambda: (copied_from(a, emb[visible]), None),
            "dummy: 1 set of 4 marker genes": lambda: dummy(1, "markers"),
            "dummy: 8 sets of 4 marker genes": lambda: dummy(8, "markers"),
            "dummy: 8 sets of 4 variable genes": lambda: dummy(8, "variance"),
            "dummy: 8 sets of 4 random genes": lambda: dummy(8, "random"),
        }
        for name, make in cases.items():
            recon, log = make()
            s, _ = ev.evaluate(recon, log)
            rows.append(dict(hidden=f"E7.5 rep{hidden}", reconstruction=name,
                             peng=s["peng_spearman"], atlas=s["atlas_spearman"],
                             truth_region=s["truth_region_spearman"], heldout=s["heldout_corr"],
                             celltype_tv=s["celltype_tv"], celltype_bacc=s["celltype_balanced_accuracy"]))
            print(f"rep{hidden} {name:34s} " + " ".join(f"{k}={v:+.3f}" for k, v in rows[-1].items()
                                                         if isinstance(v, float)), file=sys.stderr)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
