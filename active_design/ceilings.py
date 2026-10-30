"""How well can any reconstruction score on the held-out metric?

Noise ceiling
    The held-out score correlates reconstructed and true neighbourhood averages (about 25
    cells). A true average is signal plus sampling noise from its few cells. Split each
    neighbourhood's cells in two random halves; the per-gene correlation r of the two halves'
    averages across neighbourhoods gives the reliability of a full average,
    R = 2r / (1 + r) (Spearman-Brown). No prediction can correlate with the observed averages
    better than sqrt(R), on average over splits.

Information ceiling
    Suppose m genes were measured in every cell. Predict every other gene's neighbourhood
    average from the m measured averages, with a flexible regression:
      transfer   trained on the visible embryo, applied to the hidden one (what a method could do)
      cross-fit  trained on the hidden embryo's even sections, applied to its odd ones and back
                 (uses the hidden embryo, so only an upper reference)
    m genes = the first m of eval_team's round-robin marker list.

    python -m active_design.ceilings --hidden 2 --out ceilings_rep2.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from eval_team import load_embryo
from eval_team.dummy_method import choose_gene_sets
from eval_team.spatial_scores import SPARSITY_BINS, neighbourhood_bins

from .model import aligned_matrix, unit_average_operator


def pearson_cols(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A * B).sum(0) / np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))


def by_sparsity(values, detect):
    labels = ["2-5%", "5-10%", "10-25%", "25-100%"]
    s = pd.Series(values).groupby(pd.cut(detect, SPARSITY_BINS, right=False, labels=labels), observed=False).mean()
    return {f"detected {k}": v for k, v in s.items()} | {"all genes": np.nanmean(values)}


def noise_ceiling(X, bins, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        half = np.zeros(len(bins), bool)
        for b in np.unique(bins):
            idx = rng.permutation(np.where(bins == b)[0])
            half[idx[: len(idx) // 2]] = True
        A = np.asarray((unit_average_operator(bins[half]) @ X[half]).todense())
        B = np.asarray((unit_average_operator(bins[~half]) @ X[~half]).todense())
        rs.append(pearson_cols(A, B))
    r = np.nanmean(rs, 0)
    R = np.where(r > 0, 2 * r / (1 + r), 0)
    return np.sqrt(R)


def fit_predict(kind, F_train, T_train, F_test):
    sc = StandardScaler().fit(F_train)
    if kind == "ridge":
        model = RidgeCV(alphas=np.logspace(-2, 3, 12))
    else:
        model = KNeighborsRegressor(n_neighbors=15, weights="distance")
    return model.fit(sc.transform(F_train), T_train).predict(sc.transform(F_test))


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h, v = args.hidden, 3 - args.hidden
    hidden, visible = load_embryo(h), load_embryo(v)
    Xh = sp.csr_matrix(hidden.X)
    detect = np.asarray((Xh > 0).mean(0)).ravel()
    genes = hidden.var_names[detect >= 0.02]
    detect = detect[detect >= 0.02]
    Xh = aligned_matrix(hidden, genes)
    Xv = aligned_matrix(visible, genes)
    bh = neighbourhood_bins(hidden.obs, 25)                  # the evaluator's neighbourhoods
    bv = neighbourhood_bins(visible.obs, 25)
    Th = np.asarray((unit_average_operator(bh) @ Xh).todense())
    Tv = np.asarray((unit_average_operator(bv) @ Xv).todense())
    sec = hidden.obs["z"].to_numpy()
    sec_rank = pd.Series(sec).rank(method="dense").to_numpy().astype(int)
    bin_parity = pd.Series(sec_rank % 2).groupby(bh).first().sort_index().to_numpy()

    rows = []
    ceil = noise_ceiling(Xh, bh)
    rows.append(dict(hidden=f"rep{h}", what="noise ceiling", m=np.nan) | by_sparsity(ceil, detect))
    print(rows[-1], flush=True)

    markers = [g for s in choose_gene_sets(visible, genes, 64, "markers") for g in s]
    for m in (4, 16, 64, 256):
        M = pd.Index(genes).get_indexer(markers[:m])
        rest = np.setdiff1d(np.arange(len(genes)), M)
        for kind in ("ridge", "knn"):
            P = fit_predict(kind, Tv[:, M], Tv[:, rest], Th[:, M])
            r = pearson_cols(P, Th[:, rest])
            rows.append(dict(hidden=f"rep{h}", what=f"transfer {kind}", m=m) | by_sparsity(r, detect[rest]))
            print(rows[-1], flush=True)
            P = np.empty((Th.shape[0], len(rest)))
            for p in (0, 1):
                tr, te = bin_parity == p, bin_parity != p
                P[te] = fit_predict(kind, Th[tr][:, M], Th[tr][:, rest], Th[te][:, M])
            r = pearson_cols(P, Th[:, rest])
            rows.append(dict(hidden=f"rep{h}", what=f"cross-fit {kind}", m=m) | by_sparsity(r, detect[rest]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
