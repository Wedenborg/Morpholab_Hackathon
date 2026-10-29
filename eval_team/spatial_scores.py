"""Scores that compare a reconstruction with the hidden embryo's own measured data.

held-out gene score
    Each section is cut into small spatial neighbourhoods (about 25 cells, by k-means on
    x, y). For every gene, reconstructed and measured neighbourhood averages are correlated
    (Pearson) across the neighbourhoods where the method never measured that gene. This
    follows the held-out-gene evaluation of Tangram (Biancalani et al. 2021), but averages
    over neighbourhoods first, because single-cell values are mostly zeros.
    Results are also reported by gene sparsity (fraction of cells where the gene is
    detected), since sparse genes are hard to score even with a perfect reconstruction.

cell-type proportion score
    A fixed classifier (standardise, 50 principal components, logistic regression) is
    trained on the reconstruction with the true cell-type labels and tested on held-out
    sections: fold b tests every 5th section (index % 5 == b) and trains on the others,
    leaving out the sections directly next to a test section so that a reconstruction
    that borrows from neighbouring sections cannot leak labels across. Interleaving keeps
    every z level, and so every cell type, in the training data. Predicted and true cell-type proportions are compared by
    total variation distance; balanced accuracy is reported too, because errors in
    opposite directions cancel in proportions.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

SPARSITY_BINS = [0.02, 0.05, 0.10, 0.25, 1.01]


def neighbourhood_bins(obs, cells_per_bin=25, seed=0):
    """Integer neighbourhood id per cell; neighbourhoods never cross sections."""
    z = obs["z"].to_numpy()
    xy = obs[["x", "y"]].to_numpy(float)
    out = np.empty(len(obs), dtype=np.int64)
    next_id = 0
    for zz in np.unique(z):
        idx = np.where(z == zz)[0]
        k = max(1, len(idx) // cells_per_bin)
        lab = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(xy[idx]) if k > 1 else np.zeros(len(idx), int)
        out[idx] = lab + next_id
        next_id += k
    return out


def _bin_matrix(bins):
    n = bins.max() + 1
    M = sp.csr_matrix((np.ones(len(bins)), (bins, np.arange(len(bins)))), shape=(n, len(bins)))
    return sp.diags(1 / np.asarray(M.sum(1)).ravel()) @ M, M


def _columns(recon, cells, genes):
    """cells x genes matrix (sparse or dense) from an AnnData or DataFrame, rows in the
    order of `cells`; genes the reconstruction lacks are zero columns. Returns (X, n_missing)."""
    if hasattr(recon, "obs_names"):
        rows = pd.Index(recon.obs_names).get_indexer(cells)
        cols = pd.Index(recon.var_names).get_indexer(genes)
        X = recon.X
    else:
        rows = recon.index.get_indexer(cells)
        cols = recon.columns.get_indexer(genes)
        X = recon.to_numpy()
    if (rows < 0).any():
        raise ValueError(f"reconstruction is missing {(rows < 0).sum()} cells of the hidden embryo")
    present = cols >= 0
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X)
    # Selection matrix: picks the reconstruction's columns in `genes` order, zeros for absent genes.
    S = sp.csr_matrix((np.ones(present.sum()), (cols[present], np.where(present)[0])),
                      shape=(X.shape[1], len(genes)))
    return X[rows] @ S, int((~present).sum())


def heldout_gene_score(recon, truth, bins, genes, mask=None, min_bins=10, chunk=1000):
    """Per-gene Pearson correlation of neighbourhood averages over neighbourhoods where the
    gene was not measured. mask: cells x genes boolean (sparse), True = measured; None
    means nothing was measured. Returns a DataFrame indexed by gene with columns
    corr, n_bins, detect. Genes whose true neighbourhood averages are constant get NaN."""
    cells = truth.obs_names
    Mmean, Mcount = _bin_matrix(bins)
    Xt_all = sp.csr_matrix(truth.X)
    tcols = pd.Index(truth.var_names).get_indexer(genes)
    detect = np.asarray((Xt_all[:, tcols] > 0).mean(0)).ravel()
    out = []
    for s in range(0, len(genes), chunk):
        g = list(genes[s:s + chunk])
        Bt = np.asarray((Mmean @ Xt_all[:, tcols[s:s + chunk]]).todense())
        Xr, _ = _columns(recon, cells, g)
        Br = Mmean @ Xr
        Br = np.asarray(Br.todense()) if sp.issparse(Br) else np.asarray(Br)
        if mask is not None:
            measured = (Mcount @ sp.csr_matrix(mask[:, s:s + chunk]).astype(float)).toarray() > 0
            W = (~measured).astype(float)
        else:
            W = np.ones_like(Bt)
        n = W.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mt = (W * Bt).sum(0) / n
            mr = (W * Br).sum(0) / n
            dt, dr = W * (Bt - mt), W * (Br - mr)
            vt, vr = (dt ** 2).sum(0), (dr ** 2).sum(0)
            r = (dt * dr).sum(0) / np.sqrt(vt * vr)
        r = np.where(vr <= 1e-12, 0.0, r)                       # flat reconstruction scores 0
        r = np.where((vt <= 1e-12) | (n < min_bins), np.nan, r)  # nothing to score
        out.append(pd.DataFrame({"corr": r, "n_bins": n}, index=g))
    df = pd.concat(out)
    df["detect"] = detect
    return df


def sparsity_summary(per_gene, col="corr"):
    """Mean score per gene-sparsity bin (fraction of cells where the gene is detected)."""
    d = per_gene.dropna(subset=[col])
    labels = [f"detected in {lo:.0%}-{min(hi, 1):.0%} of cells" for lo, hi in zip(SPARSITY_BINS, SPARSITY_BINS[1:])]
    cut = pd.cut(d["detect"], SPARSITY_BINS, right=False, labels=labels)
    g = d.groupby(cut, observed=False)[col]
    return pd.DataFrame({"mean": g.mean(), "n_genes": g.size()})


def celltype_proportion_score(recon, truth, labels, genes, n_folds=5, n_pcs=50, seed=0):
    """Train a fixed classifier on the reconstruction (interleaved sections held out) and
    compare predicted with true cell-type proportions. Returns a dict."""
    z = truth.obs["z"].to_numpy()
    X, _ = _columns(recon, truth.obs_names, list(genes))
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    sd = X.std(0)
    X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    Z = PCA(n_components=min(n_pcs, X.shape[1] - 1), random_state=seed).fit_transform(X)
    Z /= np.where(Z.std(0) > 0, Z.std(0), 1)  # unit-scale components so the fit converges
    sections = np.sort(np.unique(z))
    sec_index = pd.Series(np.arange(len(sections)), index=sections)[z].to_numpy()
    pred = np.empty(len(labels), dtype=object)
    for b in range(n_folds):
        test_secs = np.arange(b, len(sections), n_folds)
        excluded = np.concatenate([test_secs - 1, test_secs, test_secs + 1])
        test = np.isin(sec_index, test_secs)
        train = ~np.isin(sec_index, excluded)
        clf = LogisticRegression(max_iter=1000).fit(Z[train], labels[train])
        pred[test] = clf.predict(Z[test])
    p = pd.Series(labels).value_counts(normalize=True)
    q = pd.Series(pred).value_counts(normalize=True)
    return dict(celltype_tv=float(0.5 * p.subtract(q, fill_value=0).abs().sum()),
                celltype_balanced_accuracy=float(balanced_accuracy_score(labels, pred)))
