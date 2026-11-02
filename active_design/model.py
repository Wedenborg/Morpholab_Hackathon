"""Reconstruction model for experimental design.

Factor model (fit on the visible embryo)
    The embryo is cut into spatial neighbourhoods ("units", ~30 cells, within a section).
    A unit's mean log expression is  y_u = b + W z_u + noise,  noise variance r_g per gene,
    with z_u ~ K standardised factors (PCA on the visible embryo's unit means).
    Cells deviate from their unit: z_cell = z_u + delta, delta ~ N(0, Sigma_delta), and a
    cell's measured values have per-gene noise r_cell_g.

Spatial prior (hidden embryo, positions only)
    Units are linked to their nearest units in the same section (weight 1) and to the
    nearest unit in each adjacent section (weight `cross`). Factors are independent a priori,
    each with precision Q = c (I + lam * L), L the graph Laplacian, c chosen so the mean
    prior variance is 1.

Posterior
    Gaussian, kept as a mean vector and a dense covariance over (unit, factor) and updated
    exactly after every experiment (Kalman update). The reconstruction is b + W z for every
    cell, where z is the unit's posterior mean, corrected by the cell's own measurements,
    with measured values passed through unchanged.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from eval_team.region_benchmark import canonical_celltype
from eval_team.spatial_scores import neighbourhood_bins


def aligned_matrix(adata, genes):
    """csr (cells x len(genes)) in `genes` order; genes the AnnData lacks are zero columns."""
    cols = pd.Index(adata.var_names).get_indexer(genes)
    S = sp.csr_matrix((np.ones((cols >= 0).sum()), (cols[cols >= 0], np.where(cols >= 0)[0])),
                      shape=(adata.n_vars, len(genes)))
    return (sp.csr_matrix(adata.X) @ S).tocsr().astype(np.float32)


def unit_average_operator(units):
    n_units = units.max() + 1
    M = sp.csr_matrix((np.ones(len(units)), (units, np.arange(len(units)))), shape=(n_units, len(units)))
    return sp.diags(1 / np.asarray(M.sum(1)).ravel()) @ M


class FactorModel:
    def __init__(self, n_factors=16, cells_per_unit=30, seed=0):
        self.K = n_factors
        self.cells_per_unit = cells_per_unit
        self.seed = seed

    def fit(self, visible, genes, source=None):
        """Fit the model on the visible embryo.

        source=None   factors from PCA on the visible embryo's neighbourhood averages, and
                      every gene's weights fitted on them by least squares;
        source=X      factor directions from PCA on X (rows = profiles in `genes` order, e.g.
                      mdCLA cells), and their gene weights kept and only rescaled per gene to
                      the visible embryo, so genes that are sparse in Stereo-seq keep the
                      source's pattern.
        """
        self.genes = pd.Index(genes)
        X = aligned_matrix(visible, self.genes)
        units = neighbourhood_bins(visible.obs, self.cells_per_unit, seed=self.seed + 1)
        Y = np.asarray((unit_average_operator(units) @ X).todense())
        self.b = Y.mean(0)
        self.detect = np.asarray((X > 0).mean(0)).ravel()

        own = source is None
        pca = PCA(self.K, svd_solver="arpack" if sp.issparse(source) else "randomized",
                  random_state=self.seed).fit(Y if own else source)
        W_src = pca.components_.T                                         # genes x K, unit norm
        ok = self.detect >= 0.02
        if own:
            Z = pca.transform(Y)
        else:
            # Coordinates of the visible neighbourhoods on the source's factors, using genes
            # detected well enough in Stereo-seq to carry signal.
            Z = np.linalg.lstsq(W_src[ok], (Y - self.b)[:, ok].T, rcond=None)[0].T
        sd = Z.std(0)
        Z /= sd
        if own:
            self.W = np.linalg.lstsq(Z, Y - self.b, rcond=None)[0].T
        else:
            base = W_src * sd                                             # source pattern, Z scale
            pred = Z @ base.T                                             # units x genes
            num = ((Y - self.b) * pred).sum(0)
            den = (pred ** 2).sum(0)
            beta = num / np.where(den > 0, den, np.inf)
            good = ok & (den > 0)
            beta0 = np.median(beta[good])
            # shrink poorly measured genes toward the typical rescaling
            w = self.detect / (self.detect + 0.05)
            self.W = base * (w * beta + (1 - w) * beta0)[:, None]
        r = (Y - self.b - Z @ self.W.T).var(0)
        self.r = np.maximum(r, 0.01 * np.median(r[r > 0]))               # unit-level noise
        self._fit_cells(X, units, Z, visible)
        return self

    def _fit_cells(self, X, units, Z, visible):
        # Cell level: project each cell on the factors, measure its spread around its unit.
        P = self.W @ np.linalg.inv(self.W.T @ self.W)                     # genes x K
        Zc = np.asarray(X @ P) - self.b @ P
        delta = Zc - Z[units]
        self.Sigma_delta = np.cov(delta.T)
        r_cell = np.empty(len(self.genes))
        for s in range(0, len(self.genes), 2000):
            sl = slice(s, s + 2000)
            Xs = X[:, sl].toarray()
            r_cell[sl] = (Xs - self.b[sl] - Zc @ self.W[sl].T).var(0)
        self.r_cell = np.maximum(r_cell, 0.01 * np.median(r_cell[r_cell > 0]))

        # Cell-type structure in factor space (used by the adaptive design).
        ct = canonical_celltype(visible.obs["celltype"])
        dom = pd.Series(ct).groupby(units).agg(lambda s: s.value_counts().index[0]).to_numpy()
        keep = [c for c in pd.unique(dom) if (dom == c).sum() >= 3]
        self.type_names = keep
        self.type_means = np.vstack([Z[dom == c].mean(0) for c in keep])
        self.type_prior = np.array([(dom == c).mean() for c in keep])
        self.type_prior /= self.type_prior.sum()
        within = np.vstack([Z[dom == c] - Z[dom == c].mean(0) for c in keep])
        self.type_cov = np.cov(within.T)

    def informativeness(self):
        """How much one unit-level measurement of each gene tells about the factors."""
        return (self.W ** 2).sum(1) / self.r


def unit_precision(obs, units, lam=3.0, cross=0.5, k_within=6):
    """Prior precision (n_units x n_units) for one factor over the hidden embryo's units."""
    n_units = units.max() + 1
    cent = pd.DataFrame({"x": obs["x"].to_numpy(), "y": obs["y"].to_numpy(), "z": obs["z"].to_numpy(),
                         "u": units}).groupby("u")[["x", "y", "z"]].mean().sort_index()
    xy, z = cent[["x", "y"]].to_numpy(), cent["z"].to_numpy()
    sections = np.sort(np.unique(z))
    A = sp.lil_matrix((n_units, n_units))
    by_sec = [np.where(z == s)[0] for s in sections]
    for idx in by_sec:
        if len(idx) < 2:
            continue
        kk = min(k_within + 1, len(idx))
        _, nb = NearestNeighbors(n_neighbors=kk).fit(xy[idx]).kneighbors(xy[idx])
        for i, row in zip(idx, nb):
            for j in idx[row[1:]]:
                A[i, j] = A[j, i] = 1.0
    for a, b in zip(by_sec[:-1], by_sec[1:]):
        for src, dst in [(a, b), (b, a)]:
            _, nb = NearestNeighbors(n_neighbors=1).fit(xy[dst]).kneighbors(xy[src])
            for i, j in zip(src, dst[nb[:, 0]]):
                A[i, j] = max(A[i, j], cross)
                A[j, i] = A[i, j]
    A = A.tocsr()
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    Q = np.eye(n_units) + lam * L.toarray()
    c = np.diag(np.linalg.inv(Q)).mean()
    return Q * c


class UnitPosterior:
    """Gaussian posterior over all (unit, factor) values, updated exactly per experiment."""

    def __init__(self, Q, K):
        self.K = K
        self.n_units = Q.shape[0]
        self.Sigma = np.kron(np.linalg.inv(Q), np.eye(K))
        self.mean = np.zeros(self.n_units * K)

    def idx(self, units):
        return (np.asarray(units)[:, None] * self.K + np.arange(self.K)).ravel()

    def update(self, units, W_S, r_S, y=None):
        """Observe genes with loadings W_S (|S| x K), noise r_S, in `units`; y is
        (len(units) x |S|) unit means minus b (None to update the covariance only)."""
        units = np.asarray(units)
        idx = self.idx(units)
        H = np.kron(np.eye(len(units)), W_S)                    # (n|S|) x (nK), unit-major
        R = np.diag(np.tile(r_S, len(units)))
        SigH = self.Sigma[:, idx] @ H.T                         # D x m
        S = H @ SigH[idx] + R
        G = np.linalg.solve(S, SigH.T).T                        # D x m gain
        if y is not None:
            resid = np.asarray(y).ravel() - H @ self.mean[idx]
            self.mean += G @ resid
        self.Sigma -= G @ SigH.T
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

    def unit_mean(self):
        return self.mean.reshape(self.n_units, self.K)

    def unit_cov(self, u):
        i = self.idx([u])
        return self.Sigma[np.ix_(i, i)]
