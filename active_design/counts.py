"""Count model for experimental design: negative binomial or zero-inflated negative binomial.

Stereo-seq X is log1p of SCT-corrected counts, so expm1(X) gives integer counts (sequencing
depth already normalised, so no per-cell library size is needed for Stereo-seq cells).

Model
    x_ig ~ NB(mu_ig, theta_g)  or  ZINB(mu_ig, theta_g, pi_g)
    log mu_ig = b_g + W_g . z_i,          z_i = Z_u(i) + delta_i,   delta_i ~ N(0, Sigma_delta)
    vec(Z) ~ N(0, Q^-1 (x) I_K)           same spatial prior over units as model.py

Factors (W) come from an NB factor model fitted on counts, either on the visible embryo or on
mdCLA (raw 10x counts, library size as offset). With mdCLA factors the visible embryo
calibrates the result to Stereo-seq:
    "rescale"  W_g = beta_g W_mdCLA,g with b_g, beta_g fitted on the visible embryo
               (beta shrunk toward the median for rarely detected genes)
    "refit"    every gene's weights refitted on the visible embryo, on cell factors
               computed with the mdCLA weights

Hidden embryo: the posterior over all units' Z uses a Laplace approximation (Fisher scoring
to the mode). Cells are observed at unit level: x_ig ~ (ZI)NB(exp(b_u,g + W_g . Z_u)), with
b_u, theta_u, pi_u fitted on the visible embryo with cells assigned their unit's factors, so
the cell-to-unit spread is absorbed into the dispersion. The Fisher information of a count
depends on its predicted mean, so unlike the Gaussian model, the posterior covariance (and
hence the information an experiment carries) depends on what was measured.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import gammaln
from sklearn.decomposition import PCA

import anndata as ad

from eval_team.spatial_scores import neighbourhood_bins

from .model import aligned_matrix, unit_average_operator

ETA_MIN, ETA_MAX = -30.0, 12.0


# ----------------------------------------------------------------------------- likelihood
def score_info(y, eta, theta, pi=None):
    """d loglik / d eta and the expected Fisher information for eta = log mu."""
    mu = np.exp(np.clip(eta, ETA_MIN, ETA_MAX))
    a = theta * mu / (theta + mu)
    s = theta * (y - mu) / (theta + mu)
    if pi is None:
        return s, a
    f0 = np.exp(theta * (np.log(theta) - np.log(theta + mu)))          # NB P(0)
    p0 = pi + (1 - pi) * f0
    s0 = -(1 - pi) * f0 * a / p0
    return np.where(y == 0, s0, s), p0 * s0 ** 2 + (1 - pi) * (a - f0 * a ** 2)


def loglik(y, eta, theta, pi=None):
    mu = np.exp(np.clip(eta, ETA_MIN, ETA_MAX))
    lt = np.log(theta / (theta + mu))
    ll = gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) + theta * lt + y * (np.log(mu) - np.log(theta + mu))
    if pi is None:
        return ll
    return np.where(y == 0, np.log(pi + (1 - pi) * np.exp(theta * lt)), np.log1p(-pi) + ll)


def _chunks(n, size):
    for s in range(0, n, size):
        yield slice(s, min(s + size, n))


# ----------------------------------------------------------------------------- fitting steps
def gene_step(Yc, F, B, theta, offset=None, pi=None, ridge=1.0, chunk=1000):
    """One Fisher-scoring step per gene for log mu_ig = offset_i + F_i . B_g.
    Yc: csc counts (cells x genes); F: cells x p (first column = 1, not penalised)."""
    n, p = F.shape
    off = np.zeros(n) if offset is None else offset
    FF = (F[:, :, None] * F[:, None, :]).reshape(n, p * p)
    R = np.full(p, ridge)
    R[0] = 1e-6
    B = B.copy()
    for gs in _chunks(Yc.shape[1], chunk):
        y = Yc[:, gs].toarray()
        eta = off[:, None] + F @ B[gs].T
        s, w = score_info(y, eta, theta[gs], None if pi is None else pi[gs])
        A = (FF.T @ w).T.reshape(-1, p, p) + np.diag(R)
        r = (F.T @ s).T - R * B[gs]
        B[gs] += np.clip(np.linalg.solve(A, r[..., None])[..., 0], -1, 1)
    return B


def cell_step(Yr, W, b, Z, theta, offset=None, pi=None, prior_mean=None, prior_prec=None, chunk=500):
    """One Fisher-scoring step per cell for its factors z_i (prior N(prior_mean, prior_prec^-1),
    default N(0, I))."""
    n, K = Z.shape
    G = W.shape[0]
    off = np.zeros(n) if offset is None else offset
    m = np.zeros((n, K)) if prior_mean is None else prior_mean
    P = np.eye(K) if prior_prec is None else prior_prec
    WW = (W[:, :, None] * W[:, None, :]).reshape(G, K * K)
    Z = Z.copy()
    for cs in _chunks(n, chunk):
        y = Yr[cs].toarray()
        eta = off[cs, None] + b + Z[cs] @ W.T
        s, w = score_info(y, eta, theta, pi)
        Pc = P if P.ndim == 2 else P[cs]
        A = (w @ WW).reshape(-1, K, K) + Pc
        r = s @ W - ((Z[cs] - m[cs])[:, None, :] @ Pc)[:, 0] if P.ndim == 3 else s @ W - (Z[cs] - m[cs]) @ Pc
        Z[cs] += np.clip(np.linalg.solve(A, r[..., None])[..., 0], -1, 1)
    return Z


def dispersion(Yc, eta_of, pi=None, chunk=1000):
    """Moment estimate of theta_g given the NB means exp(eta_of(gs)) (and pi_g if given)."""
    G = Yc.shape[1]
    theta = np.empty(G)
    for gs in _chunks(G, chunk):
        y = Yc[:, gs].toarray()
        mu = np.exp(np.clip(eta_of(gs), ETA_MIN, ETA_MAX))
        q = 0 if pi is None else pi[gs]
        m = (1 - q) * mu
        num = ((1 - q) * mu ** 2).sum(0)
        den = ((y - m) ** 2).sum(0) - ((1 - q) * mu * (1 + q * mu)).sum(0)
        theta[gs] = np.where(den > 0, num / np.maximum(den, 1e-12), 1e3)
    return np.clip(theta, 0.05, 1e3)


def zero_inflation(Yc, eta_of, theta, chunk=1000):
    """pi_g from the excess of zeros over what the NB predicts."""
    n, G = Yc.shape
    pi = np.empty(G)
    nz = np.diff(Yc.indptr)
    for gs in _chunks(G, chunk):
        mu = np.exp(np.clip(eta_of(gs), ETA_MIN, ETA_MAX))
        t = theta[gs]
        e0 = np.exp(t * (np.log(t) - np.log(t + mu))).sum(0)
        n0 = n - nz[gs]
        pi[gs] = (n0 - e0) / np.maximum(n - e0, 1e-9)
    return np.clip(pi, 0, 0.9)


def mean_shift(Yc, eta_of, pi=None, chunk=1000):
    """Per-gene shift of the intercept that matches total observed and predicted counts."""
    G = Yc.shape[1]
    tot = np.asarray(Yc.sum(0)).ravel()
    pred = np.empty(G)
    for gs in _chunks(G, chunk):
        pred[gs] = np.exp(np.clip(eta_of(gs), ETA_MIN, ETA_MAX)).sum(0)
    q = 0 if pi is None else pi
    return np.log((tot + 0.01) / ((1 - q) * pred + 0.01))


def glm_factors(Y, K, offset=None, n_iter=8, seed=0, verbose=True):
    """NB factor model log mu_ig = offset_i + b_g + W_g . z_i, fitted by alternating Fisher
    scoring (genes, then cells). Returns b, W (genes x K), theta, Z (cells x K, standardised)."""
    n, G = Y.shape
    off = np.zeros(n) if offset is None else offset
    Yc, Yr = Y.tocsc(), Y.tocsr()
    L = (sp.diags(np.exp(-off)) @ Yr).tocsr()
    L.data = np.log1p(L.data)
    Z = PCA(K, svd_solver="arpack", random_state=seed).fit_transform(L)
    Z /= Z.std(0)
    b = np.log((np.asarray(Yc.sum(0)).ravel() + 0.1) / np.exp(off).sum())
    B = np.hstack([b[:, None], np.zeros((G, K))])
    theta = np.full(G, 10.0)
    for it in range(n_iter):
        F = np.hstack([np.ones((n, 1)), Z])
        for _ in range(2):
            B = gene_step(Yc, F, B, theta, off)
        theta = dispersion(Yc, lambda gs: off[:, None] + F @ B[gs].T)
        Z_new = cell_step(Yr, B[:, 1:], B[:, 0], Z, theta, off)
        Z_new = cell_step(Yr, B[:, 1:], B[:, 0], Z_new, theta, off)
        m, sd = Z_new.mean(0), Z_new.std(0)
        B[:, 0] += B[:, 1:] @ m
        B[:, 1:] *= sd
        change = np.abs((Z_new - m) / sd - Z).mean()
        Z = (Z_new - m) / sd
        if verbose:
            print(f"    factor fit {it + 1}/{n_iter}: mean |dz| = {change:.3f}", flush=True)
    return B[:, 0], B[:, 1:], theta, Z


# ----------------------------------------------------------------------------- model
def counts_of(adata, genes):
    X = aligned_matrix(adata, genes)
    X.data = np.rint(np.expm1(X.data))
    X.eliminate_zeros()
    return X


class CountModel:
    def __init__(self, n_factors=16, cells_per_unit=30, seed=0):
        self.K = n_factors
        self.cells_per_unit = cells_per_unit
        self.seed = seed

    def fit(self, visible, genes, source=None, source_offset=None, loadings="rescale", n_iter=8,
            min_source_cells=20, verbose=True):
        """source: raw counts (csr cells x genes in `genes` order) giving the factors, with
        source_offset = log library size per cell. Default: the visible embryo's counts."""
        self.genes = pd.Index(genes)
        Y = counts_of(visible, self.genes)
        Yc = Y.tocsc()
        n, G, K = Y.shape[0], len(genes), self.K
        self.detect = np.diff(Yc.indptr) / n
        if source is None:
            b, W, theta, Z = glm_factors(Y, K, n_iter=n_iter, seed=self.seed, verbose=verbose)
        else:
            lib = source_offset
            _, Wm, _, _ = glm_factors(source, K, offset=lib, n_iter=n_iter, seed=self.seed, verbose=verbose)
            found = np.diff(source.tocsc().indptr) >= min_source_cells
            Wm[~found] = 0
            b = np.log((np.asarray(Yc.sum(0)).ravel() + 0.1) / n)
            W, theta = Wm.copy(), np.full(G, 10.0)
            Z = np.zeros((n, K))
            for it in range(3):
                for _ in range(3):
                    Z = cell_step(Y, W, b, Z, theta)
                F = np.hstack([np.ones((n, 1)), Z])
                if loadings == "refit":
                    B = np.hstack([b[:, None], W])
                    for _ in range(3):
                        B = gene_step(Yc, F, B, theta)
                    b, W = B[:, 0], B[:, 1:]
                elif loadings == "rescale":
                    b, beta = self._rescale(Yc, Z, Wm, b, theta)
                    # genes mdCLA does not see get their weights from the visible embryo
                    B = np.hstack([b[:, None], Wm * beta[:, None]])
                    if (~found).any():
                        B[~found] = gene_step(Yc[:, ~found], F, B[~found], theta[~found])
                    b, W = B[:, 0], B[:, 1:]
                else:
                    raise ValueError(loadings)
                theta = dispersion(Yc, lambda gs: b[gs] + Z @ W[gs].T)
                if verbose:
                    print(f"    calibration {it + 1}/3 on the visible embryo", flush=True)
        # standardise factors at unit level (the spatial prior has unit variance per factor)
        units = neighbourhood_bins(visible.obs, self.cells_per_unit, seed=self.seed + 1)
        Zu = np.asarray(unit_average_operator(units) @ Z)
        m, sd = Zu.mean(0), Zu.std(0)
        self.b = b + W @ m
        self.W = W * sd
        self.cell_Z = (Z - m) / sd
        self.unit_Z = (Zu - m) / sd
        self.Sigma_delta = np.cov((self.cell_Z - self.unit_Z[units]).T)
        self.theta_cell = theta
        self._Yc, self._units = Yc, units
        return self

    @staticmethod
    def _rescale(Yc, Z, Wm, b, theta, n_newton=4):
        """Fit (b_g, beta_g) for log mu = b_g + beta_g (W_m,g . z_i); shrink beta toward the
        median for rarely detected genes."""
        n, G = Yc.shape
        beta = np.ones(G)
        b = b.copy()
        for _ in range(n_newton):
            for gs in _chunks(G, 1000):
                y = Yc[:, gs].toarray()
                t = Z @ Wm[gs].T
                s, w = score_info(y, b[gs] + beta[gs] * t, theta[gs])
                A = np.stack([[w.sum(0) + 1e-6, (w * t).sum(0)], [(w * t).sum(0), (w * t * t).sum(0) + 1.0]])
                r = np.stack([s.sum(0), (s * t).sum(0) - (beta[gs] - 1)])
                d = np.linalg.solve(A.transpose(2, 0, 1), r.T[..., None])[..., 0]
                b[gs] += np.clip(d[:, 0], -1, 1)
                beta[gs] += np.clip(d[:, 1], -1, 1)
        detect = np.diff(Yc.indptr) / n
        ok = (detect >= 0.02) & (np.abs(Wm).sum(1) > 0)
        beta0 = np.median(beta[ok])
        w = detect / (detect + 0.05)
        beta = w * beta + (1 - w) * beta0
        # refit intercepts for the shrunk slopes
        b += mean_shift(Yc, lambda gs: b[gs] + beta[gs] * (Z @ Wm[gs].T))
        return b, beta

    def calibrate(self, zero_inflated=False, verbose=True):
        """Dispersion (and zero inflation) at cell level (given z_i) and at unit level (given
        Z_u, which also sets the unit-level intercepts b_u)."""
        Yc, W, b, units = self._Yc, self.W, self.b, self._units
        Zc, Zu = self.cell_Z, self.unit_Z[units]
        out = {}
        for level, Zl in (("cell", Zc), ("unit", Zu)):
            bl = b + mean_shift(Yc, lambda gs: b[gs] + Zl @ W[gs].T)
            theta = dispersion(Yc, lambda gs: bl[gs] + Zl @ W[gs].T)
            pi = None
            if zero_inflated:
                for _ in range(3):
                    pi = zero_inflation(Yc, lambda gs: bl[gs] + Zl @ W[gs].T, theta)
                    bl = bl + mean_shift(Yc, lambda gs: bl[gs] + Zl @ W[gs].T, pi)
                    theta = dispersion(Yc, lambda gs: bl[gs] + Zl @ W[gs].T, pi)
            out[level] = (bl, theta, pi)
        (self.b_cell, self.theta_cell, self.pi_cell), (self.b_unit, self.theta_unit, self.pi_unit) = out["cell"], out["unit"]
        self.zero_inflated = zero_inflated
        if verbose:
            ll = self.unit_loglik()
            msg = f"    unit-level log-lik per cell {ll:.1f}"
            if zero_inflated:
                msg += f"; genes with pi > 0.05: {(self.pi_unit > 0.05).mean():.1%} (cell level {(self.pi_cell > 0.05).mean():.1%})"
            print(msg, flush=True)
        return self

    def unit_loglik(self):
        Yc, Zu = self._Yc, self.unit_Z[self._units]
        tot = 0.0
        for gs in _chunks(Yc.shape[1], 1000):
            y = Yc[:, gs].toarray()
            tot += loglik(y, self.b_unit[gs] + Zu @ self.W[gs].T, self.theta_unit[gs],
                          None if self.pi_unit is None else self.pi_unit[gs]).sum()
        return tot / Yc.shape[0]

    def informativeness(self):
        """Fisher information of one cell's count about the factors, at the typical unit."""
        _, w = score_info(np.zeros(1), self.b_unit, self.theta_unit, self.pi_unit)
        return (self.W ** 2).sum(1) * w


# ----------------------------------------------------------------------------- posterior
class CountPosterior:
    """Laplace approximation to the posterior over all (unit, factor) values."""

    def __init__(self, Q, K):
        self.K, self.n_units = K, Q.shape[0]
        Qs = sp.csr_matrix(np.where(np.abs(Q) > 1e-12, Q, 0))
        self.Qk = sp.kron(Qs, sp.eye(K)).tocsc()
        self.mean = np.zeros(self.n_units * K)
        self.obs = []                                 # (unit of each cell, gene idx, counts)
        self.H = self.Qk
        self._Sigma = None

    def add(self, cell_units, gi, Y):
        self.obs.append((np.asarray(cell_units), np.asarray(gi), np.asarray(Y, float)))

    def _grad_info(self, m, Z):
        K = self.K
        g = np.zeros((self.n_units, K))
        H = np.zeros((self.n_units, K, K))
        for u, gi, Y in self.obs:
            Wg = m.W[gi]
            eta = m.b_unit[gi] + Z[u] @ Wg.T
            s, w = score_info(Y, eta, m.theta_unit[gi], None if m.pi_unit is None else m.pi_unit[gi])
            np.add.at(g, u, s @ Wg)
            np.add.at(H, u, np.einsum("cg,gk,gl->ckl", w, Wg, Wg))
        return g, H

    def fit(self, model, max_iter=30, tol=1e-4):
        K = self.K
        for _ in range(max_iter):
            Z = self.mean.reshape(self.n_units, K)
            g, Hb = self._grad_info(model, Z)
            H = (self.Qk + sp.block_diag(list(Hb), format="csc")).tocsc()
            grad = g.ravel() - self.Qk @ self.mean
            d = spla.spsolve(H, grad)
            step = np.max(np.abs(d))
            if step > 1.0:
                d /= step
            self.mean += d
            if step < tol:
                break
        g, Hb = self._grad_info(model, self.mean.reshape(self.n_units, K))
        self.H = (self.Qk + sp.block_diag(list(Hb), format="csc")).tocsc()
        self._Sigma = None
        return self

    @property
    def Sigma(self):
        if self._Sigma is None:
            self._Sigma = np.linalg.inv(self.H.toarray())
            self._Sigma = 0.5 * (self._Sigma + self._Sigma.T)
        return self._Sigma

    def idx(self, units):
        return (np.asarray(units)[:, None] * self.K + np.arange(self.K)).ravel()

    def unit_mean(self):
        return self.mean.reshape(self.n_units, self.K)

    def unit_cov(self, u):
        i = self.idx([u])
        return self.Sigma[np.ix_(i, i)]


class CountSession:
    """One design run on the hidden embryo with the count model (same interface as
    designs.Session, so designs.rotating works unchanged)."""

    def __init__(self, oracle, model, Q, units, verbose=False):
        self.oracle, self.model, self.units = oracle, model, units
        self.post = CountPosterior(Q, model.K)
        self.cell_index = pd.Index(oracle.cells.index)
        self.cell_z = oracle.cells["z"].to_numpy()
        self.sections = oracle.sections
        self.section_units = {s: np.unique(units[self.cell_z == s]) for s in self.sections}
        self.gene_pos = {g: i for i, g in enumerate(model.genes)}
        self.measured = {}
        self.n_experiments = 0
        self.verbose = verbose

    def run(self, section, genes):
        df = self.oracle.measure(list(genes), section=section)
        self.n_experiments += 1
        prev = self.measured.get(section)
        self.measured[section] = df if prev is None else prev.join(df[[g for g in df.columns if g not in prev]])
        rows = self.cell_index.get_indexer(df.index)
        gi = np.array([self.gene_pos[g] for g in df.columns])
        self.post.add(self.units[rows], gi, np.rint(np.expm1(df.to_numpy())))
        self.post.fit(self.model)
        if self.verbose:
            print(f"  exp {self.n_experiments:3d}: z={section:g} genes={list(genes)}")

    def reconstruct(self, n_newton=8):
        """Plug-in prediction log1p((1 - pi) exp(b + W z)) per cell, z = the unit's posterior
        mode, refined for measured cells by their own counts (Laplace, prior
        N(unit mode, unit covariance + Sigma_delta)). Measured values pass through."""
        m = self.model
        Z = self.post.unit_mean()[self.units].copy()
        for section, df in self.measured.items():
            rows = self.cell_index.get_indexer(df.index)
            gi = np.array([self.gene_pos[g] for g in df.columns])
            Y = np.rint(np.expm1(df.to_numpy()))
            u = self.units[rows]
            uu, inv = np.unique(u, return_inverse=True)
            P = np.stack([np.linalg.inv(self.post.unit_cov(x) + m.Sigma_delta) for x in uu])[inv]
            prior = Z[rows].copy()
            z = prior.copy()
            Wg = m.W[gi]
            pi = None if m.pi_cell is None else m.pi_cell[gi]
            for _ in range(n_newton):
                s, w = score_info(Y, m.b_cell[gi] + z @ Wg.T, m.theta_cell[gi], pi)
                A = np.einsum("cg,gk,gl->ckl", w, Wg, Wg) + P
                r = s @ Wg - np.einsum("ckl,cl->ck", P, z - prior)
                z += np.clip(np.linalg.solve(A, r[..., None])[..., 0], -1, 1)
            Z[rows] = z
        keep = 1 - (0 if m.pi_cell is None else m.pi_cell)
        X = np.log1p(keep * np.exp(np.clip(m.b_cell + Z @ m.W.T, ETA_MIN, ETA_MAX))).astype(np.float32)
        for section, df in self.measured.items():
            rows = self.cell_index.get_indexer(df.index)
            X[np.ix_(rows, [self.gene_pos[g] for g in df.columns])] = df.to_numpy()
        return ad.AnnData(X=X, obs=pd.DataFrame(index=self.cell_index), var=pd.DataFrame(index=m.genes))


# ----------------------------------------------------------------------------- adaptive design
def greedy_information(session, n_experiments, n_candidates=300, k=None):
    """Next experiment = section + genes with the largest expected information gain about the
    units' factors under the current Laplace posterior. A gene's Fisher information in unit u
    depends on its predicted mean there, so the choice depends on the measured values."""
    m, post = session.model, session.post
    k = k or session.oracle.k_max
    score = np.where(m.detect >= 0.05, m.informativeness(), 0)
    cand = np.argsort(-score)[:n_candidates]
    Wc = m.W[cand]
    for _ in range(n_experiments):
        Zm = post.unit_mean()
        best = (None, -np.inf, None)
        for s in session.sections:
            units = session.section_units[s]
            n_u = np.bincount(session.units[session.cell_z == s], minlength=post.n_units)[units]
            eta = m.b_unit[cand][None] + Zm[units] @ Wc.T                    # units x cand
            _, w = score_info(np.zeros_like(eta), eta, m.theta_unit[cand], None if m.pi_unit is None else m.pi_unit[cand])
            prec = n_u[:, None] * w                                         # units x cand
            genes, gain = _greedy_section(post, units, Wc, prec, k, _measured(session, s, cand))
            if gain > best[1]:
                best = ([m.genes[cand[g]] for g in genes], gain, s)
        session.run(best[2], best[0])


def _measured(session, section, cand):
    df = session.measured.get(section)
    if df is None:
        return []
    have = {session.gene_pos[g] for g in df.columns}
    return [i for i, c in enumerate(cand) if c in have]


def _greedy_section(post, units, Wc, prec, k, done):
    n, K = len(units), post.K
    idx = post.idx(units)
    Sig = post.Sigma[np.ix_(idx, idx)].copy()
    chosen, total, done = [], 0.0, list(done)
    for _ in range(k):
        T = Sig.reshape(n, K, n, K)
        C = np.einsum("gk,akbl,gl->gab", Wc, T, Wc, optimize=True)        # cand x n x n
        sq = np.sqrt(prec.T)                                               # cand x n
        M = np.eye(n)[None] + sq[:, :, None] * C * sq[:, None, :]
        gain = 0.5 * np.linalg.slogdet(M)[1]
        gain[done] = -np.inf
        g = int(np.argmax(gain))
        done.append(g)
        chosen.append(g)
        total += gain[g]
        H = np.kron(np.eye(n), Wc[g][None])                               # n x nK
        SH = Sig @ H.T
        Sig -= SH @ np.linalg.solve(H @ SH + np.diag(1 / prec[:, g]), SH.T)
    return chosen, total
