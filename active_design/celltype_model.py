"""Cell-type model: every cell is one of C discrete states with a spatial smoothness prior.

States come from clustering the visible embryo (k-means on 50 principal components of the
2,000 most variable genes); the transferred atlas labels are not used. Each state c has
    m_c      mean log1p expression (used for the reconstruction)
    mu_c     mean count (used for the likelihood), with a per-gene NB dispersion theta_g
             fitted on within-state variation
    pi_c     frequency in the visible embryo

Hidden embryo: a cell's measured counts c_iS give
    L_ic = sum_{g in S} log NB(c_ig; mu_cg, theta_g)
and neighbouring cells (6 nearest in the section, nearest 2 in each adjacent section) tend to
share states (Potts prior with strength beta). The posterior over states is approximated by
mean field:
    q_i ∝ prior_i * exp(L_i + beta * mean_{j in N(i)} q_j)
prior_i is pi (or a position-dependent prior from the registration). The reconstruction is the
posterior mean sum_c q_ic m_c.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from . import counts
from .model import aligned_matrix


class CellTypeModel:
    def __init__(self, n_states=40, seed=0):
        self.C, self.seed = n_states, seed

    def fit(self, visible, genes):
        self.genes = pd.Index(genes)
        X = aligned_matrix(visible, self.genes)
        m = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.multiply(X).mean(0)).ravel() - m ** 2
        hv = np.argsort(-var)[:2000]
        P = PCA(50, random_state=self.seed).fit_transform(X[:, hv].toarray())
        self.labels = KMeans(self.C, n_init=4, random_state=self.seed).fit_predict(P)
        onehot = sp.csr_matrix((np.ones(len(self.labels)), (self.labels, np.arange(len(self.labels)))),
                               shape=(self.C, len(self.labels)))
        n_c = np.asarray(onehot.sum(1)).ravel()
        avg = sp.diags(1 / n_c) @ onehot
        self.m = np.asarray((avg @ X).todense())                             # C x G, log1p
        Y = counts.counts_of(visible, self.genes)
        self.mu = np.asarray((avg @ Y).todense()) + 1e-3                     # C x G, counts
        lab = self.labels
        self.theta = counts.dispersion(Y.tocsc(), lambda gs: np.log(self.mu[:, gs][lab]))
        self.pi = n_c / n_c.sum()
        self.detect = np.diff(X.tocsc().indptr) / X.shape[0]
        self.gene_pos = {g: i for i, g in enumerate(self.genes)}
        return self

    def loglik(self, obs, measured):
        """n x C log-likelihood of each cell's measured counts under each state."""
        L = np.zeros((len(obs), self.C))
        cell_index = pd.Index(obs.index)
        for section, df in measured.items():
            rows = cell_index.get_indexer(df.index)
            gi = np.array([self.gene_pos[g] for g in df.columns])
            Y = np.rint(np.expm1(df.to_numpy()))
            eta = np.log(self.mu[:, gi])                                     # C x S
            L[rows] = counts.loglik(Y[:, None, :], eta[None], self.theta[gi]).sum(2)
        return L

    @staticmethod
    def graph(obs, k_within=6, k_across=2):
        """Row-normalised adjacency over cells: nearest cells in the section and in the
        adjacent sections."""
        z = obs["z"].to_numpy()
        xy = obs[["x", "y"]].to_numpy(float)
        secs = np.sort(np.unique(z))
        idx = {s: np.where(z == s)[0] for s in secs}
        rows, cols = [], []
        for a, s in enumerate(secs):
            here = idx[s]
            nn = NearestNeighbors(n_neighbors=min(k_within + 1, len(here))).fit(xy[here])
            nb = nn.kneighbors(xy[here], return_distance=False)[:, 1:]
            rows.append(np.repeat(here, nb.shape[1]))
            cols.append(here[nb.ravel()])
            for b in (a - 1, a + 1):
                if 0 <= b < len(secs):
                    other = idx[secs[b]]
                    nb = NearestNeighbors(n_neighbors=min(k_across, len(other))).fit(xy[other]).kneighbors(
                        xy[here], return_distance=False)
                    rows.append(np.repeat(here, nb.shape[1]))
                    cols.append(other[nb.ravel()])
        r, c = np.concatenate(rows), np.concatenate(cols)
        A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(obs), len(obs)))
        A = ((A + A.T) > 0).astype(float)
        return sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A

    def posterior(self, obs, measured, beta=2.0, log_prior=None, n_iter=30, A=None):
        L = self.loglik(obs, measured)
        lp = np.log(self.pi)[None] if log_prior is None else log_prior
        A = self.graph(obs) if A is None else A
        base = lp + L
        q = _softmax(base)
        for _ in range(n_iter):
            q = 0.5 * q + 0.5 * _softmax(base + beta * (A @ q))
        self.q = q
        return q

    def reconstruct(self, q):
        return (q @ self.m).astype(np.float32)

    def registration_prior(self, reg, k=50, alpha=5.0):
        """log p(state | position): states of the k visible cells nearest each hidden cell's
        registered position, with alpha pseudo-counts spread by pi."""
        _, nb = reg.tree.query(reg.pos_h, k=k)
        cnt = np.zeros((len(nb), self.C))
        np.add.at(cnt, (np.repeat(np.arange(len(nb)), k), self.labels[nb].ravel()), 1)
        return np.log((cnt + alpha * self.pi) / (k + alpha))


def _softmax(a):
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)
