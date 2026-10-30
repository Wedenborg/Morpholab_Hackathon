"""Reconstruction by registering the hidden embryo onto the visible one.

Both embryos are in the same units (um, 10 um sections, x/y centred), with z running
proximal-distal. The hidden embryo's cells are mapped into the visible embryo's frame by
    x/y: rotation by angle a, optional mirror, shift (dx, dy);
    z:   scaled so the two embryos have the same extent.
The transform is chosen by how well the visible cells at the mapped positions predict the
values measured so far (grid search, then a finer grid around the best).

A hidden cell is then predicted as the average of k visible cells chosen by
    d = |pos_i - pos_j|^2 / s_pos^2 + mean_g (x_ig - x_jg)^2 / s_x^2
over the cell's measured genes g (standardised by the visible embryo's spread), among its
150 nearest visible cells by position. With no measured genes, or `use_genes=False`, the
choice is by position only (copying the aligned visible embryo).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .model import aligned_matrix


def _rotate(xy, angle, mirror):
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if mirror:
        xy = xy * np.array([1, -1])
    return xy @ R.T


class Registration:
    def __init__(self, visible, genes):
        self.genes = pd.Index(genes)
        self.Xv = aligned_matrix(visible, self.genes)
        o = visible.obs
        self.pos_v = np.c_[o["x"].to_numpy(float) - o["x"].mean(), o["y"].to_numpy(float) - o["y"].mean(), o["z"].to_numpy(float)]
        self.tree = cKDTree(self.pos_v)
        m = np.asarray(self.Xv.mean(0)).ravel()
        self.sd_v = np.sqrt(np.maximum(np.asarray(self.Xv.multiply(self.Xv).mean(0)).ravel() - m ** 2, 1e-6))
        self.gene_pos = {g: i for i, g in enumerate(self.genes)}

    def _hidden_pos(self, obs, angle, mirror, dx, dy):
        xy = np.c_[obs["x"].to_numpy(float) - obs["x"].mean(), obs["y"].to_numpy(float) - obs["y"].mean()]
        z = obs["z"].to_numpy(float)
        z = (z - z.min()) / max(np.ptp(z), 1) * np.ptp(self.pos_v[:, 2]) + self.pos_v[:, 2].min()
        return np.c_[_rotate(xy, angle, mirror) + [dx, dy], z]

    def _fit_score(self, pos, rows, cols, vals, k=20):
        _, nb = self.tree.query(pos[rows], k=k)
        pred = self._V[nb][:, :, cols].mean(1)
        # correlation per (gene) over the sampled measured cells
        p, v = pred - pred.mean(0), vals - vals.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (p * v).sum(0) / np.sqrt((p ** 2).sum(0) * (v ** 2).sum(0))
        return np.nanmean(r)

    def fit(self, obs, measured, max_cells=600, seed=0, verbose=True):
        """measured: {section: DataFrame (cells x genes, log1p)}. Score each candidate transform
        by the mean correlation, per measured (section, gene), of predicted and measured
        values on a sample of measured cells."""
        rng = np.random.default_rng(seed)
        cell_index = pd.Index(obs.index)
        union = sorted({self.gene_pos[g] for df in measured.values() for g in df.columns})
        self._V = np.asarray(self.Xv[:, union].todense())
        col = {g: i for i, g in enumerate(union)}
        samples = []
        for section, df in measured.items():
            take = rng.choice(len(df), min(max_cells, len(df)), replace=False)
            samples.append((cell_index.get_indexer(df.index[take]),
                            np.array([col[self.gene_pos[g]] for g in df.columns]), df.to_numpy()[take]))

        def score(t):
            pos = self._hidden_pos(obs, *t)
            return np.mean([self._fit_score(pos, r, g, v) for r, g, v in samples])

        grid = [(a, m, 0.0, 0.0) for m in (False, True) for a in range(0, 360, 15)]
        scores = [score(t) for t in grid]
        best = grid[int(np.argmax(scores))]
        fine = [(best[0] + da, best[1], dx, dy) for da in range(-12, 13, 4)
                for dx in (-40, -20, 0, 20, 40) for dy in (-40, -20, 0, 20, 40)]
        fscores = [score(t) for t in fine]
        self.transform = fine[int(np.argmax(fscores))]
        self.fit_score = float(np.max(fscores))
        self.coarse_scores = pd.Series(scores, index=[f"{a}{'m' if m else ''}" for a, m, _, _ in grid])
        self.pos_h = self._hidden_pos(obs, *self.transform)
        if verbose:
            a, m, dx, dy = self.transform
            print(f"    registration: angle {a}, mirror {m}, shift ({dx}, {dy}); fit score {self.fit_score:.3f}", flush=True)
        return self

    def neighbours(self, obs, measured, k=30, n_candidates=150, s_pos=30.0, s_x=1.0, use_genes=True):
        """Row-normalised sparse matrix (hidden cells x visible cells) of the chosen neighbours."""
        n = len(obs)
        dpos, cand = self.tree.query(self.pos_h, k=n_candidates)
        cost = dpos ** 2 / s_pos ** 2
        if use_genes:
            cell_index = pd.Index(obs.index)
            for section, df in measured.items():
                rows = cell_index.get_indexer(df.index)
                gi = np.array([self.gene_pos[g] for g in df.columns])
                Vv = np.asarray(self.Xv[:, gi].todense()) / self.sd_v[gi]
                Vh = df.to_numpy() / self.sd_v[gi]
                diff = Vv[cand[rows]] - Vh[:, None, :]                       # cells x cand x genes
                cost[rows] += (diff ** 2).mean(2) / s_x ** 2
        pick = np.argsort(cost, 1)[:, :k]
        cols = np.take_along_axis(cand, pick, 1)
        return sp.csr_matrix((np.full(n * k, 1 / k), (np.repeat(np.arange(n), k), cols.ravel())),
                             shape=(n, self.Xv.shape[0]))

    def predict(self, obs, measured, **kw):
        A = self.neighbours(obs, measured, **kw)
        return np.asarray((A @ self.Xv).todense(), dtype=np.float32)
