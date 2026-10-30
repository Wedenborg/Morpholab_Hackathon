"""Experimental design on the cell-type model (celltype_model.py).

Experiments are 4 genes on a region: a whole section or a quarter of a section (k-means on
x, y). At an equal number of cell x gene measurements, quarters allow 4x as many
experiments.

Predictive design (`greedy_predictive`)
    The reconstruction is x_i = sum_c q_ic m_c, so confusing states c and c' in a cell costs
    roughly D_cc' = sum_g ((m_cg - m_c'g) / sd_g)^2 in squared prediction error over all genes.
    Measuring gene g separates them with Bhattacharyya distance (Poisson, corrected for
    overdispersion)
        B_g(c,c') = 0.5 (sqrt(mu_cg) - sqrt(mu_c'g))^2 / (1 + mean(mu_cg, mu_c'g) / theta_g),
    so the chance of still confusing them is at most exp(-sum_{g in S} B_g). The expected
    reduction in prediction error from measuring S on region R is then approximated by
        gain(R, S) = sum_{c,c'} W_R[c,c'] D_cc' (1 - prod_{g in S} exp(-B_g(c,c'))),
        W_R = sum_{i in R} q_i q_i^T  (off-diagonal: how much the region's cells are torn
              between c and c').
    Genes are added greedily; the region with the largest gain is measured. q depends on the
    measured values, so the design is adaptive. This targets prediction of all genes rather
    than information about the states themselves (unlike designs.adaptive_celltype).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def section_regions(obs):
    z = obs["z"].to_numpy()
    return {f"z={s:g}": np.where(z == s)[0] for s in np.sort(np.unique(z))}


def quarter_regions(obs, seed=0):
    z = obs["z"].to_numpy()
    xy = obs[["x", "y"]].to_numpy(float)
    out = {}
    for s in np.sort(np.unique(z)):
        idx = np.where(z == s)[0]
        k = min(4, len(idx))
        lab = KMeans(k, n_init=4, random_state=seed).fit_predict(xy[idx])
        for j in range(k):
            out[f"z={s:g}/q{j}"] = idx[lab == j]
    return out


class StateSession:
    def __init__(self, oracle, model, obs, regions, beta=2.0):
        self.oracle, self.model, self.obs, self.regions, self.beta = oracle, model, obs, regions, beta
        self.A = model.graph(obs)
        self.cell_names = pd.Index(obs.index)
        self.measured = {}                     # experiment id -> DataFrame (cells x genes, log1p)
        self.region_genes = {r: set() for r in regions}
        self.q = np.tile(model.pi, (len(obs), 1))

    def run(self, region, genes):
        cells = self.cell_names[self.regions[region]]
        df = self.oracle.measure(list(genes), cells=list(cells))
        self.measured[len(self.measured)] = df
        self.region_genes[region] |= set(genes)
        self.q = self.model.posterior(self.obs, self.measured, beta=self.beta, A=self.A)

    def reconstruct(self, how="mean"):
        if how == "mean":
            return self.model.reconstruct(self.q)
        return self.model.m[self.q.argmax(1)].astype(np.float32)


def rotating(session, gene_sets, n_experiments):
    names, P = list(session.regions), len(gene_sets)
    R = len(names)
    for e in range(n_experiments):
        session.run(names[e % R], gene_sets[(e + e // R) % P])


def _pair_terms(model, min_detect=0.05, n_candidates=300):
    m, mu, th = model.m, model.mu, model.theta
    sd = np.sqrt(np.maximum((model.pi[:, None] * (m - model.pi @ m) ** 2).sum(0), 1e-6))
    keep = model.detect >= 0.02                     # the genes the held-out score uses
    Ms = m[:, keep] / sd[keep]
    sq = (Ms ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None] - 2 * Ms @ Ms.T, 0)                 # C x C
    detect = model.detect
    cand = np.where(detect >= min_detect)[0]
    r = np.sqrt(mu[:, cand])
    avg = 0.5 * (mu[:, None, cand] + mu[None, :, cand])
    B = 0.5 * (r[:, None] - r[None]) ** 2 / (1 + avg / th[cand])            # C x C x cand
    sep = 1 - np.exp(-B)
    prior = np.outer(model.pi, model.pi) * D
    score = np.einsum("cd,cdg->g", prior, sep)
    top = np.argsort(-score)[:n_candidates]
    return D, np.exp(-B[:, :, top]), cand[top]


def greedy_predictive(session, n_experiments, k=4, n_candidates=300):
    model = session.model
    D, keep_conf, cand = _pair_terms(model, n_candidates=n_candidates)       # keep_conf: C x C x cand
    names = list(session.regions)
    gene_names = model.genes
    for _ in range(n_experiments):
        best = (None, -np.inf, None)
        for r in names:
            q = session.q[session.regions[r]]
            W = q.T @ q
            np.fill_diagonal(W, 0)
            WD = W * D
            remaining = np.ones_like(D)
            taken = {j for j, g in enumerate(cand) if gene_names[g] in session.region_genes[r]}
            chosen, total = [], 0.0
            for _ in range(k):
                gain = np.einsum("cd,cdg->g", WD * remaining, 1 - keep_conf)
                gain[list(taken | set(chosen))] = -np.inf
                j = int(np.argmax(gain))
                chosen.append(j)
                total += gain[j]
                remaining = remaining * keep_conf[:, :, j]
            if total > best[1]:
                best = ([gene_names[cand[j]] for j in chosen], total, r)
        session.run(best[2], best[0])
