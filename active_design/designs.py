"""Experimental designs on top of the reconstruction model.

Every design spends a fixed number of experiments. One experiment = up to 4 genes on one
whole section, requested through eval_team's MeasurementOracle (so it is logged). After
each experiment the posterior is updated with the section's unit averages.

    fixed_panel         the same genes on every section (k genes -> ceil(k/4) per section)
    rotating            sections take turns through sets of 4 marker genes
    greedy_information  planned: next experiment = section + 4 genes with the largest
                        expected information gain about the factors (log det criterion,
                        genes added one at a time). Does not depend on measured values.
    adaptive_celltype   adaptive: next experiment = section + 4 genes that best separate the
                        cell types each unit could still be, given what was measured so far.
"""
import anndata as ad
import numpy as np
import pandas as pd

from .model import UnitPosterior


class Session:
    """One design run on the hidden embryo: experiments, posterior, reconstruction."""

    def __init__(self, oracle, model, Q, units, verbose=False):
        self.oracle, self.model, self.units = oracle, model, units
        self.post = UnitPosterior(Q, model.K)
        cells = oracle.cells
        self.cell_z = cells["z"].to_numpy()
        self.sections = oracle.sections
        self.section_units = {s: np.unique(units[self.cell_z == s]) for s in self.sections}
        self.gene_pos = {g: i for i, g in enumerate(model.genes)}
        self.measured = {}            # section -> DataFrame (cells x genes measured there)
        self.n_experiments = 0
        self.verbose = verbose

    def run(self, section, genes):
        df = self.oracle.measure(list(genes), section=section)
        self.n_experiments += 1
        prev = self.measured.get(section)
        self.measured[section] = df if prev is None else prev.join(df[[g for g in df.columns if g not in prev]])
        gi = np.array([self.gene_pos[g] for g in genes])
        u = self.units[self.cell_z == section]
        y = pd.DataFrame(df.to_numpy(), index=u).groupby(level=0).mean()
        self.post.update(y.index.to_numpy(), self.model.W[gi], self.model.r[gi], y.to_numpy() - self.model.b[gi])
        if self.verbose:
            print(f"  exp {self.n_experiments:3d}: z={section:g} genes={list(genes)}")

    def reconstruct(self):
        m, W, b = self.model, self.model.W, self.model.b
        Z = self.post.unit_mean()[self.units].copy()                # cells x K
        cell_index = pd.Index(self.oracle.cells.index)
        for section, df in self.measured.items():
            gi = np.array([self.gene_pos[g] for g in df.columns])
            rows = cell_index.get_indexer(df.index)
            u_cells = self.units[rows]
            Ws = W[gi]
            for u in np.unique(u_cells):
                sel = u_cells == u
                C = self.post.unit_cov(u) + m.Sigma_delta
                gain = C @ Ws.T @ np.linalg.inv(Ws @ C @ Ws.T + np.diag(m.r_cell[gi]))
                resid = df.to_numpy()[sel] - b[gi] - Z[rows[sel]] @ Ws.T
                Z[rows[sel]] += resid @ gain.T
        X = np.clip(Z @ W.T + b, 0, None).astype(np.float32)
        for section, df in self.measured.items():
            rows = cell_index.get_indexer(df.index)
            X[np.ix_(rows, [self.gene_pos[g] for g in df.columns])] = df.to_numpy()
        return ad.AnnData(X=X, obs=pd.DataFrame(index=cell_index), var=pd.DataFrame(index=m.genes))


# ----------------------------------------------------------------------------- designs
def fixed_panel(session, genes):
    for s in session.sections:
        for i in range(0, len(genes), session.oracle.k_max):
            session.run(s, genes[i:i + session.oracle.k_max])


def rotating(session, gene_sets, n_experiments):
    S, P = len(session.sections), len(gene_sets)
    for e in range(n_experiments):
        session.run(session.sections[e % S], gene_sets[(e + e // S) % P])


def _candidates(session, n):
    m = session.model
    score = np.where(m.detect >= 0.05, m.informativeness(), 0)
    return np.argsort(-score)[:n]


def _measured_positions(session, section, cand):
    """Positions in `cand` of genes already measured on this section (re-measuring returns
    the same values, so it would be counted twice)."""
    df = session.measured.get(section)
    if df is None:
        return []
    have = {session.gene_pos[g] for g in df.columns}
    return [i for i, c in enumerate(cand) if c in have]


def _greedy_section_info(session, section, cand, k):
    """Greedy 4-gene choice for one section by information gain. Returns (genes, gain)."""
    post, m = session.post, session.model
    units = session.section_units[section]
    n, K = len(units), m.K
    idx = post.idx(units)
    Sig = post.Sigma[np.ix_(idx, idx)].copy()
    Wc, rc = m.W[cand], m.r[cand]
    done = _measured_positions(session, section, cand)
    chosen, total = [], 0.0
    for _ in range(k):
        T = Sig.reshape(n, K, n, K)
        C = np.einsum("gk,akbl,gl->gab", Wc, T, Wc, optimize=True)   # g x n x n
        M = np.eye(n)[None] + C / rc[:, None, None]
        gain = 0.5 * np.linalg.slogdet(M)[1]
        gain[done] = -np.inf
        best = int(np.argmax(gain))
        done.append(best)
        chosen.append(cand[best])
        total += gain[best]
        H = np.kron(np.eye(n), Wc[best][None])                      # n x nK
        SH = Sig @ H.T
        Sig -= SH @ np.linalg.solve(H @ SH + rc[best] * np.eye(n), SH.T)
    return [m.genes[g] for g in chosen], total


def greedy_information(session, n_experiments, n_candidates=300, candidates=None):
    """candidates: optional list of gene names to choose from (default: the most
    informative well-detected genes)."""
    cand = (np.array([session.gene_pos[g] for g in candidates]) if candidates is not None
            else _candidates(session, n_candidates))
    k = session.oracle.k_max
    for _ in range(n_experiments):
        best = max((_greedy_section_info(session, s, cand, k) + (s,) for s in session.sections),
                   key=lambda t: t[1])
        session.run(best[2], best[0])


def _type_probs(session, units):
    """Posterior probability of each visible-embryo cell type for each unit."""
    m, post = session.model, session.post
    mean = post.unit_mean()
    out = np.empty((len(units), len(m.type_names)))
    for i, u in enumerate(units):
        C = m.type_cov + post.unit_cov(u)
        Ci = np.linalg.inv(C)
        d = m.type_means - mean[u]
        out[i] = np.log(m.type_prior) - 0.5 * np.einsum("ck,kl,cl->c", d, Ci, d)
    out = np.exp(out - out.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


def adaptive_celltype(session, n_experiments, n_candidates=300):
    """Pick the section and genes that most reduce the chance of confusing the cell types
    each unit could still be (a Bhattacharyya bound on pairwise confusion)."""
    m = session.model
    cand = _candidates(session, n_candidates)
    Wc = m.W[cand]
    diff = m.type_means[:, None, :] - m.type_means[None, :, :]                  # C x C x K
    proj = np.einsum("gk,cdk->gcd", Wc, diff) ** 2                              # g x C x C
    spread = m.r[cand] + np.einsum("gk,kl,gl->g", Wc, m.type_cov, Wc)
    sep = 1 - np.exp(-proj / (8 * spread[:, None, None]))                       # g x C x C
    iu = np.triu_indices(len(m.type_names), 1)
    sep = sep[:, iu[0], iu[1]]                                                  # g x pairs
    k = session.oracle.k_max
    for _ in range(n_experiments):
        best = (None, -1.0, None)
        for s in session.sections:
            p = _type_probs(session, session.section_units[s])
            conf = (p[:, iu[0]] * p[:, iu[1]]).sum(0)                           # pairs
            chosen, total = [], 0.0
            done = _measured_positions(session, s, cand)
            for _ in range(k):
                score = sep @ conf
                score[chosen + done] = -np.inf
                g = int(np.argmax(score))
                chosen.append(g)
                total += score[g]
                conf = conf * (1 - sep[g])
            if total > best[1]:
                best = ([m.genes[cand[g]] for g in chosen], total, s)
        session.run(best[2], best[0])
