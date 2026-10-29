"""What a method sees of the hidden embryo, and what it hands back.

A method gets a MeasurementOracle for the hidden embryo. Through it the method sees the
cells' positions and the gene library, and it can request experiments: up to k genes
(default 4) measured on one section or on any list of cells. Cell types, germ layers and
unmeasured expression stay hidden. Every request is logged.

A submission is two files:
- reconstruction .h5ad: every cell of the hidden embryo x every gene in the library,
  on the same scale as the data's X (log1p of counts);
- experiments .csv: the oracle's log, one row per experiment, with columns
  step, stage, section_z, genes (';'-separated), cells (';'-separated cell names, empty
  when the whole section was measured).

The benchmark uses the log to tell measured from predicted values.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

K_MAX = 4
LOG_COLUMNS = ["step", "stage", "section_z", "genes", "cells"]


class MeasurementOracle:
    def __init__(self, truth, k_max=K_MAX, stage="E7.5"):
        self._truth = truth
        self._X = sp.csr_matrix(truth.X)
        self.k_max = k_max
        self.stage = stage
        self._log = []

    @property
    def cells(self):
        """Cell names and positions (x, y within the section; z = section)."""
        return self._truth.obs[["x", "y", "z"]].copy()

    @property
    def genes(self):
        return pd.Index(self._truth.var_names)

    @property
    def sections(self):
        return np.sort(self._truth.obs["z"].unique())

    def measure(self, genes, section=None, cells=None):
        """Measure `genes` on a whole section (section=z) or on a list of cell names.
        Returns a DataFrame (cells x genes) of log1p values."""
        genes = list(dict.fromkeys(genes))
        if not 1 <= len(genes) <= self.k_max:
            raise ValueError(f"an experiment measures 1 to {self.k_max} genes, got {len(genes)}")
        missing = [g for g in genes if g not in self.genes]
        if missing:
            raise ValueError(f"genes not in the library: {missing}")
        if (section is None) == (cells is None):
            raise ValueError("give exactly one of section or cells")
        obs = self._truth.obs
        if section is not None:
            rows = np.where(obs["z"].to_numpy() == section)[0]
            if len(rows) == 0:
                raise ValueError(f"no section at z={section}")
            cell_field = ""
        else:
            rows = pd.Index(obs.index).get_indexer(list(cells))
            if (rows < 0).any():
                raise ValueError("unknown cell names")
            cell_field = ";".join(obs.index[rows])
        cols = self.genes.get_indexer(genes)
        vals = self._X[rows][:, cols].toarray()
        zs = obs["z"].to_numpy()[rows]
        self._log.append(dict(step=len(self._log), stage=self.stage,
                              section_z=section if section is not None else (zs[0] if len(set(zs)) == 1 else np.nan),
                              genes=";".join(genes), cells=cell_field))
        return pd.DataFrame(vals, index=obs.index[rows], columns=genes)

    def log(self):
        return pd.DataFrame(self._log, columns=LOG_COLUMNS)


def load_log(path):
    return pd.read_csv(path, keep_default_na=False, na_values={"section_z": [""]})


def measured_mask(log, obs, genes):
    """Boolean sparse matrix (cells x genes): True where an experiment in `log` measured
    that gene in that cell."""
    genes = pd.Index(genes)
    cell_index = pd.Index(obs.index)
    z = obs["z"].to_numpy()
    rows, cols = [], []
    for _, e in log.iterrows():
        g = genes.get_indexer([s for s in str(e["genes"]).split(";") if s])
        g = g[g >= 0]
        if isinstance(e["cells"], str) and e["cells"]:
            r = cell_index.get_indexer(e["cells"].split(";"))
            r = r[r >= 0]
        else:
            r = np.where(z == float(e["section_z"]))[0]
        rows.append(np.repeat(r, len(g)))
        cols.append(np.tile(g, len(r)))
    if not rows:
        return sp.csr_matrix((len(obs), len(genes)), dtype=bool)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    m = sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(obs), len(genes)))
    m.sum_duplicates()
    return m.astype(bool)


def check_log(log, k_max=K_MAX):
    """Problems with a submitted log, as a list of strings (empty if fine)."""
    problems = []
    missing = [c for c in LOG_COLUMNS if c not in log.columns]
    if missing:
        return [f"missing columns {missing}"]
    n_genes = log["genes"].astype(str).str.split(";").map(lambda g: len([s for s in g if s]))
    if (n_genes > k_max).any():
        problems.append(f"{(n_genes > k_max).sum()} experiments measure more than {k_max} genes")
    return problems
