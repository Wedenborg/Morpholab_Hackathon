"""Reference-only method; observes no hidden expression outside measured inputs."""

import numpy as np
import pandas as pd

from .mlp import MLPImputer
from .selection import HVGSelector


class HaochenMethod:
    def __init__(self, reference, library_genes, *, selection="detected_hvg", top_k=2000,
                 threshold_pct=50.0, min_cells=10, seed=42, mlp_options=None):
        self.library = pd.Index(library_genes)
        if not self.library.is_unique or not reference.var_names.is_unique:
            raise ValueError("Library and reference gene identifiers must be unique.")
        shared = self.library[self.library.isin(reference.var_names)]
        if len(shared) < 2:
            raise ValueError("Need at least two shared reference/library genes.")
        self.reference = reference[:, shared].copy()
        self.reference.X = self.reference.X.tocsr() if hasattr(self.reference.X, "tocsr") else self.reference.X
        self.shared_columns = self.library.get_indexer(shared)
        self.unsupported_genes = self.library[~self.library.isin(shared)].tolist()
        self.selector = HVGSelector(self.reference, strategy=selection, top_k=top_k,
                                    threshold_pct=threshold_pct, min_cells=min_cells, seed=seed)
        self.mlp_options = {"seed": seed, **(mlp_options or {})}
        # Validate model options before requesting measurements.
        MLPImputer(**self.mlp_options)

    def select(self, measured_genes, n_select):
        return self.selector.select(measured_genes, n_select)

    def fit(self, measured_genes):
        genes = pd.Index(measured_genes)
        positions = self.reference.var_names.get_indexer(genes)
        if not genes.is_unique or np.any(positions < 0):
            raise ValueError("Measured genes must be distinct and supported by the reference.")
        self.panel = genes
        self.panel_columns = self.library.get_indexer(genes)
        self.model = MLPImputer(**self.mlp_options).fit(self.reference.X, positions)
        return self

    def predict(self, observed):
        shared = self.model.predict(observed)
        result = np.zeros((shared.shape[0], len(self.library)),
                          dtype=np.result_type(observed.dtype, np.float32))
        result[:, self.shared_columns] = shared
        return result
