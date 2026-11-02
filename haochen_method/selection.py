"""Rank reference HVGs and optionally filter strictly by detection prevalence."""

import numpy as np
import pandas as pd

from .reference import rank_sc_hvgs


class HVGSelector:
    def __init__(self, reference, *, strategy="detected_hvg", top_k=2000,
                 threshold_pct=50.0, min_cells=10, seed=42):
        if strategy not in {"detected_hvg", "hvg", "random_hvg"}:
            raise ValueError("Unknown HVG selection strategy.")
        if not np.isfinite(threshold_pct) or not 0 <= threshold_pct <= 100:
            raise ValueError("threshold_pct must be between 0 and 100.")
        ranked, self.table = rank_sc_hvgs(reference, top_k, min_cells)
        self.table = self.table.rename(columns=lambda name: name.replace("_sc", "_reference"))
        self.table = self.table.rename(columns={"sc_detected_cells": "detected_cells_reference"})
        detected = self.table["detected_cells_reference"].to_numpy()
        self.table["detected_fraction_reference"] = detected / reference.n_obs
        keep = detected[ranked] * 100 > threshold_pct * reference.n_obs
        candidates = ranked[keep] if strategy == "detected_hvg" else ranked
        self.candidates = pd.Index(reference.var_names[candidates])
        self.table["acquisition_candidate"] = self.table.index.isin(self.candidates)
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def select(self, measured_genes, n_select):
        if not isinstance(n_select, (int, np.integer)) or n_select < 0:
            raise ValueError("n_select must be a nonnegative integer.")
        available = self.candidates[~self.candidates.isin(measured_genes)]
        if len(available) < n_select:
            raise ValueError(f"Requested {n_select} genes but only {len(available)} candidates remain.")
        if self.strategy == "random_hvg":
            return self.rng.choice(available.to_numpy(), n_select, replace=False).tolist()
        return available[:n_select].tolist()
