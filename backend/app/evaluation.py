from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


_METRIC_INFO = {
    "peng_spearman": (
        "Peng region-pattern Spearman",
        "Higher is better.",
    ),
    "atlas_spearman": (
        "Atlas cell-type-pattern Spearman",
        "Higher is better.",
    ),
    "truth_region_spearman": (
        "Truth region-pattern Spearman",
        "Higher is better; compares reconstructed region averages with this embryo.",
    ),
    "heldout_corr": (
        "Held-out gene correlation",
        "Higher is better; measured genes are excluded using the experiment log.",
    ),
    "celltype_tv": (
        "Cell-type proportion distance",
        "Lower is better.",
    ),
    "celltype_balanced_accuracy": (
        "Cell-type balanced accuracy",
        "Higher is better.",
    ),
}


def evaluation_unavailable(reason: str) -> dict[str, Any]:
    return {
        "label": "E7.5 benchmark not run",
        "synthetic": False,
        "available": False,
        "notice": reason,
        "metrics": [],
    }


def _cumulative_reconstruction(truth: Any, visible: Any, measured_genes: list[str], n_neighbours: int = 20):
    """Nearest-neighbour reconstruction that matches every cell on all measured genes.

    Each cell is matched to its nearest reference cells using every gene measured so far
    (previous rounds plus this one), so adding genes gives the match more to work with.
    Measured values replace predictions where they exist.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    library = truth.var_names
    truth_cols = library.get_indexer(measured_genes)
    visible_cols = pd.Index(visible.var_names).get_indexer(measured_genes)
    shared = (truth_cols >= 0) & (visible_cols >= 0)

    # Reference expression laid out in the target's gene order (zeros for genes it lacks).
    vcols = pd.Index(visible.var_names).get_indexer(library)
    mapper = sp.csr_matrix(
        (np.ones((vcols >= 0).sum()), (vcols[vcols >= 0], np.where(vcols >= 0)[0])),
        shape=(visible.n_vars, len(library)),
    )
    Xv = sp.csr_matrix(visible.X) @ mapper
    Xt = sp.csr_matrix(truth.X)

    if shared.any():
        query = Xt[:, truth_cols[shared]].toarray()
        reference = visible[:, visible_cols[shared]].X
        reference = reference.toarray() if sp.issparse(reference) else np.asarray(reference)
        k = min(n_neighbours, reference.shape[0])
        _, nearest = cKDTree(reference).query(query, k=k)
        nearest = nearest.reshape(len(query), k)
        weights = sp.csr_matrix(
            (np.full(nearest.size, 1.0 / k), (np.repeat(np.arange(len(query)), k), nearest.ravel())),
            shape=(len(query), visible.n_obs),
        )
        recon = (weights @ Xv).tocsr()
    else:
        recon = sp.csr_matrix(Xt.shape)

    keep = truth_cols[truth_cols >= 0]
    if len(keep):
        mask = sp.csr_matrix(
            (np.ones(len(keep) * Xt.shape[0]), (np.repeat(np.arange(Xt.shape[0]), len(keep)), np.tile(keep, Xt.shape[0]))),
            shape=Xt.shape,
        )
        recon = recon - recon.multiply(mask) + Xt.multiply(mask)
    recon.eliminate_zeros()
    return ad.AnnData(X=recon.astype(np.float32), obs=pd.DataFrame(index=truth.obs_names), var=pd.DataFrame(index=library))


def evaluate_mouse_embryo_baseline(
    spatial_path: Path,
    reference_path: Path,
    measured_gene_ids: list[str],
) -> dict[str, Any]:
    """Score a gene panel by reconstructing the E7.5 embryo from it with eval_team."""
    import anndata as ad
    import pandas as pd

    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from eval_team.evaluate import Evaluator
    from eval_team.submission import LOG_COLUMNS

    truth = ad.read_h5ad(spatial_path)
    required_obs = {"x", "y", "z", "celltype", "germ_layer"}
    missing_obs = sorted(required_obs.difference(truth.obs.columns))
    if missing_obs:
        raise ValueError(f"The E7.5 benchmark dataset is missing required obs columns: {', '.join(missing_obs)}")

    available_genes = set(truth.var_names)
    measured_genes = list(dict.fromkeys(gene for gene in measured_gene_ids if gene in available_genes))
    reconstruction = _cumulative_reconstruction(truth, ad.read_h5ad(reference_path), measured_genes)
    log = _section_experiment_log(truth, measured_genes, LOG_COLUMNS)

    evaluator = Evaluator(truth)
    summary, _ = evaluator.evaluate(reconstruction, log)
    return _format_evaluation(summary, evaluator, len(measured_genes), len(log))


def run_eval_team_dummy_method(
    spatial_path: Path,
    reference_path: Path,
    how: str,
    n_sets: int,
    previous_genes: list[str],
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from eval_team import dummy_method
    from eval_team.evaluate import Evaluator
    from eval_team.submission import LOG_COLUMNS, MeasurementOracle

    truth = ad.read_h5ad(spatial_path)
    visible = ad.read_h5ad(reference_path)
    missing_obs = sorted({"x", "y", "z", "celltype", "germ_layer"}.difference(truth.obs.columns))
    if missing_obs:
        raise ValueError(f"The E7.5 benchmark dataset is missing required obs columns: {', '.join(missing_obs)}")
    reconstruction, log = dummy_method.run(
        MeasurementOracle(truth),
        visible,
        n_sets=n_sets,
        how=how,
        seed=seed,
        exclude=previous_genes,
    )

    available_genes = set(truth.var_names)
    prior = list(dict.fromkeys(gene for gene in previous_genes if gene in available_genes))
    if prior:
        prior_log = _section_experiment_log(truth, prior, LOG_COLUMNS)
        prior_log["step"] += int(log["step"].max()) + 1 if len(log) else 0
        log = pd.concat([log, prior_log], ignore_index=True)

    prior_set = set(prior)
    selected_genes = list(dict.fromkeys(
        gene
        for value in log["genes"].astype(str)
        for gene in value.split(";")
        if gene and gene not in prior_set
    ))
    reconstruction = _cumulative_reconstruction(truth, visible, [*prior, *selected_genes])
    evaluator = Evaluator(truth)
    summary, _ = evaluator.evaluate(reconstruction, log)
    evaluation = _format_evaluation(
        summary,
        evaluator,
        len(selected_genes) + len(prior),
        len(log),
    )
    return selected_genes, evaluation


def _section_experiment_log(truth: Any, genes: list[str], columns: list[str]):
    import pandas as pd

    rows = []
    step = 0
    for section_z in sorted(truth.obs["z"].unique()):
        for start in range(0, len(genes), 4):
            rows.append({
                "step": step,
                "stage": "E7.5",
                "section_z": section_z,
                "genes": ";".join(genes[start:start + 4]),
                "cells": "",
            })
            step += 1
    return pd.DataFrame(rows, columns=columns)


def _format_evaluation(
    summary: dict[str, Any],
    evaluator: Any,
    measured_gene_count: int,
    experiment_count: int,
) -> dict[str, Any]:
    metrics = []
    for metric_id, (label, description) in _METRIC_INFO.items():
        value = summary.get(metric_id)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        true_data = summary.get(f"{metric_id}_true_data")
        metrics.append({
            "id": metric_id,
            "label": label,
            "description": description,
            "value": float(value),
            "true_data": float(true_data) if isinstance(true_data, (int, float)) and math.isfinite(true_data) else None,
        })

    reference_names = sorted(evaluator.regions.refs)
    notice = (
        "Computed with the repository's eval_team benchmark. Every cell is matched to its nearest cells "
        "in the reference embryo using all genes measured so far (previous rounds plus this one), and "
        "measured genes keep their observed values; the experiment log excludes measured values from "
        "held-out scoring. This is a retrospective benchmark on the local E7.5 spatial dataset, "
        "not a prospective performance guarantee."
    )
    if not reference_names:
        notice += " Peng and atlas reference tables are not installed, so those external-reference scores are omitted."

    return {
        "label": "eval_team E7.5 reconstruction benchmark",
        "synthetic": False,
        "available": True,
        "notice": notice,
        "metrics": metrics,
        "summary": {
            key: value
            for key, value in summary.items()
            if isinstance(value, (str, int, float, bool))
            and (not isinstance(value, float) or math.isfinite(value))
        },
        "measured_gene_count": measured_gene_count,
        "experiment_count": experiment_count,
        "external_references": reference_names,
    }
