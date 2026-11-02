"""Measure -> refit -> reconstruct -> score, including the final acquisition."""

import json
from pathlib import Path
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import evaluate_batches


def _write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        return item
    Path(path).write_text(json.dumps(convert(value), indent=2, allow_nan=False) + "\n")


def acquire(oracle, genes, cells):
    """Measure all sections, retaining genuine measured zeros explicitly."""
    genes = list(genes)
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError("Acquisition genes must be nonempty and distinct.")
    result = None
    seen = np.zeros((len(cells), len(genes)), dtype=bool)
    for start in range(0, len(genes), oracle.k_max):
        chunk = genes[start:start + oracle.k_max]
        for section in oracle.sections:
            values = oracle.measure(chunk, section=section)
            rows = cells.get_indexer(values.index)
            if np.any(rows < 0) or not values.index.is_unique:
                raise ValueError("Oracle returned unknown or duplicate cell identifiers.")
            values = values.loc[:, chunk].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Oracle returned nonfinite expression.")
            if result is None:
                result = np.zeros(seen.shape, dtype=np.result_type(values.dtype, np.float32))
            elif not np.can_cast(values.dtype, result.dtype, casting="safe"):
                result = result.astype(np.result_type(values.dtype, result.dtype))
            result[rows, start:start + len(chunk)] = values
            seen[rows, start:start + len(chunk)] = True
    if result is None or not seen.all():
        raise ValueError("Not all cells were covered by the oracle's sections.")
    return result


def write_reconstruction(path, method, observed, cells, *, batch_size):
    """Write dense predictions directly to chunked HDF5, one cell batch at a time."""
    shape = (len(cells), len(method.library))
    skeleton = ad.AnnData(X=sparse.csr_matrix(shape, dtype=np.float32),
                         obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=method.library))
    skeleton.uns["log1p"] = {"base": None}
    skeleton.write_h5ad(path)
    with h5py.File(path, "r+") as handle:
        del handle["X"]
        output = handle.create_dataset("X", shape=shape, dtype=np.result_type(observed.dtype, np.float32),
                                       chunks=(min(batch_size, shape[0]), min(1024, shape[1])))
        output.attrs["encoding-type"] = "array"
        output.attrs["encoding-version"] = "0.2.0"
        for start in range(0, shape[0], batch_size):
            stop = min(start + batch_size, shape[0])
            prediction = method.predict(observed[start:stop])
            if prediction.shape != (stop - start, shape[1]) or not np.isfinite(prediction).all():
                raise ValueError("Invalid reconstruction batch.")
            # Even zero-valued observations pass through unchanged.
            np.testing.assert_array_equal(prediction[:, method.panel_columns], observed[start:stop])
            output[start:stop] = prediction


def run_experiment(method, oracle, output, *, initial_genes, rounds=5, n_select=4,
                   batch_size=256, evaluator=None, diagnostic_truth=None, config=None):
    """evaluator(reconstruction_path, experiment_log) is called only after fitting.

    diagnostic_truth is optional, evaluator-owned, aligned log1p expression used
    exclusively for supplementary entry losses and per-cell correlations.
    """
    for name, value, minimum in [("rounds", rounds, 0), ("n_select", n_select, 1),
                                  ("batch_size", batch_size, 1)]:
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    panel = list(initial_genes)
    if not panel or len(set(panel)) != len(panel):
        raise ValueError("Initial genes must be nonempty and distinct.")
    if not pd.Index(panel).isin(method.reference.var_names).all():
        raise ValueError("Initial genes must match exact reference and library identifiers.")
    if not method.library.equals(pd.Index(oracle.genes)):
        raise ValueError("Method and oracle gene axes differ.")
    available = method.selector.candidates[~method.selector.candidates.isin(panel)]
    if len(available) < rounds * n_select:
        raise ValueError(f"Need {rounds * n_select} acquisition candidates, have {len(available)}.")
    cells = pd.Index(oracle.cells.index)
    if len(cells) == 0 or not cells.is_unique:
        raise ValueError("Oracle cells must be nonempty and unique.")
    if len(oracle.log()):
        raise ValueError("Use a fresh oracle so experiment budgets include exactly this run.")
    if diagnostic_truth is not None and diagnostic_truth.shape != (len(cells), len(method.library)):
        raise ValueError("Diagnostic truth must match the oracle cell/gene axes.")
    output = Path(output)
    # Every run owns a fresh directory; never overwrite existing results.
    output.mkdir(parents=True, exist_ok=False)
    method.selector.table.to_csv(output / "hvg.csv")
    _write_json(output / "config.json", config or {})
    _write_json(output / "reference_coverage.json", {
        "library_genes": len(method.library), "supported_genes": len(method.shared_columns),
        "unsupported_genes": method.unsupported_genes,
        "unsupported_prediction": "zero; excluded from selection and model training",
    })
    observed = acquire(oracle, panel, cells)
    records = []
    for round_number in range(rounds + 1):
        started = time.monotonic()
        if round_number == 0:
            new_genes = panel.copy()
        else:
            new_genes = method.select(panel, n_select)
            observed = np.column_stack([observed, acquire(oracle, new_genes, cells)])
            panel.extend(new_genes)
        log = oracle.log()
        log.to_csv(output / "experiments.csv", index=False)
        print(f"Round {round_number}: {len(panel)} measured genes; fitting MLP", flush=True)
        method.fit(panel)
        path = output / ".working_reconstruction.h5ad"
        write_reconstruction(path, method, observed, cells, batch_size=batch_size)
        scores, tables = evaluator(path, log) if evaluator is not None else ({}, {})
        scores = {"benchmark_" + key: value for key, value in scores.items()}
        if diagnostic_truth is not None:
            with h5py.File(path, "r") as handle:
                scores.update(evaluate_batches(diagnostic_truth, lambda a, b: handle["X"][a:b],
                                               method.panel_columns, batch_size))
        row = {"round": round_number, "selection": method.selector.strategy, "method": "mlp",
               "n_sequenced": len(panel), "n_experiments": len(log),
               "new_genes": json.dumps(new_genes), "panel": json.dumps(panel),
               "mlp_training_loss": method.model.loss_history_[-1] if method.model.loss_history_ else np.nan,
               "runtime_seconds": time.monotonic() - started, **scores}
        records.append(row)
        pd.DataFrame(records).to_csv(output / "history.csv", index=False)
        round_dir = output / f"round_{round_number:02d}"
        round_dir.mkdir()
        log.to_csv(round_dir / "experiments.csv", index=False)
        _write_json(round_dir / "summary.json", row)
        for name, table in tables.items():
            table.to_csv(round_dir / f"{name}.csv")
        print(f"  round {round_number} saved: {len(log)} experiments", flush=True)
    path.rename(output / "reconstruction.h5ad")
    _write_json(output / "summary.json", records[-1])
    return pd.DataFrame(records)
