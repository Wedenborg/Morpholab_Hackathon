"""MSE, L1, Pearson and Spearman on aligned expression matrices.

All functions accept dense or scipy sparse inputs. ``axis=None`` compares
flattened entries; ``axis=0`` compares cells for each gene; ``axis=1`` compares
genes for each cell. Correlations for constant vectors are NaN.
"""

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from ._validation import _gene_indices


def _pair(truth, prediction, axis):
    if axis not in (None, 0, 1):
        raise ValueError("axis must be None, 0, or 1.")
    truth = truth.toarray() if sparse.issparse(truth) else np.asarray(truth)
    prediction = prediction.toarray() if sparse.issparse(prediction) else np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim not in (1, 2) or truth.size == 0:
        raise ValueError("truth and prediction must have the same nonempty 1D/2D shape.")
    if truth.ndim == 1 and axis == 1:
        raise ValueError("axis=1 requires two-dimensional inputs.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Evaluation inputs must contain only finite values.")
    return truth.astype(np.float64, copy=False), prediction.astype(np.float64, copy=False)


def mse_loss(truth, prediction, axis=None):
    """Mean squared error over entries along axis."""
    truth, prediction = _pair(truth, prediction, axis)
    return np.mean((truth - prediction) ** 2, axis=axis)


def l1_loss(truth, prediction, axis=None):
    """Mean absolute error (MAE), with the same reduction as mse_loss."""
    truth, prediction = _pair(truth, prediction, axis)
    return np.mean(np.abs(truth - prediction), axis=axis)


def _pearson(truth, prediction, axis):
    truth = truth - np.mean(truth, axis=axis, keepdims=True)
    prediction = prediction - np.mean(prediction, axis=axis, keepdims=True)
    numerator = np.sum(truth * prediction, axis=axis)
    denominator = np.sqrt(np.sum(truth ** 2, axis=axis) * np.sum(prediction ** 2, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator > 0, numerator / denominator, np.nan)
    return np.clip(result, -1, 1)


def pearson(truth, prediction, axis=None):
    """Pearson correlation; undefined constant-vector correlations return NaN."""
    truth, prediction = _pair(truth, prediction, axis)
    return _pearson(truth, prediction, axis)


def spearman(truth, prediction, axis=None):
    """Pearson correlation of average ranks, including correct tie handling."""
    truth, prediction = _pair(truth, prediction, axis)
    truth = rankdata(truth, method="average", axis=axis)
    prediction = rankdata(prediction, method="average", axis=axis)
    return _pearson(truth, prediction, axis)


def evaluate(truth, prediction, axis=None):
    """Compute all four metrics with a common reduction axis."""
    return {name: metric(truth, prediction, axis=axis) for name, metric in METRICS.items()}


METRICS = {"mse": mse_loss, "pearson": pearson, "spearman": spearman, "l1": l1_loss}


def evaluate_batches(truth, predict_batch, sequenced_genes, batch_size=256):
    """Evaluate a fitted imputer without materializing an entire dense N x G.

    ``predict_batch(start, stop)`` returns predictions on the full shared gene
    axis. Report all/shared and currently unmeasured metrics separately. Losses
    average over entries. Correlations average per-cell correlations across
    genes, omitting constant vectors; *_n_valid counts valid cells. This makes
    results independent of batch size. If no unmeasured genes remain, their
    metrics are NaN with zero valid cells. Fitting must happen before this call.
    """
    if len(truth.shape) != 2 or min(truth.shape) < 1:
        raise ValueError("truth must be a nonempty N x G matrix.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    genes = _gene_indices(sequenced_genes, truth.shape[1])
    missing = np.ones(truth.shape[1], dtype=bool)
    missing[genes] = False
    totals = {f"{metric}_{scope}": 0.0 for scope in ("all", "unmeasured") for metric in METRICS}
    counts = dict.fromkeys(totals, 0)
    for start in range(0, truth.shape[0], batch_size):
        stop = min(start + batch_size, truth.shape[0])
        expected, prediction = _pair(truth[start:stop], predict_batch(start, stop), 1)
        for scope, mask in [("all", slice(None)), ("unmeasured", missing)]:
            if scope == "unmeasured" and not missing.any():
                continue
            for metric, values in evaluate(expected[:, mask], prediction[:, mask], axis=1).items():
                key = f"{metric}_{scope}"
                valid = np.isfinite(values)
                totals[key] += values[valid].sum()
                counts[key] += int(valid.sum())
    result = {key: totals[key] / counts[key] if counts[key] else np.nan for key in totals}
    for key in totals:
        if key.startswith(("pearson_", "spearman_")):
            result[f"{key}_n_valid"] = counts[key]
    return result
